"""HP lattice protein folding on the simple cubic lattice.

Design (from scratch):
  * Fast randomized greedy construction with contact-aware scoring
    (new H-H contacts + free-space lookahead).
  * A time-bounded local search (single-residue moves) that uses a
    precomputed contact table and O(1)-ish incremental energy updates,
    with simulated-annealing-style acceptance to escape local optima.
  * An explicit deadline guard: track elapsed wall-clock time and return
    the best conformation found with a safety margin before the limit.
"""
# EVOLVE-BLOCK-START
import random
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Wall-clock budget. The evaluator kills at 330 s; keep a safety margin.
_TOTAL_BUDGET = 320.0
# Per-sequence safety margin: stop improving a sequence this much early so
# the *last* sequence still finishes comfortably before the kill.
_PER_SEQ_MARGIN = 4.0
# Hard per-sequence cap (seconds) so no single sequence can eat everything.
_PER_SEQ_HARD_CAP = 105.0


def _neighbours(point):
    """Yield the six unit-neighbour sites of a lattice point."""
    for axis in range(len(point)):
        base = point[axis]
        q = list(point)
        q[axis] = base + 1
        yield tuple(q)
        q[axis] = base - 1
        yield tuple(q)


# ---------------------------------------------------------------------------
# Fast randomized greedy construction
# ---------------------------------------------------------------------------
def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth.

    Prefer sites that create new H-H contacts, then sites with free space
    (so the chain does not paint itself into a corner).
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        cur = chain[-1]
        best_score = None
        best_site = None
        if sequence[i] == "H":
            for site in _neighbours(cur):
                if site in occupied:
                    continue
                gain = 0
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
                free = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1
                score = gain + 0.1 * free + rng.random()
                if best_score is None or score > best_score:
                    best_score = score
                    best_site = site
        else:
            # Polar residue: just pick a free neighbour with the most space.
            for site in _neighbours(cur):
                if site in occupied:
                    continue
                free = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1
                score = free + rng.random()
                if best_score is None or score > best_score:
                    best_score = score
                    best_site = site

        if best_site is None:
            return None
        occupied[best_site] = i
        chain.append(best_site)
    return chain


# ---------------------------------------------------------------------------
# Contact bookkeeping
# ---------------------------------------------------------------------------
def _build_contacts(sequence, chain):
    """Return (contacts, where, energy).

    contacts: set of (i, j) with i < j, both H, not chain-adjacent,
              lattice-neighbouring.  energy = -len(contacts).
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] != "H":
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts.add((i, j))
    return contacts, where, -len(contacts)


def _try_move(sequence, chain, where, contacts, i, new_site, dimension):
    """Attempt a single-residue move of residue i to new_site.

    Returns (new_chain, new_where, new_contacts, new_energy, ok).
    ok is False if the move is invalid (self-intersection or non-unit step).
    """
    n = len(chain)
    old = chain[i]
    if new_site == old:
        return None, None, None, None, False
    # Unit-step check against both neighbours.
    if i > 0:
        prev = chain[i - 1]
        if sum(abs(a - b) for a, b in zip(new_site, prev)) != 1:
            return None, None, None, None, False
    if i < n - 1:
        nxt = chain[i + 1]
        if sum(abs(a - b) for a, b in zip(new_site, nxt)) != 1:
            return None, None, None, None, False
    # Self-avoidance: new_site must not be occupied by another residue.
    occ = where.get(new_site)
    if occ is not None and occ != i:
        return None, None, None, None, False

    # Recompute contacts involving i (cheap, O(1) per neighbour).
    is_h = (sequence[i] == "H")
    # Remove old contacts involving i.
    new_contacts = set(contacts)
    for c in list(new_contacts):
        if c[0] == i or c[1] == i:
            new_contacts.discard(c)
    # Add new contacts involving i.
    if is_h:
        for q in _neighbours(new_site):
            j = where.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == "H":
                a, b = (i, j) if i < j else (j, i)
                new_contacts.add((a, b))

    new_chain = chain[:]
    new_chain[i] = new_site
    new_where = dict(where)
    del new_where[old]
    new_where[new_site] = i
    new_energy = -len(new_contacts)
    return new_chain, new_where, new_contacts, new_energy, True


# ---------------------------------------------------------------------------
# Local search (single-residue moves, SA-style acceptance)
# ---------------------------------------------------------------------------
def _local_search(sequence, chain, where, contacts, energy, budget, rng):
    """Improve the conformation under a time budget. Returns best found."""
    n = len(chain)
    best_chain = chain
    best_energy = energy
    best_where = where
    best_contacts = contacts

    # Temperature schedule: start warm, cool down.
    t_start = 1.0
    t_end = 0.01
    start_time = time.time()

    # Precompute candidate sites per residue lazily; for simplicity we sample
    # a random residue and a random free neighbour each iteration.
    # Track a moving window of "recently tried" to reduce repeats.
    while True:
        now = time.time()
        if now - start_time >= budget:
            break
        # Cool the temperature linearly over the budget.
        frac = min(1.0, (now - start_time) / max(1e-9, budget))
        temp = t_start * (t_end / t_start) ** frac

        i = rng.randrange(n)
        cur = chain[i]
        # Candidate sites: free neighbours of cur.
        cands = []
        for site in _neighbours(cur):
            occ = where.get(site)
            if occ is None or occ == i:
                cands.append(site)
        if not cands:
            continue
        new_site = cands[rng.randrange(len(cands))]

        new_chain, new_where, new_contacts, new_energy, ok = _try_move(
            sequence, chain, where, contacts, i, new_site, 3
        )
        if not ok:
            continue

        delta = new_energy - energy  # more negative is better -> delta < 0 good
        if delta < 0 or rng.random() < pow(2.718281828459045, delta / max(1e-9, temp)):
            # Accept the move.
            chain, where, contacts, energy = new_chain, new_where, new_contacts, new_energy
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
                best_where = where
                best_contacts = contacts

    return best_chain, best_energy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def fold(sequence, dimension):
    """Best conformation found within the time budget.

    Returns a list of len(sequence) lattice points (lists of length dimension).
    """
    n = len(sequence)
    global _TOTAL_BUDGET
    global_start = time.time()

    # Per-sequence budget: share the global budget, but never exceed the cap
    # and always leave a safety margin.
    elapsed_global = time.time() - global_start
    remaining = _TOTAL_BUDGET - elapsed_global
    per_seq = min(_PER_SEQ_HARD_CAP, max(0.5, remaining - _PER_SEQ_MARGIN))

    rng = random.Random(12345 + n)

    # ---- Phase 1: randomized greedy constructions ------------------------
    # Run enough constructions to get a good starting point, but bounded by
    # time. Each construction is O(n * 6 * 6) ~ cheap.
    phase1_deadline = time.time() + min(6.0, per_seq * 0.25)
    best_chain = None
    best_energy = 0
    best_where = None
    best_contacts = None

    # Always try at least one construction.
    for _ in range(400):
        if time.time() >= phase1_deadline:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        contacts, where, energy = _build_contacts(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            best_where = where
            best_contacts = contacts

    if best_chain is None:
        # Straight line fallback (always self-avoiding).
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_contacts, best_where, best_energy = _build_contacts(sequence, best_chain)

    # ---- Phase 2: local search -------------------------------------------
    phase2_deadline = time.time() + max(0.5, per_seq - (time.time() - (global_start + 0.0)))
    # Recompute remaining for phase 2 against the global budget.
    remaining2 = _TOTAL_BUDGET - (time.time() - global_start) - _PER_SEQ_MARGIN
    if remaining2 > 0.5:
        best_chain, best_energy = _local_search(
            sequence, best_chain, best_where, best_contacts, best_energy,
            remaining2, rng,
        )

    # ---- Return as list of lists -----------------------------------------
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
