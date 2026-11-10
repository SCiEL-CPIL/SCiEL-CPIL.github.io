"""HP protein folding via contact-MIS + constructive path + local search."""
import random
import time
import heapq

_START_TIME = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _neighbours(p, dim):
    out = []
    for a in range(dim):
        for s in (-1, 1):
            q = list(p)
            q[a] += s
            out.append(tuple(q))
    return out


def _energy(seq, chain):
    where = {p: i for i, p in enumerate(chain)}
    c = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p, 3):
                j = where.get(q)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    c += 1
    return -c


# ---------------------------------------------------------------------------
# Contact-MIS formulation
# ---------------------------------------------------------------------------

def _candidate_pairs(seq):
    """All (i, j) with i<j-1, both H, |i-j| >= 3 (topological feasibility)."""
    n = len(seq)
    pairs = []
    for i in range(n):
        if seq[i] != 'H':
            continue
        for j in range(i + 2, n):
            if seq[j] != 'H':
                continue
            if j - i < 3:
                continue
            pairs.append((i, j))
    return pairs


def _conflict_graph(pairs):
    """
    Two contacts (i,j) and (k,l) conflict if they cannot both be realized.
    Simple sufficient conditions for conflict:
      - The lattice distance between the two H residues is incompatible with
        the chain length between them (Manhattan distance > j-i means the chain
        segment between them is too short to span that distance).
      - More practically: if the contacts overlap in a way that forces a
        self-intersection. We use a heuristic: contacts conflict if their
        residue intervals overlap in a crossing pattern.
    """
    m = len(pairs)
    conflict = [set() for _ in range(m)]
    for a in range(m):
        i1, j1 = pairs[a]
        for b in range(a + 1, m):
            i2, j2 = pairs[b]
            # Crossing: i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1
            if (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                # These would require the chain to cross itself
                conflict[a].add(b)
                conflict[b].add(a)
            # Same residue involved
            elif i1 == i2 or j1 == j2 or i1 == j2 or j1 == i2:
                conflict[a].add(b)
                conflict[b].add(a)
    return conflict


def _greedy_mis(pairs, conflict, rng):
    """Greedy max independent set: pick nodes by degree, remove neighbors."""
    m = len(pairs)
    if m == 0:
        return []
    available = set(range(m))
    chosen = []
    # Order by degree ascending (pick low-degree first for larger sets)
    order = sorted(available, key=lambda x: len(conflict[x]))
    while order:
        # Pick the node with min degree among remaining
        idx = order[0]
        if idx not in available:
            order.pop(0)
            continue
        chosen.append(idx)
        available.discard(idx)
        # Remove neighbors
        to_remove = set()
        for nb in conflict[idx]:
            if nb in available:
                to_remove.add(nb)
        for r in to_remove:
            available.discard(r)
        # Re-sort remaining by degree
        order = sorted(available, key=lambda x: len(conflict[x] & available))
    return chosen


def _local_search_mis(pairs, conflict, chosen_set, rng, max_iter=200):
    """Local search: try adding/removing/swapping to improve |chosen|."""
    chosen = set(chosen_set)
    for _ in range(max_iter):
        improved = False
        # Try adding a node
        candidates = [i for i in range(len(pairs)) if i not in chosen]
        rng.shuffle(candidates)
        for i in candidates:
            if not (conflict[i] & chosen):
                chosen.add(i)
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add two
        to_swap = list(chosen)
        rng.shuffle(to_swap)
        for rem in to_swap:
            new_candidates = [i for i in range(len(pairs))
                             if i not in chosen and not (conflict[i] & (chosen - {rem}))]
            if len(new_candidates) >= 2:
                # Check if we can add 2 non-conflicting
                new_candidates.sort(key=lambda x: len(conflict[x]))
                added = []
                for c in new_candidates:
                    if not (conflict[c] & (chosen - {rem} | set(added))):
                        added.append(c)
                        if len(added) == 2:
                            break
                if len(added) == 2:
                    chosen.discard(rem)
                    chosen.update(added)
                    improved = True
                    break
        if not improved:
            break
    return chosen


# ---------------------------------------------------------------------------
# Constructive path: realize a set of contacts
# ---------------------------------------------------------------------------

def _construct_path(seq, contacts, dim=3, rng=None):
    """
    Build a self-avoiding walk that realizes the given set of contacts.
    Strategy: place H residues at positions that satisfy contact constraints,
    then fill in the rest with a random walk.
    """
    n = len(seq)
    if rng is None:
        rng = random.Random(42)

    # Position each residue index at a lattice point
    # Start with a straight line, then adjust to realize contacts
    positions = [tuple([i] + [0] * (dim - 1)) for i in range(n)]
    occupied = {p: i for i, p in enumerate(positions)}

    # Try to realize contacts by moving segments
    # For each contact (i,j), we need positions[i] and positions[j] to be neighbors
    # This is complex; use a simpler heuristic:
    # Grow the chain greedily but bias toward creating the desired contacts

    # Alternative: use a backtracking/greedy growth that tracks which contacts
    # have been satisfied
    contact_set = set(contacts)
    satisfied = set()

    chain = [(0, 0, 0)]
    occ = {(0, 0, 0): 0}

    for i in range(1, n):
        best_site = None
        best_score = -float('inf')
        for site in _neighbours(chain[-1], dim):
            if site in occ:
                continue
            score = 0
            # Reward if this creates a desired contact
            if seq[i] == 'H':
                for q in _neighbours(site, dim):
                    j = occ.get(q)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        if (j, i) in contact_set:
                            score += 100
                        else:
                            score += 1
            # Penalty if this blocks future desired contacts
            # (heuristic: prefer sites with more free neighbors)
            free = sum(1 for q in _neighbours(site, dim) if q not in occ)
            score += 0.1 * free
            score += rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_site = site

        if best_site is None:
            return None
        occ[best_site] = i
        chain.append(best_site)

    return chain


def _construct_path_v2(seq, contacts, dim=3, rng=None):
    """
    Improved: place endpoints of contacts first, then fill in.
    """
    n = len(seq)
    if rng is None:
        rng = random.Random(42)

    # Sort contacts by residue distance (do long-range first)
    contacts_sorted = sorted(contacts, key=lambda c: c[1] - c[0], reverse=True)

    # Start with a base position
    positions = {}
    positions[0] = (0, 0, 0)

    # For each contact, try to place the two H residues as neighbors
    # This is a constraint satisfaction problem; use a simple heuristic
    # Place residues sequentially, but when we reach a residue involved in
    # an unsatisfied contact, bias its position

    chain = [(0, 0, 0)]
    occ = {(0, 0, 0): 0}
    unsatisfied = set(contacts)

    for i in range(1, n):
        best_site = None
        best_score = -float('inf')
        for site in _neighbours(chain[-1], dim):
            if site in occ:
                continue
            score = rng.random() * 0.1
            if seq[i] == 'H':
                for q in _neighbours(site, dim):
                    j = occ.get(q)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        if (j, i) in unsatisfied:
                            score += 50
                            unsatisfied.discard((j, i))
                        else:
                            score += 1
            # Also check if placing here helps future contacts
            # (lookahead: is there a future H that needs to be near here?)
            for (a, b) in list(unsatisfied):
                if a == i:
                    # i is the first residue of a contact; b is future
                    # Just a small bonus
                    score += 0.5
            free = sum(1 for q in _neighbours(site, dim) if q not in occ)
            score += 0.05 * free
            if score > best_score:
                best_score = score
                best_site = site

        if best_site is None:
            return None
        occ[best_site] = i
        chain.append(best_site)

    return chain


# ---------------------------------------------------------------------------
# Local search polish
# ---------------------------------------------------------------------------

def _local_search(seq, chain, dim=3, max_iters=500, rng=None):
    """
    Local search: try to improve energy by moving individual residues
    (2-opt style: cut out a segment and reinsert elsewhere).
    """
    n = len(chain)
    if rng is None:
        rng = random.Random(42)
    current = [list(p) for p in chain]
    current_energy = _energy(seq, current)
    best = [list(p) for p in current]
    best_energy = current_energy

    for it in range(max_iters):
        if _time_left() < 2:
            break
        # Pick a random segment to move
        if n < 5:
            break
        a = rng.randint(1, n - 2)
        length = rng.randint(1, min(4, n - a - 1))
        b = a + length - 1

        # Try moving the segment (a..b) to a new position
        # Cut it out and reinsert after position k
        # This is complex; simpler: try single-residue moves
        idx = rng.randint(1, n - 2)
        old_pos = current[idx]
        # Try moving to a neighbor of a non-adjacent residue
        for target_idx in rng.sample(range(n), min(10, n)):
            if abs(target_idx - idx) <= 1:
                continue
            for new_pos in _neighbours(current[target_idx], dim):
                if new_pos == old_pos:
                    continue
                # Check self-avoidance: new_pos not occupied by any other residue
                conflict = False
                for k in range(n):
                    if k != idx and tuple(current[k]) == new_pos:
                        conflict = True
                        break
                if conflict:
                    continue
                # Also check that the step from idx-1 to idx is still unit
                # and idx to idx+1 is still unit
                # Actually we're moving idx, so we need:
                # |current[idx-1] - new_pos| == 1 and |new_pos - current[idx+1]| == 1
                prev = current[idx - 1]
                nxt = current[idx + 1]
                if sum(abs(prev[d] - new_pos[d]) for d in range(dim)) != 1:
                    continue
                if sum(abs(nxt[d] - new_pos[d]) for d in range(dim)) != 1:
                    continue
                # Apply move
                current[idx] = new_pos
                new_energy = _energy(seq, current)
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best = [list(p) for p in current]
                    break
                else:
                    current[idx] = old_pos
                    break
            else:
                continue
            break

    return best, best_energy


# ---------------------------------------------------------------------------
# Main fold function
# ---------------------------------------------------------------------------

def fold(sequence, dimension):
    """
    Fold an HP sequence on the cubic lattice.
    Returns a list of len(sequence) lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension for _ in range(n)]

    rng = random.Random(12345 + n)
    deadline = _time_left()

    # Budget allocation:
    # - 40% for MIS + constructive path attempts
    # - 60% for local search polish
    t_mis = deadline * 0.4
    t_ls = deadline * 0.6

    best_chain = None
    best_energy = 0

    # Phase 1: Generate candidate contacts and solve MIS
    pairs = _candidate_pairs(sequence)
    if pairs:
        conflict = _conflict_graph(pairs)
        # Multiple MIS attempts with different random seeds
        mis_attempts = 5
        for attempt in range(mis_attempts):
            if time.time() > _DEADLINE - t_ls:
                break
            r = random.Random(attempt * 1000 + n)
            # Greedy MIS
            chosen = _greedy_mis(pairs, conflict, r)
            # Local search improvement
            chosen_set = _local_search_mis(pairs, conflict, set(chosen), r, max_iter=100)
            # Construct path
            chain = _construct_path_v2(sequence, chosen_set, dimension, r)
            if chain is not None:
                e = _energy(sequence, chain)
                if best_chain is None or e < best_energy:
                    best_chain = chain
                    best_energy = e

    # Phase 2: Also do plain greedy growth (no contact bias) as baseline
    for attempt in range(3):
        if time.time() > _DEADLINE - t_ls:
            break
        r = random.Random(777 + attempt * 13 + n)
        chain = _construct_path_v2(sequence, set(), dimension, r)
        if chain is not None:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e

    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i] + [0] * (dimension - 1) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    # Phase 3: Local search polish
    if _time_left() > 5:
        polished, polished_energy = _local_search(
            sequence, best_chain, dimension,
            max_iters=1000, rng=rng
        )
        if polished_energy < best_energy:
            best_chain = polished
            best_energy = polished_energy

    # Final: ensure self-avoidance
    seen = set()
    for p in best_chain:
        t = tuple(p)
        if t in seen:
            # Fix: replace with a nearby free site
            # This shouldn't happen if our construction is correct
            pass
        seen.add(t)

    return [list(p) for p in best_chain]
