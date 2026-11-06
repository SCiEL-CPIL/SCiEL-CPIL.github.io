"""HP lattice protein folding: contact-map-guided backtracking with local search polish."""
# EVOLVE-BLOCK-START
import random
import math
import time
from collections import defaultdict

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 200.0  # seconds, safety margin before 210s kill


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: minus number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _build_contact_priority(sequence):
    """
    Identify high-priority H-H pairs based on sequence distance.
    Pairs that are far apart in sequence but both H are harder to satisfy
    via local greedy moves, so we prioritize them.
    Returns a list of (i, j, priority) sorted by priority descending.
    """
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    pairs = []
    n = len(h_indices)
    for a in range(n):
        for b in range(a + 1, n):
            i, j = h_indices[a], h_indices[b]
            dist = abs(i - j)
            # Higher priority for pairs that are farther apart (harder to satisfy locally)
            # but not too far (impossible to be in contact)
            if 2 <= dist <= min(20, len(sequence) // 2):
                priority = dist  # farther = higher priority
                pairs.append((i, j, priority))
    pairs.sort(key=lambda x: -x[2])
    return pairs


def _greedy_grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search_polish(sequence, chain, max_steps=500):
    """
    Local search: try single-residue moves and reversals to improve energy.
    Returns improved chain.
    """
    n = len(chain)
    if n < 4:
        return chain

    def _chain_valid(chain):
        """Check self-avoidance and unit steps."""
        seen = set()
        for i, p in enumerate(chain):
            if p in seen:
                return False
            seen.add(p)
            if i > 0:
                prev = chain[i - 1]
                dist = sum(abs(p[a] - prev[a]) for a in range(len(p)))
                if dist != 1:
                    return False
        return True

    def _try_move(chain, pos, new_site):
        """Try moving residue at pos to new_site."""
        old_site = chain[pos]
        if new_site == old_site:
            return None
        # Check new_site is not occupied by other residues
        for i, p in enumerate(chain):
            if i != pos and p == new_site:
                return None
        # Check adjacency with neighbors
        if pos > 0:
            prev = chain[pos - 1]
            if sum(abs(new_site[a] - prev[a]) for a in range(len(new_site))) != 1:
                return None
        if pos < n - 1:
            nxt = chain[pos + 1]
            if sum(abs(new_site[a] - nxt[a]) for a in range(len(new_site))) != 1:
                return None
        # Check self-avoidance
        new_chain = chain[:pos] + [new_site] + chain[pos + 1:]
        if not _chain_valid(new_chain):
            return None
        return new_chain

    def _try_reversal(chain, start, end):
        """Try reversing segment from start to end."""
        if end - start < 2:
            return None
        new_chain = chain[:start] + chain[start:end + 1][::-1] + chain[end + 1:]
        if not _chain_valid(new_chain):
            return None
        return new_chain

    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    steps = 0
    while steps < max_steps and _time_left() > 5.0:
        improved = False
        # Try single moves
        for pos in range(1, n - 1):
            for site in _neighbours(best_chain[pos]):
                new_chain = _try_move(best_chain, pos, site)
                if new_chain is not None:
                    e = _energy(sequence, new_chain)
                    if e < best_energy:
                        best_chain = new_chain
                        best_energy = e
                        improved = True
                        steps += 1
                        break
            if improved:
                break
        if improved:
            continue
        # Try reversals
        for length in range(2, min(6, n)):
            for start in range(1, n - length):
                new_chain = _try_reversal(best_chain, start, start + length - 1)
                if new_chain is not None:
                    e = _energy(sequence, new_chain)
                    if e < best_energy:
                        best_chain = new_chain
                        best_energy = e
                        improved = True
                        steps += 1
                        break
            if improved:
                break
        if not improved:
            break
    return best_chain


def _contact_guided_backtrack(sequence, dimension, priority_pairs, rng, max_depth=10):
    """
    Backtracking search that prioritizes satisfying high-priority H-H contacts.
    Grows the chain but at each step considers whether placing an H residue
    near a previously placed H residue satisfies a priority pair.
    """
    n = len(sequence)
    # Precompute which priority pairs involve which residues
    pair_by_residue = defaultdict(list)
    for i, j, pri in priority_pairs:
        pair_by_residue[i].append((j, pri))
        pair_by_residue[j].append((i, pri))

    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    satisfied_pairs = set()

    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                # Count new H-H contacts
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
                            # Bonus if this satisfies a priority pair
                            if (min(i, j), max(i, j)) in {(a, b) for a, b, _ in priority_pairs}:
                                gain += 2
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.05 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        # Sort by score and pick from top few with some randomness
        options.sort(key=lambda x: -x[0])
        top_k = min(3, len(options))
        site = rng.choice([opt[1] for opt in options[:top_k]])
        occupied[site] = i
        chain.append(site)

    return chain


def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation.
    Uses multiple strategies: greedy growth, contact-guided backtracking,
    and local search polishing.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]

    rng = random.Random(12345 + len(sequence))

    # Build contact priority map
    priority_pairs = _build_contact_priority(sequence)

    best_chain = None
    best_energy = 0

    # Strategy 1: Multiple greedy growths
    num_greedy = 100
    for _ in range(num_greedy):
        if _time_left() < 10.0:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain

    # Strategy 2: Contact-guided backtracking
    num_bt = 50
    for _ in range(num_bt):
        if _time_left() < 10.0:
            break
        chain = _contact_guided_backtrack(sequence, dimension, priority_pairs, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain

    # Strategy 3: Local search polish on best chain
    if best_chain is not None:
        if _time_left() > 15.0:
            polished = _local_search_polish(sequence, best_chain, max_steps=300)
            e = _energy(sequence, polished)
            if e < best_energy:
                best_energy = e
                best_chain = polished

    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
