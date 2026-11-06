# EVOLVE-BLOCK-START
import time
import random
import math
import sys

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 185.0  # leave 25s buffer for overhead/sandbox


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _contact_potential(sequence, i, j):
    """
    Estimate the value of forming a contact between residues i and j.
    Higher is better. Only long-range H-H pairs matter.
    """
    if sequence[i] != 'H' or sequence[j] != 'H':
        return 0
    if abs(i - j) <= 1:
        return 0
    # Prefer longer-range contacts (harder to satisfy, higher impact)
    dist = abs(i - j)
    # Normalize: max dist is n-2
    # We just use dist as a proxy for difficulty/value
    return dist


def _greedy_backtrack(sequence, dimension, rng, max_time=0.5, max_depth=100000):
    """
    Constraint-guided backtracking search.
    Places residues one by one, prioritizing moves that:
      1. Form new H-H contacts with already-placed residues.
      2. Keep many free neighbours (avoid dead-ends).
      3. Are consistent with high-potential future contacts (heuristic).
    Uses a time budget and random restarts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    start_time = time.time()
    best_chain = None
    best_energy = 0  # 0 contacts

    # Pre-compute high-priority contact pairs
    # We'll use a simple heuristic: for each H, its best potential partners
    # For speed, we just count how many H-H neighbours a site would form

    # Multiple restarts with different random seeds/strategies
    restarts = 0
    while _remaining() > 2.0 and time.time() - start_time < max_time * 10:  # Allow multiple passes
        restarts += 1
        # Vary the strategy slightly each restart
        rng_restart = random.Random(42 + len(sequence) * 1000 + restarts)

        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        current_contacts = 0

        # For backtracking, we'd need to store the full state. 
        # For simplicity and speed, we do a greedy walk with lookahead, 
        # but with a "commitment" to high-value contacts.
        # If stuck, we restart.

        success = True
        for i in range(1, n):
            if _remaining() < 2.5:
                success = False
                break

            last = chain[-1]
            res = sequence[i]
            options = []

            for site in _neighbours(last):
                if site in occupied:
                    continue

                # Count new contacts
                gain = 0
                if res == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # Lookahead: how many free neighbours does this site have?
                free = sum(1 for q in _neighbours(site) if q not in occupied)

                # Future potential: does this site align with other H's that are close in sequence?
                # This is expensive, so we use a simpler heuristic:
                # Prefer sites that are "close" to other H residues in the remaining sequence?
                # Too expensive. Stick to gain + free.

                # Weighting: Strong preference for contacts, then free space
                # Add randomness to explore different paths
                noise = rng_restart.random() * 0.5
                score = gain * 5.0 + 0.2 * free + noise

                options.append((score, site))

            if not options:
                success = False
                break

            # Sort by score descending
            options.sort(key=lambda x: -x[0])

            # Choose top option, but with some probability choose 2nd or 3rd to explore
            if len(options) > 1 and rng_restart.random() < 0.3:
                # Pick from top 3
                top_k = min(3, len(options))
                # Weighted random choice among top_k
                weights = [1.0 / (k + 1) for k in range(top_k)]
                r = rng_restart.random()
                cum = 0.0
                chosen = options[0][1]
                for k in range(top_k):
                    cum += weights[k]
                    if r <= cum:
                        chosen = options[k][1]
                        break
                site = chosen
            else:
                site = options[0][1]

            occupied[site] = i
            chain.append(site)

        if success:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)

            # If we found a good solution, we can stop early if it's optimal?
            # We don't know the optimum, so we keep searching until time runs out.
            # But if we have very little time left, stop.
            if _remaining() < 5.0:
                break

        # If we found a perfect score (theoretical max), we could stop, but we don't know it.
        # So we just loop until time is up.

    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return best_chain


def _local_search(sequence, chain, dimension, rng, max_iter=50):
    """
    Improve a valid chain via local moves: single residue moves.
    Only accepts improving moves (hill climbing) with occasional random kicks.
    """
    if chain is None:
        return None, 0

    current_energy = _energy(sequence, chain)
    current_chain = list(chain)
    n = len(current_chain)

    # Precompute position lookup
    pos_map = {p: i for i, p in enumerate(current_chain)}

    for it in range(max_iter):
        if _remaining() < 1.0:
            break

        best_delta = 0
        best_i = -1
        best_new_pos = None

        # Try moving each residue (randomly sample a subset for speed if n is large)
        # For n <= 100, trying all is fine
        for i in range(1, n - 1):
            p = current_chain[i]
            for q in _neighbours(p):
                if q == current_chain[i - 1] or q == current_chain[i + 1]:
                    continue
                if q in pos_map and pos_map[q] != i:
                    continue

                # Check unit steps (implied by neighbours, but q must be adjacent to i-1 and i+1)
                # Actually, if we move i to q, we need q to be adjacent to i-1 and i+1.
                # But q is a neighbour of p. p is adjacent to i-1 and i+1.
                # q might not be adjacent to i-1 or i+1.
                # Example: i-1 is (0,0), p is (1,0), i+1 is (1,1).
                # Neighbours of p: (2,0), (0,0), (1,1), (1,-1).
                # If q=(2,0), dist to i-1(0,0) is 2. Invalid.
                # So we must check.
                prev = current_chain[i - 1]
                nxt = current_chain[i + 1]
                if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                    continue
                if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                    continue

                # Compute energy delta
                delta = 0
                if sequence[i] == 'H':
                    # Remove old contacts
                    for old_q in _neighbours(p):
                        j = pos_map.get(old_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta += 1
                    # Add new contacts
                    for new_q in _neighbours(q):
                        j = pos_map.get(new_q)
                        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                            delta -= 1

                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_pos = q

        if best_delta < 0 and best_i >= 0:
            old_p = current_chain[best_i]
            del pos_map[old_p]
            current_chain[best_i] = best_new_pos
            pos_map[best_new_pos] = best_i
            current_energy += best_delta
        else:
            # No improving move. Try a random move to escape local minima (simulated annealing style)
            if rng.random() < 0.5:
                i = rng.randint(1, n - 2)
                p = current_chain[i]
                moves = []
                for q in _neighbours(p):
                    if q == current_chain[i - 1] or q == current_chain[i + 1]:
                        continue
                    if q in pos_map and pos_map[q] != i:
                        continue
                    prev = current_chain[i - 1]
                    nxt = current_chain[i + 1]
                    if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                        continue
                    if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                        continue
                    moves.append(q)

                if moves:
                    q = rng.choice(moves)
                    delta = 0
                    if sequence[i] == 'H':
                        for old_q in _neighbours(p):
                            j = pos_map.get(old_q)
                            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                                delta += 1
                        for new_q in _neighbours(q):
                            j = pos_map.get(new_q)
                            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                                delta -= 1

                    # Accept with probability exp(-delta/T)
                    T = 2.0 * (max_iter - it) / max_iter + 0.1
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        old_p = current_chain[i]
                        del pos_map[old_p]
                        current_chain[i] = q
                        pos_map[q] = i
                        current_energy += delta

    return current_chain, current_energy


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Uses constraint-guided backtracking/greedy growth followed by local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 0

    # Allocate time budget based on sequence length
    # Shorter sequences can afford more restarts
    if n <= 25:
        search_time_per_seq = 10.0
    elif n <= 50:
        search_time_per_seq = 8.0
    else:
        search_time_per_seq = 6.0

    # Phase 1: Greedy/Backtracking search
    start = time.time()
    while time.time() - start < search_time_per_seq and _remaining() > 5.0:
        chain = _greedy_backtrack(sequence, dimension, rng, max_time=search_time_per_seq / 5)
        if chain is None:
            break

        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)

        # Phase 2: Local search improvement
        if _remaining() < 5.0:
            break

        improved_chain, improved_energy = _local_search(sequence, chain, dimension, rng, max_iter=30)
        if improved_chain is not None and improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain

    # If no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
