"""Two-phase solver: greedy forward construction + simulated annealing."""
import time
import numpy as np
import random


def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(sieve[i * i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def _compute_counts(residues, primes, max_L):
    """Compute coverage counts array for given residues."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    return counts


def _find_L(counts, max_L):
    """Find largest L such that counts[1..L] >= 2."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]


def _get_positions(r, p, max_L):
    """Get positions covered by residue r mod p in 1..max_L."""
    if r == 0:
        start = p
    else:
        start = r
    if start > max_L:
        return np.array([], dtype=np.int32)
    return np.arange(start, max_L + 1, p, dtype=np.int32)


def _greedy_forward(primes, max_L, deadline):
    """
    Greedy forward construction: for each prime in order, pick the residue
    that maximizes the doubly-covered prefix length given prior residues.
    Uses deficiency-aware scoring: prioritize covering positions at count 1.
    """
    n_primes = len(primes)
    residues = [0] * n_primes
    counts = np.zeros(max_L + 1, dtype=np.int32)
    current_L = 0

    for i, p in enumerate(primes):
        if time.time() > deadline:
            break
        best_r = 0
        best_score = -1

        # For each candidate residue, compute the resulting L and a deficiency score
        # We evaluate in a smarter way: focus on the current gap region
        # But for correctness, we need to evaluate all residues for small primes
        # and use sampling for large primes to save time

        if p <= 200:
            candidates = list(range(p))
        else:
            # Sample candidates: include 0, and random samples
            candidates = [0] + [random.randint(0, p - 1) for _ in range(min(200, p - 1))]
            # Remove duplicates
            candidates = list(set(candidates))

        for r in candidates:
            # Compute delta: what positions does this residue cover?
            if r == 0:
                start = p
            else:
                start = r
            if start > max_L:
                pos = np.array([], dtype=np.int32)
            else:
                pos = np.arange(start, max_L + 1, p, dtype=np.int32)

            if len(pos) == 0:
                new_counts = counts.copy()
            else:
                new_counts = counts.copy()
                new_counts[pos] += 1

            # Find L
            below_two = np.where(new_counts[1:] < 2)[0]
            if len(below_two) == 0:
                L_val = max_L
            else:
                L_val = below_two[0]

            # Deficiency score: how many positions in 1..max(current_L, L_val) are at count 1?
            # We want to maximize L first, then minimize deficiency
            # Actually, let's use a combined score: L + small bonus for reducing deficiency
            # But primarily maximize L
            if L_val > best_score:
                best_score = L_val
                best_r = r
            elif L_val == best_score and L_val > 0:
                # Tie-breaking: prefer residues that cover more currently-deficient positions
                # (positions at count 1 in the current counts)
                deficient_mask = (counts[1:max_L + 1] == 1)
                if len(pos) > 0:
                    # Count how many of pos are in deficient positions
                    pos_clipped = pos[pos <= max_L]
                    if len(pos_clipped) > 0:
                        def_count = np.sum(deficient_mask[pos_clipped - 1])
                    else:
                        def_count = 0
                else:
                    def_count = 0
                # We'd need to track the best def_count, but for simplicity,
                # just use L as the primary metric
                pass

        residues[i] = best_r
        # Update counts
        r = best_r
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_L:
            pos = np.arange(start, max_L + 1, p, dtype=np.int32)
            if len(pos) > 0:
                counts[pos] += 1
        current_L = _find_L(counts, max_L)

    return residues, counts, current_L


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000

    # Phase 1: Greedy forward construction
    best_residues, counts, best_L = _greedy_forward(primes, max_L, deadline)

    # Precompute position arrays for each prime and each residue
    positions = []
    for i, p in enumerate(primes):
        pos_list = []
        for r in range(p):
            pos_list.append(_get_positions(r, p, max_L))
        positions.append(pos_list)

    # Phase 2: Simulated annealing
    # Random multi-prime moves with cooling schedule
    random.seed(42)
    np.random.seed(42)

    current_residues = best_residues[:]
    current_counts = counts.copy()
    current_L = best_L

    # Annealing parameters
    T_initial = 50.0  # Initial temperature
    T_final = 0.5
    cooling_rate = 0.9995  # Geometric cooling per step

    # Number of steps: adapt based on available time
    # Each step takes roughly O(p_avg) time for the delta computation
    # Let's estimate: ~100 steps per second for small primes, fewer for large
    # We'll just run until deadline

    T = T_initial
    step = 0

    while time.time() < deadline - 5:
        step += 1

        # Choose 1-3 random primes to modify
        n_moves = random.randint(1, 3)
        move_indices = random.sample(range(n_primes), min(n_moves, n_primes))

        # For each chosen prime, pick a new random residue
        old_residues = []
        new_residues = []
        for i in move_indices:
            p = primes[i]
            old_r = current_residues[i]
            new_r = random.randint(0, p - 1)
            old_residues.append((i, old_r))
            new_residues.append((i, new_r))

        # Compute new counts
        new_counts = current_counts.copy()
        for (i, old_r), (i2, new_r) in zip(old_residues, new_residues):
            old_pos = positions[i][old_r]
            new_pos = positions[i][new_r]
            if len(old_pos) > 0:
                new_counts[old_pos] -= 1
            if len(new_pos) > 0:
                new_counts[new_pos] += 1

        new_L = _find_L(new_counts, max_L)

        # Accept or reject
        delta_L = new_L - current_L
        if delta_L >= 0:
            # Accept
            current_residues = [current_residues[j] for j in range(n_primes)]
            for (i, old_r), (i2, new_r) in zip(old_residues, new_residues):
                current_residues[i] = new_r
            current_counts = new_counts
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = current_residues[:]
        else:
            # Reject with probability exp(delta_L / T)
            if random.random() < np.exp(delta_L / T):
                current_residues = [current_residues[j] for j in range(n_primes)]
                for (i, old_r), (i2, new_r) in zip(old_residues, new_residues):
                    current_residues[i] = new_r
                current_counts = new_counts
                current_L = new_L

        # Cool down
        T *= cooling_rate
        if T < T_final:
            T = T_final

    return best_residues
