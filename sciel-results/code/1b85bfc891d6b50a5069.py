"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate primes up to 1000 using sieve
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)

    # Precompute coverage arrays for each prime: cover[i][r] = boolean mask of length MAX_L+1
    # where cover[i][r][x] = True if x % p_i == r
    MAX_L = 3000
    cover = np.zeros((n_primes, 1000, MAX_L + 1), dtype=np.bool_)
    for i, p in enumerate(primes):
        for r in range(p):
            # x in [1, MAX_L] such that x % p == r
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= MAX_L:
                cover[i, r, first::p] = True
        # Note: for r=0, first=p, so cover[i,0,p::p]=True
        # For r>0, first=r, so cover[i,r,r::p]=True (r, r+p, ...)
        # But we need to make sure r>=1 for r>0 case. Since r in 1..p-1, r>=1.
        # For r=0, we set first=p, so multiples of p starting from p.

    def compute_L(residues, check_limit=MAX_L):
        """Compute largest L such that every integer 1..L has coverage >= 2.
        Uses precomputed cover arrays for speed."""
        residues_arr = np.array(residues, dtype=np.int64)
        # Sum up covers: for each prime i, add cover[i, residues[i], :]
        counts = np.zeros(check_limit + 1, dtype=np.int32)
        for i in range(n_primes):
            counts += cover[i, residues_arr[i], :check_limit+1].astype(np.int32)
        
        # Find largest L where counts[1..L] are all >= 2
        # Use cumsum of (counts < 2) to find first gap
        gap_mask = (counts[1:check_limit+1] < 2).astype(np.int32)
        if gap_mask.size == 0:
            return check_limit
        first_gap_idx = np.argmax(gap_mask)
        if gap_mask[first_gap_idx] == 0:
            return check_limit
        return first_gap_idx  # L = number of consecutive integers from 1 that are covered >= 2

    def evaluate(residues):
        """Quick evaluation of L for given residues."""
        return compute_L(residues)

    # Initial assignment: try several strategies
    best_residues = None
    best_L = 0

    # Strategy 1: Seed's approach
    residues1 = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    L1 = evaluate(residues1)
    if L1 > best_L:
        best_L = L1
        best_residues = residues1[:]

    # Strategy 2: All zeros
    residues2 = [0] * n_primes
    L2 = evaluate(residues2)
    if L2 > best_L:
        best_L = L2
        best_residues = residues2[:]

    # Strategy 3: r_p = 1 for all
    residues3 = [1 % p for p in primes]
    L3 = evaluate(residues3)
    if L3 > best_L:
        best_L = L3
        best_residues = residues3[:]

    # Strategy 4: Random initializations
    random.seed(42)
    for _ in range(10):
        residues_rand = [random.randint(0, p - 1) for p in primes]
        Lr = evaluate(residues_rand)
        if Lr > best_L:
            best_L = Lr
            best_residues = residues_rand[:]

    # Local search / hill climbing with incremental updates
    if best_residues is not None:
        current = best_residues[:]
        current_L = best_L

        # Precompute for incremental updates: maintain counts array
        # When we change residue of prime i from old_r to new_r, we can update counts
        # by subtracting cover[i, old_r, :] and adding cover[i, new_r, :]
        
        def get_counts(residues):
            """Compute full counts array for given residues."""
            counts = np.zeros(MAX_L + 1, dtype=np.int32)
            for i in range(n_primes):
                counts += cover[i, residues[i], :].astype(np.int32)
            return counts

        def find_L_from_counts(counts):
            """Find largest L from a counts array."""
            gap_mask = (counts[1:MAX_L+1] < 2).astype(np.int32)
            if gap_mask.size == 0:
                return MAX_L
            first_gap_idx = np.argmax(gap_mask)
            if gap_mask[first_gap_idx] == 0:
                return MAX_L
            return first_gap_idx

        # Maintain counts for current residues
        counts = get_counts(current)
        current_L = find_L_from_counts(counts)
        if current_L > best_L:
            best_L = current_L
            best_residues = current[:]

        no_improve_count = 0
        max_no_improve = 5000
        random.seed(42)

        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            # Pick a random coordinate to try changing
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = current[i]

            # Try a few new values
            improved = False
            for _ in range(3):
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue

                # Incremental update
                counts -= cover[i, old_r, :].astype(np.int32)
                counts += cover[i, new_r, :].astype(np.int32)
                current[i] = new_r

                new_L = find_L_from_counts(counts)
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    break
                else:
                    # Revert
                    counts -= cover[i, new_r, :].astype(np.int32)
                    counts += cover[i, old_r, :].astype(np.int32)
                    current[i] = old_r

            if improved:
                no_improve_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current[:]
            else:
                no_improve_count += 1

        # More aggressive: try all residues for each prime in order of "importance"
        # Primes with smaller p have more frequent coverage, so they matter more
        if time.time() < deadline - 10.0:
            # Sort primes by size (smaller primes first - they cover more integers)
            order = sorted(range(n_primes), key=lambda i: primes[i])

            for _ in range(3):  # Multiple passes
                if time.time() > deadline - 10.0:
                    break
                for i in order:
                    if time.time() > deadline - 10.0:
                        break
                    p = primes[i]
                    old_r = current[i]
                    best_r = old_r
                    best_local_L = current_L

                    # Try a subset of residues (can't try all if p is large)
                    if p <= 50:
                        candidates = list(range(p))
                    else:
                        candidates = [old_r] + [random.randint(0, p-1) for _ in range(20)]

                    for new_r in candidates:
                        if new_r == old_r:
                            continue
                        # Incremental update
                        counts -= cover[i, old_r, :].astype(np.int32)
                        counts += cover[i, new_r, :].astype(np.int32)
                        current[i] = new_r

                        new_L = find_L_from_counts(counts)
                        if new_L > best_local_L:
                            best_local_L = new_L
                            best_r = new_r
                        else:
                            # Revert to old_r for next candidate
                            counts -= cover[i, new_r, :].astype(np.int32)
                            counts += cover[i, old_r, :].astype(np.int32)
                            current[i] = old_r

                    # Set to best_r
                    if best_r != old_r:
                        counts -= cover[i, old_r, :].astype(np.int32)
                        counts += cover[i, best_r, :].astype(np.int32)
                        current[i] = best_r
                        current_L = best_local_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = current[:]
                    else:
                        # Ensure we're back at old_r
                        current[i] = old_r

    return best_residues if best_residues is not None else [((i + 2) // 2) % p for i, p in enumerate(primes)]
# EVOLVE-BLOCK-END
