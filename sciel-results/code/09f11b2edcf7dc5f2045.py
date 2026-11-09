import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n = len(primes)
    
    # Initial M
    M = 10000
    counts = np.zeros(M + 1, dtype=np.int64)
    residues = np.zeros(n, dtype=np.int64)

    # Initialize with 0s
    for i, p in enumerate(primes):
        residues[i] = 0
        counts[p::p] += 1

    def calc_L(counts, M):
        if counts[1] < 2:
            return 0
        # Find first index < 2
        # Using numpy to find the first element < 2 in counts[1:]
        # np.where returns indices
        # If no such index, return M
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx

    def extend_M(counts, residues, M, new_M):
        new_counts = np.zeros(new_M + 1, dtype=np.int64)
        new_counts[:M + 1] = counts
        for j, pj in enumerate(primes):
            rj = residues[j]
            # Find first index in [M+1, new_M] congruent to rj mod pj
            if rj == 0:
                start_m = (M + 1 + pj - 1) // pj
            else:
                start_m = max(0, (M + 1 - rj + pj - 1) // pj)
            first_idx = rj + start_m * pj
            if first_idx <= new_M:
                new_counts[first_idx::pj] += 1
        return new_counts

    def update_counts_and_L(counts, residues, i, r_old, r_new, p, current_L, M):
        """
        Updates counts for a move and returns new_L, and potentially extended M/counts.
        """
        if r_old == r_new:
            return current_L, M, counts
        
        # Apply move to counts
        if r_old == 0:
            counts[p::p] -= 1
        else:
            counts[r_old::p] -= 1
            
        if r_new == 0:
            counts[p::p] += 1
        else:
            counts[r_new::p] += 1
            
        # Determine new_L
        # Check if we broke the prefix
        # We only need to check if any x <= current_L has count < 2
        # Specifically, if we decremented a count at x <= current_L to < 2
        
        # Efficient check:
        # If r_old != 0:
        #   indices = np.arange(r_old, current_L + 1, p)
        #   if np.any(counts[indices] < 2):
        #       # We broke it. Find new L by scanning back from current_L
        #       # Or just recalc L from scratch (expensive if M is large, but we can optimize)
        #       # Let's find the first gap in [1, current_L]
        #       mask = counts[1:current_L+1] < 2
        #       if np.any(mask):
        #           new_L = np.argmax(mask)
        #       else:
        #           new_L = current_L
        #       # Then try to extend from new_L
        #       while new_L < M and counts[new_L + 1] >= 2:
        #           new_L += 1
        #   else:
        #       # Didn't break. Try to extend from current_L
        #       new_L = current_L
        #       while new_L < M and counts[new_L + 1] >= 2:
        #           new_L += 1
        # Else (r_old == 0):
        #   indices = np.arange(p, current_L + 1, p)
        #   ... same logic

        # To simplify and ensure correctness, if the move *could* have broken the prefix, we recalc L.
        # "Could have broken" means we decremented a count at some x <= current_L.
        # This happens if there exists x <= current_L such that x % p == r_old (and r_old != r_new).
        
        broke_prefix = False
        if r_old != 0:
            if r_old <= current_L:
                # There is at least one x in [1, current_L] with x % p == r_old
                # We need to check if any of them dropped to < 2
                # We can just check the min count in that range? No, we need to know if ANY is < 2.
                # Since we only decremented these, if any was 2, it is now 1.
                # So if any was 2, we broke.
                # We can check: min(counts[indices]) < 2
                # But we just decremented them.
                # So if min(counts[indices]) < 2, it means it was <= 1 before, so it was 0 or 1?
                # Wait. If it was 2, it is now 1. 1 < 2.
                # If it was 1, it is now 0. 0 < 2.
                # If it was 3, it is now 2. 2 is not < 2.
                # So we broke if min(counts[indices]) < 2.
                idxs = np.arange(r_old, current_L + 1, p)
                if np.min(counts[idxs]) < 2:
                    broke_prefix = True
        else:
            if p <= current_L:
                idxs = np.arange(p, current_L + 1, p)
                if np.min(counts[idxs]) < 2:
                    broke_prefix = True

        if broke_prefix:
            # Recalculate L from scratch (or scan back)
            # Scanning back is faster if drop is small
            new_L = current_L
            while new_L >= 1 and counts[new_L] < 2:
                new_L -= 1
            # Now try to extend
            while new_L < M and counts[new_L + 1] >= 2:
                new_L += 1
        else:
            # Did not break. Try to extend.
            new_L = current_L
            while new_L < M and counts[new_L + 1] >= 2:
                new_L += 1

        # Check if we hit the boundary and might be able to extend further
        if new_L >= M - 10:
            new_M = M * 2
            counts = extend_M(counts, residues, M, new_M)
            M = new_M
            # Recalc L with new M
            new_L = calc_L(counts, M)

        return new_L, M, counts

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()

    # Greedy Repair Phase
    # Try to fix gaps from 1 to M
    # We do this for a few iterations or until time budget
    repair_deadline = start_time + 5.0
    while time.time() < repair_deadline:
        # Find first gap
        if current_L >= M:
            break
        gap_x = current_L + 1
        if counts[gap_x] >= 2:
            # Should have extended L, recalc
            current_L = calc_L(counts, M)
            continue
            
        # Need to increase counts[gap_x]
        # Find a prime p to change
        # Candidates: primes where r_p != gap_x % p
        # We want to pick one that doesn't break [1, current_L]
        
        # Shuffle primes
        perm = np.random.permutation(n)
        moved = False
        for i in perm:
            p = primes[i]
            r_old = residues[i]
            r_new = gap_x % p
            
            if r_old == r_new:
                continue
            
            # Check if move breaks prefix
            # Similar to update_counts_and_L logic
            safe = True
            if r_old != 0:
                if r_old <= current_L:
                    idxs = np.arange(r_old, current_L + 1, p)
                    if np.min(counts[idxs]) < 3: # If min is 2, it drops to 1. If min is 1, drops to 0.
                        # We need min to be >= 3 to be safe?
                        # If min is 2, it becomes 1. 1 < 2. Breaks.
                        # If min is 1, it becomes 0. Breaks.
                        # So we need min >= 3.
                        safe = False
            else:
                if p <= current_L:
                    idxs = np.arange(p, current_L + 1, p)
                    if np.min(counts[idxs]) < 3:
                        safe = False
            
            if safe:
                # Apply move
                if r_old == 0:
                    counts[p::p] -= 1
                else:
                    counts[r_old::p] -= 1
                    
                if r_new == 0:
                    counts[p::p] += 1
                else:
                    counts[r_new::p] += 1
                    
                residues[i] = r_new
                
                # Update L
                # Since we fixed the gap, we might extend
                # Check if we broke prefix (we checked safe, so no)
                new_L = current_L
                while new_L < M and counts[new_L + 1] >= 2:
                    new_L += 1
                    
                if new_L >= M - 10:
                    new_M = M * 2
                    counts = extend_M(counts, residues, M, new_M)
                    M = new_M
                    new_L = calc_L(counts, M)
                    
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                moved = True
                break
        
        if not moved:
            # Stuck in repair, break to MCMC
            break

    # MCMC Phase
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9999
    stuck_counter = 0
    
    while time.time() < deadline:
        i = random.randint(0, n - 1)
        p = primes[i]
        r_old = residues[i]
        r_new = random.randint(0, p - 1)

        if r_new == r_old:
            continue

        new_L, M, counts = update_counts_and_L(counts, residues, i, r_old, r_new, p, current_L, M)
        
        if new_L >= current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                stuck_counter = 0
        else:
            delta_L = new_L - current_L
            if random.random() < math.exp(delta_L / T):
                current_L = new_L
                residues[i] = r_new
            else:
                # Revert
                # Re-apply old move
                if r_new == 0:
                    counts[p::p] -= 1
                else:
                    counts[r_new::p] -= 1
                if r_old == 0:
                    counts[p::p] += 1
                else:
                    counts[r_old::p] += 1
                # residues[i] stays r_old
                # Recalc L? No, it reverts to previous state.
                # But we might have extended M.
                # If we extended M, we can't easily revert M.
                # So we just keep the extended M.
                # current_L is restored to previous value?
                # The revert restores counts to previous state (for the window).
                # So current_L should be the old current_L.
                pass

        T = max(T * cooling_rate, T_min)
        stuck_counter += 1
        if stuck_counter > 500:
            T = 10.0
            stuck_counter = 0

    return best_residues.tolist()

