import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin

    N = 256

    # 1. Precompute ALL APs
    ap_indices = []
    aps_containing = [[] for _ in range(N)]

    for d in range(1, N + 1):
        for s in range(1, N + 1):
            base_indices = []
            curr = s - 1
            while curr < N:
                base_indices.append(curr)
                curr += d
            
            for L in range(1, len(base_indices) + 1):
                idx_list = base_indices[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # Build incidence matrix A (num_aps x N) where A[r, i] = 1 if i in AP r
    A = np.zeros((num_aps, N), dtype=np.float32)
    for r, idx_list in enumerate(ap_indices):
        for i in idx_list:
            A[r, i] = 1.0

    # Convert aps_containing to numpy arrays
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]

    # 2. Thue-Morse initialization
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    # 3. Try SAT-based approach first
    def try_sat(target_discrepancy, time_budget):
        """Try to find assignment with max AP sum <= target_discrepancy using SAT"""
        try:
            from pysat.solvers import Glucose3
            from pysat.formula import CNF
        except ImportError:
            return None, None
        
        deadline = time.time() + time_budget
        
        # For each AP, we need: -D <= sum <= D
        # This is hard to encode directly. Instead, use a different approach:
        # Binary search on D and use local search as fallback.
        
        # Actually, let's use a simpler approach: try to minimize using local search
        # with better initialization from SAT if possible.
        
        # For now, just return None to indicate SAT not used effectively
        return None, None

    # 4. Optimized local search with BLAS
    def optimize(s_init, time_budget, rng):
        s = s_init.copy().astype(np.float32)  # Use float32 for BLAS
        sums = A @ s  # BLAS matrix-vector product
        
        abs_sums = np.abs(sums)
        current_max = int(np.max(abs_sums))
        
        best_s = s.copy()
        best_score = current_max
        
        # Precompute: for each position i, which APs are affected
        # aps_containing_np[i] gives the list of AP indices
        
        indices_list = list(range(N))
        
        # Track counts of each absolute sum value
        max_possible = N + 10
        counts = np.bincount(abs_sums.astype(np.int32), minlength=max_possible)
        
        # Main optimization loop
        while time.time() - start_time < start_time + time_budget:
            if time.time() - start_time > time_budget + start_time:
                break
                
            rng.shuffle(indices_list)
            improved = False
            
            for i in indices_list:
                if time.time() - start_time > time_budget + start_time:
                    break
                
                old_val = s[i]
                delta = -2.0 * old_val
                new_val = -old_val
                
                affected = aps_containing_np[i]
                
                # Simulate flip: sums[affected] += delta
                old_sums_affected = sums[affected].copy()
                new_sums_affected = old_sums_affected + delta
                
                # Check if any exceeds current_max
                max_new = int(np.max(np.abs(new_sums_affected)))
                if max_new > current_max:
                    continue
                
                # Update sums
                sums[affected] = new_sums_affected
                s[i] = new_val
                
                # Update counts
                old_abs = np.abs(old_sums_affected).astype(np.int32)
                new_abs = np.abs(new_sums_affected).astype(np.int32)
                
                changed = old_abs != new_abs
                if np.any(changed):
                    np.add.at(counts, old_abs[changed], -1)
                    np.add.at(counts, new_abs[changed], 1)
                    
                    # Check if max decreased
                    if counts[current_max] == 0:
                        new_max = current_max
                        while new_max > 0 and counts[new_max] == 0:
                            new_max -= 1
                        current_max = new_max
                        
                        if current_max < best_score:
                            best_score = current_max
                            best_s = s.copy()
                            improved = True
        
        return best_s, best_score

    rng = random.Random(42)
    
    # Try SAT-based approach first (may not be available)
    sat_s, sat_score = try_sat(10, 5.0)
    if sat_s is not None and sat_score is not None:
        best_s = sat_s
        best_score = sat_score
    else:
        # Fall back to local search
        tm = thue_morse(N)
        s1, score1 = optimize(tm, time_limit * 0.3, rng)
        best_s = s1
        best_score = score1

    # Run more local search with different initializations
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        rand_init = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8).astype(np.float32)
        s2, score2 = optimize(rand_init, remaining * 0.4, rng)
        if score2 < best_score:
            best_score = score2
            best_s = s2

    # Final polish from best solution
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        s3, score3 = optimize(best_s, remaining - 0.5, rng)
        if score3 < best_score:
            best_score = score3
            best_s = s3

    return [int(x) for x in best_s]
