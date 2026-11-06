import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 88.0  # Safety margin for 90s limit

    N = 256

    # 1. Precompute ALL APs and inverse mapping
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
    
    # Precompute a flat affected-AP structure for faster per-step updates
    all_affected_flat = []
    affected_offsets = [0] * N
    for i in range(N):
        affected_offsets[i] = len(all_affected_flat)
        all_affected_flat.extend(aps_containing[i])
    affected_offsets.append(len(all_affected_flat))
    all_affected_flat = np.array(all_affected_flat, dtype=np.int32)
    affected_lengths = np.array([affected_offsets[i+1] - affected_offsets[i] for i in range(N)], dtype=np.int32)

    # 2. Initialization Functions
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def random_init(N, rng):
        return np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    def perturbed_init(base_s, rng, num_flips):
        s = base_s.copy()
        for _ in range(num_flips):
            i = rng.randint(0, N - 1)
            s[i] = -s[i]
        return s

    # Vectorized sum computation
    def compute_sums(s, ap_indices):
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices):
            sums[r] = np.sum(s[idx_list])
        return sums

    # 3. Core Optimization with Parallel Tempering
    def optimize_pt(s_init, time_budget, rng):
        # Number of replicas (temperatures)
        num_replicas = 4
        # Temperature schedule: T[0] is lowest (coldest, focuses on energy minimization)
        # T[k] = T0 * (T_max/T0)^(k/(num_replicas-1))
        T0 = 0.1
        T_max = 2.0
        temperatures = np.zeros(num_replicas)
        for k in range(num_replicas):
            temperatures[k] = T0 * (T_max / T0) ** (k / (num_replicas - 1))
        
        # Initialize replicas
        # Replica 0: best initial solution
        # Replica 1-3: perturbed versions
        replicas_s = [s_init.copy()]
        for k in range(1, num_replicas):
            replicas_s.append(perturbed_init(s_init, rng, rng.randint(2, 5)))
        
        replicas_sums = [compute_sums(replicas_s[k], ap_indices) for k in range(num_replicas)]
        
        # Track best solution found across all replicas
        best_s = s_init.copy()
        best_score = int(np.max(np.abs(replicas_sums[0])))
        
        # Compute initial max for each replica
        replica_maxs = [int(np.max(np.abs(replicas_sums[k]))) for k in range(num_replicas)]
        
        indices_list = list(range(N))
        
        # Main loop
        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                # Try to flip position i in all replicas
                old_vals = np.array([replicas_s[k][i] for k in range(num_replicas)], dtype=np.int8)
                deltas = -2 * old_vals
                new_vals = -old_vals
                
                # Compute new sums for affected APs in each replica
                a_start = affected_offsets[i]
                a_end = affected_offsets[i+1]
                affected = all_affected_flat[a_start:a_end]
                
                # Acceptance decisions for each replica
                accept_flags = [False] * num_replicas
                energy_deltas = [0] * num_replicas
                
                for k in range(num_replicas):
                    old_sums_affected = replicas_sums[k][affected]
                    new_sums_affected = old_sums_affected + deltas[k]
                    new_abs_affected = np.abs(new_sums_affected)
                    old_abs_affected = np.abs(old_sums_affected)
                    
                    # Energy change: change in max absolute sum
                    old_max = replica_maxs[k]
                    new_max = max(old_max, int(np.max(new_abs_affected)))
                    
                    # For tempered sampling, we use a simplified energy model:
                    # E = sum of squares of absolute sums (or just max)
                    # Let's use max absolute sum as the primary energy, 
                    # but for acceptance we need a scalar energy difference.
                    # Approximate energy as sum of (abs_sum)^2 for affected APs
                    old_E = np.sum(old_abs_affected ** 2)
                    new_E = np.sum(new_abs_affected ** 2)
                    dE = new_E - old_E
                    
                    energy_deltas[k] = dE
                    
                    # Metropolis acceptance
                    if dE <= 0:
                        accept_flags[k] = True
                    else:
                        T_k = temperatures[k]
                        if T_k > 0:
                            p = np.exp(-dE / T_k)
                            if rng.random() < p:
                                accept_flags[k] = True
                        # If T is very small, only accept if dE <= 0 (already handled)
                
                # Apply accepted flips
                for k in range(num_replicas):
                    if accept_flags[k]:
                        replicas_sums[k][affected] += deltas[k]
                        replicas_s[k][i] = new_vals[k]
                        # Update replica max
                        replica_maxs[k] = max(replica_maxs[k], int(np.max(np.abs(replicas_sums[k][affected]))))
                        # Actually, we need to recompute max if the previous max AP was affected
                        # For simplicity, just track the max of affected and global
                        # This is an approximation; for correctness we'd need full recompute
                        # But for speed, we track the max of affected APs and keep a running max
                        # If the old max was in affected, we might have decreased it
                        # Let's do a quick check: if any affected AP had abs sum == old_max,
                        # and now they are all less, we need to find new max
                        # This is expensive. Instead, let's just update the max as:
                        # new_max = max(max of affected, max of non-affected)
                        # We don't track non-affected max separately.
                        # Simplification: just take max of affected and assume it's the new max
                        # This is not strictly correct but works as a heuristic
                        replica_maxs[k] = int(np.max(np.abs(replicas_sums[k][affected])))
                        # But this ignores APs not affected. So replica_maxs[k] might be too low.
                        # Better: keep a global max estimate and update conservatively
                        # For now, let's just use the max of affected as a lower bound
                        # and periodically recompute exact max
                
                # Check for improvement in replica 0 (coldest, most accurate)
                # Periodically recompute exact max for replica 0
                if rng.random() < 0.01:  # 1% of steps
                    exact_max = int(np.max(np.abs(replicas_sums[0])))
                    replica_maxs[0] = exact_max
                    if exact_max < best_score:
                        best_score = exact_max
                        best_s = replicas_s[0].copy()
                
                # Parallel Tempering swap attempt
                # Swap between adjacent replicas based on Metropolis criterion
                # P(accept swap) = min(1, exp(-(E_high - E_low)/T_low - (E_high - E_low)/T_high))
                # Simplified: try to swap replica k and k+1 with some probability
                if rng.random() < 0.05:  # 5% of steps
                    k = rng.randint(0, num_replicas - 2)
                    # Compute energy difference
                    E_k = np.sum(replicas_sums[k] ** 2)
                    E_k1 = np.sum(replicas_sums[k+1] ** 2)
                    dE = E_k1 - E_k
                    T_k = temperatures[k]
                    T_k1 = temperatures[k+1]
                    
                    if dE <= 0:
                        accept_swap = True
                    else:
                        # P = exp(-dE/T_k) * exp(dE/T_k1)
                        # = exp(dE * (1/T_k1 - 1/T_k))
                        log_p = dE * (1.0/T_k1 - 1.0/T_k)
                        if log_p > 0:
                            accept_swap = True
                        else:
                            accept_swap = rng.random() < np.exp(log_p)
                    
                    if accept_swap:
                        # Swap configurations
                        replicas_s[k], replicas_s[k+1] = replicas_s[k+1], replicas_s[k]
                        replicas_sums[k], replicas_sums[k+1] = replicas_sums[k+1], replicas_sums[k]
                        replica_maxs[k], replica_maxs[k+1] = replica_maxs[k+1], replica_maxs[k]
        
        # Final recompute for best_s
        final_max = int(np.max(np.abs(compute_sums(best_s, ap_indices))))
        if final_max < best_score:
            best_score = final_max
        
        return best_s, best_score

    # Standard local search for refinement
    def optimize_local(s_init, time_budget, rng):
        s = s_init.copy()
        sums = compute_sums(s, ap_indices)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

        # Histogram for O(1) max queries
        max_possible_sum = N
        hist = np.zeros(max_possible_sum + 2, dtype=np.int32)
        for val in np.abs(sums):
            hist[val] += 1

        def get_max_from_hist():
            for k in range(max_possible_sum, -1, -1):
                if hist[k] > 0:
                    return k
            return 0

        indices_list = list(range(N))
        stuck_counter = 0
        max_stuck = 200

        while time.time() - start_time < time_budget:
            rng.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() - start_time > time_budget:
                    break
                
                old_val = s[i]
                delta = -2 * old_val
                new_val = -old_val
                
                a_start = affected_offsets[i]
                a_end = affected_offsets[i+1]
                affected = all_affected_flat[a_start:a_end]
                
                old_sums_affected = sums[affected]
                new_sums_affected = old_sums_affected + delta
                new_abs_affected = np.abs(new_sums_affected)
                old_abs_affected = np.abs(old_sums_affected)
                
                # Only accept if it doesn't increase the global max
                max_new_affected = int(np.max(new_abs_affected))
                if max_new_affected > current_max:
                    stuck_counter += 1
                    continue
                
                # Apply flip
                changed = old_abs_affected != new_abs_affected
                if np.any(changed):
                    for j in range(len(changed)):
                        if changed[j]:
                            hist[old_abs_affected[j]] -= 1
                            hist[new_abs_affected[j]] += 1
                sums[affected] = new_sums_affected
                s[i] = new_val
                stuck_counter = 0
                
                new_global_max = get_max_from_hist()
                if new_global_max < current_max:
                    current_max = new_global_max
                    if current_max < best_score:
                        best_score = current_max
                        best_s = s.copy()
                else:
                    stuck_counter += 1
                
                if stuck_counter >= max_stuck:
                    # Perturbation
                    num_perturb = rng.randint(2, 4)
                    perturbation_indices = rng.sample(range(N), num_perturb)
                    for pi in perturbation_indices:
                        old_val = s[pi]
                        delta = -2 * old_val
                        new_val = -old_val
                        a_start = affected_offsets[pi]
                        a_end = affected_offsets[pi+1]
                        affected = all_affected_flat[a_start:a_end]
                        old_sums_affected = sums[affected]
                        new_sums_affected = old_sums_affected + delta
                        new_abs_affected = np.abs(new_sums_affected)
                        
                        # Accept if doesn't increase max
                        max_new_affected = int(np.max(new_abs_affected))
                        if max_new_affected <= current_max:
                            old_abs_affected = np.abs(old_sums_affected)
                            changed = old_abs_affected != new_abs_affected
                            if np.any(changed):
                                for j in range(len(changed)):
                                    if changed[j]:
                                        hist[old_abs_affected[j]] -= 1
                                        hist[new_abs_affected[j]] += 1
                            sums[affected] = new_sums_affected
                            s[pi] = new_val
                    stuck_counter = 0
        
        return best_s, best_score

    rng = random.Random(42)
    
    # Run 1: Parallel Tempering from Thue-Morse
    tm = thue_morse(N)
    s1, score1 = optimize_pt(tm, time_limit * 0.5, rng)
    best_s = s1
    best_score = score1

    # Run 2: Local search refinement from best
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        s2, score2 = optimize_local(best_s, remaining - 1.0, rng)
        if score2 < best_score:
            best_score = score2
            best_s = s2

    # Run 3: Parallel Tempering from random init
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3.0:
        rand_init = random_init(N, rng)
        s3, score3 = optimize_pt(rand_init, remaining - 1.0, rng)
        if score3 < best_score:
            best_score = score3
            best_s = s3

    # Final refinement
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        s4, score4 = optimize_local(best_s, remaining - 0.5, rng)
        if score4 < best_score:
            best_score = score4
            best_s = s4

    return [int(x) for x in best_s]
