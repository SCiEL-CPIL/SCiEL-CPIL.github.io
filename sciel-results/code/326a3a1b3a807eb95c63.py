import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    N = 256

    # 1. Precompute ALL APs and mappings
    ap_indices = []
    aps_containing = [[] for _ in range(N)]
    
    # Generate all APs: (start, step, length)
    # start in 1..N, step in 1..N
    # length L such that start + (L-1)*step <= N
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            curr = s - 1
            if curr >= N:
                continue
            # Build the sequence
            seq = []
            while curr < N:
                seq.append(curr)
                curr += d
            
            for L in range(1, len(seq) + 1):
                idx_list = seq[:L]
                ap_idx = len(ap_indices)
                ap_indices.append(idx_list)
                for idx in idx_list:
                    aps_containing[idx].append(ap_idx)
    
    num_aps = len(ap_indices)
    
    # Convert to numpy arrays for speed
    ap_indices_np = [np.array(lst, dtype=np.int32) for lst in ap_indices]
    aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]
    
    # Flat array for affected APs per position
    all_affected_flat = []
    affected_offsets = [0] * (N + 1)
    for i in range(N):
        affected_offsets[i] = len(all_affected_flat)
        all_affected_flat.extend(aps_containing[i])
    affected_offsets[N] = len(all_affected_flat)
    all_affected_flat = np.array(all_affected_flat, dtype=np.int32)
    affected_lengths = np.array([affected_offsets[i+1] - affected_offsets[i] for i in range(N)], dtype=np.int32)

    # 2. Initialization Strategies
    def thue_morse(N):
        seq = np.zeros(N, dtype=np.int8)
        for i in range(N):
            parity = i.bit_count() & 1
            seq[i] = 1 if parity == 0 else -1
        return seq

    def random_init(N, rng):
        return np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)

    def greedy_construct(rng):
        s = np.zeros(N, dtype=np.int8)
        s[0] = 1
        sums = np.zeros(num_aps, dtype=np.int32)
        
        # Initialize sums for APs ending at 0
        # Only singleton {0}
        # We need to find APs ending at 0.
        # An AP ends at i if its last element is i.
        # In our generation, an AP is defined by start, step, length.
        # The last element is start + (length-1)*step.
        # We didn't store "aps_ending_at". Let's compute it on the fly or precompute.
        # Precomputation is better.
        pass # We will use a simpler greedy or just random + polish.
        # Given the time constraint and complexity of maintaining incremental sums for greedy
        # with ~80k APs, a fast random init + strong local search is often better.
        # Let's stick to random init + polish, maybe with a few greedy-like initializations.
        
        # Actually, let's implement a simple greedy that picks the sign minimizing the max violation of *newly completed* APs.
        # But we need to know which APs are completed at step i.
        # Let's precompute aps_ending_at.
        return s, sums

    # Precompute aps_ending_at for potential greedy use
    aps_ending_at = [[] for _ in range(N)]
    for ap_idx, idx_list in enumerate(ap_indices):
        end_pos = idx_list[-1]
        aps_ending_at[end_pos].append(ap_idx)
    aps_ending_at_np = [np.array(lst, dtype=np.int32) for lst in aps_ending_at]
    
    # Precompute parent pointers for incremental sum update in greedy
    # Parent of (start, step, L) is (start, step, L-1)
    # We need a map from (start, step, L) to index
    ap_params = []
    for d in range(1, N + 1):
        for s in range(1, N + 1):
            curr = s - 1
            if curr >= N:
                continue
            seq = []
            while curr < N:
                seq.append(curr)
                curr += d
            for L in range(1, len(seq) + 1):
                ap_params.append((s-1, d, L))
                
    ap_to_idx = {}
    for idx, (st, stp, ln) in enumerate(ap_params):
        ap_to_idx[(st, stp, ln)] = idx
        
    parent = np.full(num_aps, -1, dtype=np.int32)
    for idx, (st, stp, ln) in enumerate(ap_params):
        if ln > 1:
            parent[idx] = ap_to_idx.get((st, stp, ln-1), -1)

    def fast_greedy(rng):
        s = np.zeros(N, dtype=np.int8)
        s[0] = 1
        sums = np.zeros(num_aps, dtype=np.int32)
        
        # Init
        for ap_idx in aps_ending_at_np[0]:
            sums[ap_idx] = 1
            
        for i in range(1, N):
            best_val = 1
            best_max = N + 1
            
            for val in [1, -1]:
                max_new = 0
                for ap_idx in aps_ending_at_np[i]:
                    p = parent[ap_idx]
                    if p != -1:
                        new_sum = sums[p] + val
                    else:
                        new_sum = val
                    if abs(new_sum) > max_new:
                        max_new = abs(new_sum)
                
                if max_new < best_max:
                    best_max = max_new
                    best_val = val
                elif max_new == best_max and rng.random() < 0.5:
                    best_val = val

            s[i] = best_val
            for ap_idx in aps_ending_at_np[i]:
                p = parent[ap_idx]
                if p != -1:
                    sums[ap_idx] = sums[p] + best_val
                else:
                    sums[ap_idx] = best_val
        return s, sums

    def compute_sums(s):
        sums = np.zeros(num_aps, dtype=np.int16)
        for r, idx_list in enumerate(ap_indices_np):
            sums[r] = np.sum(s[idx_list])
        return sums

    # 3. Local Search with Simulated Annealing / Tabu-like features
    def optimize(s_init, time_budget, rng):
        s = s_init.copy()
        sums = compute_sums(s)
        current_max = int(np.max(np.abs(sums)))
        best_s = s.copy()
        best_score = current_max

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
        
        # Parameters
        T = 1.0
        T_min = 0.01
        cooling = 0.99
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
                
                # Check if any new sum exceeds current_max
                if np.any(new_abs_affected > current_max):
                    # Allow worsening moves with probability based on temperature?
                    # For now, strict local search first, then relax.
                    # Let's try a soft constraint: accept if max violation is small.
                    max_violation = int(np.max(new_abs_affected - current_max))
                    if max_violation > 0:
                        # Calculate probability of accepting a worsening move
                        # p = exp(-delta_E / T)
                        # delta_E is the increase in max sum
                        p_accept = np.exp(-max_violation * 2.0 / T)
                        if rng.random() < p_accept:
                            # Accept
                            changed = old_abs_affected != new_abs_affected
                            if np.any(changed):
                                for j in range(len(changed)):
                                    if changed[j]:
                                        hist[old_abs_affected[j]] -= 1
                                        hist[new_abs_affected[j]] += 1
                            sums[affected] = new_sums_affected
                            s[i] = new_val
                            new_global_max = get_max_from_hist()
                            if new_global_max < current_max:
                                current_max = new_global_max
                                if current_max < best_score:
                                    best_score = current_max
                                    best_s = s.copy()
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                        else:
                            stuck_counter += 1
                    else:
                        # Non-worsening move
                        changed = old_abs_affected != new_abs_affected
                        if np.any(changed):
                            for j in range(len(changed)):
                                if changed[j]:
                                    hist[old_abs_affected[j]] -= 1
                                    hist[new_abs_affected[j]] += 1
                        sums[affected] = new_sums_affected
                        s[i] = new_val
                        new_global_max = get_max_from_hist()
                        if new_global_max < current_max:
                            current_max = new_global_max
                            if current_max < best_score:
                                best_score = current_max
                                best_s = s.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    # Move is valid (no new violations beyond current_max)
                    changed = old_abs_affected != new_abs_affected
                    if np.any(changed):
                        for j in range(len(changed)):
                            if changed[j]:
                                hist[old_abs_affected[j]] -= 1
                                hist[new_abs_affected[j]] += 1
                    sums[affected] = new_sums_affected
                    s[i] = new_val
                    new_global_max = get_max_from_hist()
                    if new_global_max < current_max:
                        current_max = new_global_max
                        if current_max < best_score:
                            best_score = current_max
                            best_s = s.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                
                # Cooling
                T = max(T_min, T * cooling)
                
                # Perturbation if stuck
                if stuck_counter >= max_stuck:
                    num_perturb = rng.randint(2, 5)
                    pert_indices = rng.sample(range(N), num_perturb)
                    for pi in pert_indices:
                        s[pi] = -s[pi]
                        # Update sums for affected APs of pi
                        a_s = affected_offsets[pi]
                        a_e = affected_offsets[pi+1]
                        aff = all_affected_flat[a_s:a_e]
                        delta_p = -2 * s[pi] # Wait, s[pi] is already flipped.
                        # Let's just recompute sums for these APs? No, too slow.
                        # Let's do:
                        old_val = -s[pi]
                        delta_p = -2 * old_val
                        sums[aff] += delta_p
                        # Update hist
                        old_abs = np.abs(sums[aff] - delta_p)
                        new_abs = np.abs(sums[aff])
                        changed = old_abs != new_abs
                        if np.any(changed):
                            for j in range(len(changed)):
                                if changed[j]:
                                    hist[old_abs[j]] -= 1
                                    hist[new_abs[j]] += 1
                    
                    # Recalculate global max
                    current_max = get_max_from_hist()
                    T = 1.0
                    stuck_counter = 0

        return best_s, best_score

    # 4. Main Execution
    rng = random.Random(42)
    best_s_overall = None
    best_score_overall = N

    # Try a few initializations
    inits = [
        thue_morse(N),
        random_init(N, rng),
        fast_greedy(rng)[0]
    ]
    
    # Add some perturbed randoms
    for _ in range(3):
        inits.append(random_init(N, rng))

    for init_s in inits:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Give each init a portion of the time
        time_left = time_limit - (time.time() - start_time)
        budget_per_init = min(10.0, time_left * 0.5)
        
        s_polished, score_polished = optimize(init_s, budget_per_init, rng)
        
        if best_s_overall is None or score_polished < best_score_overall:
            best_score_overall = score_polished
            best_s_overall = s_polished.copy()

    # Final polish on the best found
    time_left = time_limit - (time.time() - start_time)
    if time_left > 2.0 and best_s_overall is not None:
        s_final, score_final = optimize(best_s_overall, time_left * 0.9, rng)
        if score_final < best_score_overall:
            best_score_overall = score_final
            best_s_overall = s_final

    return [int(x) for x in best_s_overall]
