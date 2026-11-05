import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    target_sums = range(target_min, target_max + 1)
    
    def calculate_counts(A_mask):
        N = np.zeros(257, dtype=np.int32)
        A_indices = np.where(A_mask)[0]
        for i_idx in range(len(A_indices)):
            a = A_indices[i_idx]
            for j_idx in range(i_idx, len(A_indices)):
                b = A_indices[j_idx]
                s = a + b
                if target_min <= s <= target_max:
                    N[s] += 1
        return N

    def get_objective(N):
        target_N = N[target_min:target_max+1]
        missing = np.sum(target_N == 0)
        if missing > 0:
            return (missing, float('inf'), float('inf'))
        peak = np.max(target_N)
        sum_sq = np.sum(target_N**2)
        return (0, peak, sum_sq)

    def greedy_init():
        A_mask = np.zeros(129, dtype=bool)
        N = np.zeros(257, dtype=np.int32)
        
        # Random start
        k = random.randint(25, 35)
        init_set = set(random.sample(range(129), k))
        A_mask = np.zeros(129, dtype=bool)
        for x in init_set:
            A_mask[x] = True
        
        N = calculate_counts(A_mask)
        
        # Repair: add elements to cover missing sums
        while True:
            missing_sums = [s for s in target_sums if N[s] == 0]
            if not missing_sums:
                break
            
            # Find best element to add
            best_x = None
            best_score = -1
            best_new_N = None
            
            candidates = [x for x in range(129) if not A_mask[x]]
            random.shuffle(candidates)
            
            for x in candidates:
                new_N = N.copy()
                A_indices = np.where(A_mask)[0]
                for a in A_indices:
                    s = x + a
                    if target_min <= s <= target_max:
                        new_N[s] += 1
                if 2*x in target_sums:
                    new_N[2*x] += 1
                
                covered = 0
                for s in missing_sums:
                    if new_N[s] > 0:
                        covered += 1
                
                if covered > best_score:
                    best_score = covered
                    best_x = x
                    best_new_N = new_N
            
            if best_x is None or best_score == 0:
                break
                
            A_mask[best_x] = True
            N = best_new_N
            
        return A_mask, N

    best_peak = float('inf')
    best_A_mask = None
    best_sum_sq = float('inf')

    # Differential Evolution inspired local search
    def local_search(A_mask, N, time_limit):
        A_mask = A_mask.copy()
        N = N.copy()
        current_obj = get_objective(N)
        current_peak = current_obj[1]
        current_sum_sq = current_obj[2]
        
        if current_peak == float('inf'):
            return A_mask, N, current_obj
            
        # Maintain a population of 4 solutions to allow DE-style crossover
        # Pop[0] is current best, others are perturbations
        pop_size = 4
        pop = []
        pop.append((A_mask.copy(), N.copy(), current_peak, current_sum_sq))
        
        for _ in range(pop_size - 1):
            # Create random perturbations of current best
            pert_mask = A_mask.copy()
            # Flip a few bits
            for _ in range(random.randint(1, 5)):
                if random.random() < 0.5:
                    # Add
                    candidates = [x for x in range(129) if not pert_mask[x]]
                    if candidates:
                        x = random.choice(candidates)
                        pert_mask[x] = True
                else:
                    # Remove
                    members = [x for x in range(129) if pert_mask[x]]
                    if members:
                        x = random.choice(members)
                        pert_mask[x] = False
            
            pert_N = calculate_counts(pert_mask)
            pert_obj = get_objective(pert_N)
            pop.append((pert_mask, pert_N, pert_obj[1], pert_obj[2]))
        
        while time.time() < time_limit:
            improved = False
            
            # Try DE-style crossover on the population
            # Pick 3 distinct random indices
            if len(pop) >= 3:
                idxs = random.sample(range(len(pop)), 3)
                base_idx, donor1_idx, donor2_idx = idxs[0], idxs[1], idxs[2]
                
                base_mask, base_N, base_peak, base_sq = pop[base_idx]
                d1_mask, d1_N, _, _ = pop[donor1_idx]
                d2_mask, d2_N, _, _ = pop[donor2_idx]
                
                # Crossover: v = base + F * (d1 - d2)
                # In binary space, this means:
                # If base is 0 and d1 is 1 and d2 is 0 -> likely 1
                # If base is 1 and d1 is 0 and d2 is 1 -> likely 0
                # Simplified: Randomly mix bits from base, d1, d2
                v_mask = base_mask.copy()
                crossover_rate = 0.2
                for i in range(129):
                    if random.random() < crossover_rate:
                        # Pick from d1 or d2
                        if random.random() < 0.5:
                            v_mask[i] = d1_mask[i]
                        else:
                            v_mask[i] = d2_mask[i]
                
                # Ensure validity? If not, repair or reject.
                # Let's just evaluate. If invalid, skip or repair.
                v_N = calculate_counts(v_mask)
                v_obj = get_objective(v_N)
                
                if v_obj[0] == 0: # Valid
                    v_peak = v_obj[1]
                    v_sq = v_obj[2]
                    
                    # Selection: Keep the best of (base, v)
                    if (v_peak < base_peak) or (v_peak == base_peak and v_sq < base_sq):
                        pop[base_idx] = (v_mask, v_N, v_peak, v_sq)
                        # Update global best if improved
                        if v_peak < current_peak or (v_peak == current_peak and v_sq < current_sum_sq):
                            A_mask = v_mask.copy()
                            N = v_N.copy()
                            current_peak = v_peak
                            current_sum_sq = v_sq
                            improved = True
                    else:
                        # If v is worse, maybe replace a worse member of pop?
                        # Find worst member in pop
                        worst_idx = -1
                        worst_val = float('-inf')
                        for i in range(len(pop)):
                            p, sq = pop[i][2], pop[i][3]
                            val = -p * 10000 - sq # Minimize peak, then sq
                            if val > worst_val:
                                worst_val = val
                                worst_idx = i
                        
                        if worst_idx != base_idx and (v_peak < pop[worst_idx][2] or (v_peak == pop[worst_idx][2] and v_sq < pop[worst_idx][3])):
                            pop[worst_idx] = (v_mask, v_N, v_peak, v_sq)

            # Also try standard random swaps to maintain diversity and escape
            if time.time() > time_limit:
                break
                
            # Random swap on current best
            A_indices = np.where(A_mask)[0]
            if len(A_indices) > 0:
                a_idx = random.randint(0, len(A_indices)-1)
                a = A_indices[a_idx]
                b = random.randint(0, 128)
                if not A_mask[b]:
                    new_N = N.copy()
                    # Remove a
                    for u_idx in range(len(A_indices)):
                        u = A_indices[u_idx]
                        if u == a:
                            continue
                        s = a + u
                        if target_min <= s <= target_max:
                            new_N[s] -= 1
                    s = 2*a
                    if target_min <= s <= target_max:
                        new_N[s] -= 1
                            
                    # Add b
                    for u_idx in range(len(A_indices)):
                        u = A_indices[u_idx]
                        if u == a:
                            continue
                        s = b + u
                        if target_min <= s <= target_max:
                            new_N[s] += 1
                    s = 2*b
                    if target_min <= s <= target_max:
                        new_N[s] += 1

                    new_obj = get_objective(new_N)
                    if new_obj[0] == 0:
                        if new_obj[1] < current_peak or (new_obj[1] == current_peak and new_obj[2] < current_sum_sq):
                            A_mask[a] = False
                            A_mask[b] = True
                            N = new_N
                            current_peak = new_obj[1]
                            current_sum_sq = new_obj[2]
                            improved = True
                            # Update pop[0]
                            pop[0] = (A_mask.copy(), N.copy(), current_peak, current_sum_sq)

            if not improved:
                # If stuck, maybe restart population?
                # Or just continue.
                pass

        return A_mask, N, (0, current_peak, current_sum_sq)

    # Main Loop
    for i in range(50):
        if time.time() > deadline:
            break
            
        # Greedy Init
        A_mask, N = greedy_init()
        
        # Check if valid
        obj = get_objective(N)
        if obj[0] > 0:
            continue
            
        peak = obj[1]
        sum_sq = obj[2]
        
        if peak < best_peak or (peak == best_peak and sum_sq < best_sum_sq):
            best_peak = peak
            best_sum_sq = sum_sq
            best_A_mask = A_mask.copy()
            
        # Local Search
        time_left = deadline - time.time()
        if time_left > 1:
            A_opt, N_opt, obj_opt = local_search(A_mask, N, time_left - 1)
            if obj_opt[0] == 0:
                if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                    best_peak = obj_opt[1]
                    best_sum_sq = obj_opt[2]
                    best_A_mask = A_opt.copy()

    if best_A_mask is None:
        # Fallback
        return list(range(64, 129))
        
    # Convert mask to list
    A_list = [i for i in range(129) if best_A_mask[i]]
    return A_list
