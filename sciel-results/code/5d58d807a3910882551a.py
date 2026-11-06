"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    # Set a safety deadline. The evaluator kills at 90s, but we want to return well before.
    deadline = time.time() + 55.0
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts. 
    # conflict[i] is a set of indices j such that lcm(nums[i], nums[j]) <= 256.
    # Using sets for fast intersection/union checks.
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(nums[i], nums[j])
            if (nums[i] // g) * nums[j] <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    def is_valid_add(selected, i):
        """Check if adding i to selected is valid (no conflicts)."""
        if len(selected) < len(conflict[i]):
            return not selected.intersection(conflict[i])
        else:
            return not conflict[i].intersection(selected)
    
    def greedy_with_prefix(prefix_skip=0):
        """Greedy construction that skips the first 'prefix_skip' smallest numbers."""
        selected = set()
        # Start from index prefix_skip (0-indexed in sorted order)
        for idx in range(prefix_skip, n):
            i = idx  # nums is already sorted ascending
            if is_valid_add(selected, i):
                selected.add(i)
        return selected
    
    def local_search(init_set, max_iters=15, time_budget=None):
        """
        Local search starting from init_set.
        Returns (best_set, best_score).
        """
        if time_budget is None:
            time_budget = deadline - 2.0
        
        best = set(init_set)
        best_score = sum(1.0 / nums[i] for i in best)
        
        for it in range(max_iters):
            if time.time() > time_budget:
                break
                
            improved = False
            
            # 1. Try adding one element with highest reciprocal
            best_add = -1
            best_add_val = 0.0
            for i in range(n):
                if i in best:
                    continue
                if is_valid_add(best, i):
                    val = 1.0 / nums[i]
                    if val > best_add_val:
                        best_add_val = val
                        best_add = i
            
            if best_add != -1:
                best.add(best_add)
                best_score += best_add_val
                improved = True
            
            # 2. Try swap: remove one, add one
            if time.time() < time_budget - 1.0:
                best_swap = None
                best_swap_gain = 0.0
                
                best_list = list(best)
                for i in best_list:
                    if time.time() > time_budget:
                        break
                    
                    for j in range(n):
                        if j in best:
                            continue
                        # Only swap if j has higher reciprocal
                        if nums[j] >= nums[i]:
                            continue
                        
                        # Check compatibility: j must not conflict with best\{i}
                        inter = conflict[j] & best
                        if i in inter:
                            if len(inter) > 1:
                                continue
                        else:
                            if len(inter) > 0:
                                continue
                                
                        gain = 1.0 / nums[j] - 1.0 / nums[i]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (i, j)
                            
                if best_swap is not None:
                    i, j = best_swap
                    best.remove(i)
                    best.add(j)
                    best_score += best_swap_gain
                    improved = True
            
            # 3. Try remove one element (if it blocks better additions)
            if time.time() < time_budget - 1.0:
                best_remove = None
                best_remove_gain = 0.0
                best_remove_add = -1
                
                for i in best:
                    if time.time() > time_budget:
                        break
                    
                    # If we remove i, what's the best we can add?
                    temp_best = best - {i}
                    best_add_candidate = -1
                    best_add_val_candidate = 0.0
                    for j in range(n):
                        if j in temp_best:
                            continue
                        if is_valid_add(temp_best, j):
                            val = 1.0 / nums[j]
                            if val > best_add_val_candidate:
                                best_add_val_candidate = val
                                best_add_candidate = j
                    
                    if best_add_candidate != -1:
                        # Net gain: add val - remove val
                        net_gain = best_add_val_candidate - 1.0 / nums[i]
                        if net_gain > best_remove_gain:
                            best_remove_gain = net_gain
                            best_remove = i
                            best_remove_add = best_add_candidate
                
                if best_remove is not None:
                    best.remove(best_remove)
                    best.add(best_remove_add)
                    best_score += best_remove_gain
                    improved = True
            
            if not improved:
                break
        
        return best, best_score

    best_set = set()
    best_score = 0.0
    
    # Phase 1: Try multiple exclusion prefixes
    # The key insight: skipping small numbers allows different solution clusters
    prefix_candidates = [0, 2, 5, 10, 20, 40, 80, 120]
    for prefix_skip in prefix_candidates:
        if time.time() > deadline - 10.0:
            break
        
        greedy_set = greedy_with_prefix(prefix_skip)
        if not greedy_set:
            continue
        
        ls_set, ls_score = local_search(greedy_set, max_iters=10)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
    
    # Phase 2: Random restarts with perturbation
    random.seed(42)
    remaining_time = deadline - 3.0
    restart_count = 0
    max_restarts = 80
    
    while time.time() < remaining_time and restart_count < max_restarts:
        restart_count += 1
        
        # Different strategies for diversity
        strategy = restart_count % 4
        
        if strategy == 0:
            # Exclusion prefix + greedy
            prefix_skip = random.choice([1, 3, 7, 15, 30, 60, 100])
            current_set = greedy_with_prefix(prefix_skip)
        elif strategy == 1:
            # Perturb best: remove random elements and refill
            current_set = set(best_set)
            num_remove = random.randint(3, 8)
            for _ in range(num_remove):
                if current_set:
                    victim = random.choice(list(current_set))
                    current_set.remove(victim)
            # Refill greedily
            for i in range(n):
                if i not in current_set and is_valid_add(current_set, i):
                    current_set.add(i)
        elif strategy == 2:
            # Random valid subset
            s = set(random.sample(range(n), random.randint(30, 80)))
            valid_s = set()
            for i in sorted(s, key=lambda x: nums[x]):
                if is_valid_add(valid_s, i):
                    valid_s.add(i)
            current_set = valid_s
        else:
            # Greedy with random noise in ordering
            current_set = set()
            rand_order = list(range(n))
            random.shuffle(rand_order)
            rand_order.sort(key=lambda i: nums[i] + random.random() * 20)
            for i in rand_order:
                if is_valid_add(current_set, i):
                    current_set.add(i)
        
        if not current_set:
            continue
        
        ls_set, ls_score = local_search(current_set, max_iters=5)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
    
    # Phase 3: Final polish with more iterations on best
    if time.time() < deadline - 1.0:
        ls_set, ls_score = local_search(best_set, max_iters=30)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
    
    return [nums[i] for i in best_set]
# EVOLVE-BLOCK-END
