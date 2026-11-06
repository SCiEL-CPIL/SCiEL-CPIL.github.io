"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 55.0
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(nums[i], nums[j])
            if (nums[i] // g) * nums[j] <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Precompute odd-part chains: numbers with the same odd part
    # For a number x, odd_part(x) = x / 2^v2(x)
    def odd_part(x):
        while x % 2 == 0:
            x //= 2
        return x
    
    chains = {}  # odd_part -> list of indices
    for i in range(n):
        op = odd_part(nums[i])
        if op not in chains:
            chains[op] = []
        chains[op].append(i)
    
    # For each chain, precompute which pairs conflict (they all do if lcm <= 256)
    # Actually, within a chain {d, 2d, 4d, ...}, lcm(2^a*d, 2^b*d) = 2^max(a,b)*d
    # So two elements conflict iff 2^max(a,b)*d <= 256
    # This means within each chain, it's a "threshold" structure: if you pick 2^a*d,
    # you can't pick any 2^b*d where 2^max(a,b)*d <= 256
    
    def is_valid_add(selected, i):
        """Check if adding i to selected is valid (no conflicts)."""
        if len(selected) < len(conflict[i]):
            return not selected.intersection(conflict[i])
        else:
            return not conflict[i].intersection(selected)
    
    def get_chain_conflict_mask(i, selected):
        """Return how many elements in selected conflict with i."""
        return len(conflict[i] & selected)
    
    def greedy_with_prefix(prefix_skip=0, chain_aware=True):
        """Greedy construction that skips the first 'prefix_skip' smallest numbers."""
        selected = set()
        for idx in range(prefix_skip, n):
            i = idx
            if is_valid_add(selected, i):
                selected.add(i)
        return selected
    
    def greedy_chain_aware():
        """
        Greedy that considers chains: for each odd-part chain, pick at most one element.
        Within a chain, prefer the smallest (highest reciprocal) element that doesn't conflict.
        Process chains in order of their best possible contribution.
        """
        selected = set()
        
        # Sort chains by the reciprocal of their smallest element (descending)
        chain_list = []
        for op, indices in chains.items():
            if not indices:
                continue
            # The smallest element in this chain has the highest reciprocal
            best_idx = min(indices, key=lambda i: nums[i])
            chain_list.append((1.0 / nums[best_idx], op, indices))
        
        chain_list.sort(key=lambda x: -x[0])
        
        for _, op, indices in chain_list:
            # Try each element in this chain from smallest to largest
            for i in sorted(indices, key=lambda x: nums[x]):
                if is_valid_add(selected, i):
                    selected.add(i)
                    break  # Only pick one from this chain
        
        return selected
    
    def local_search(init_set, max_iters=20, time_budget=None):
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
            
            # 3. Try two-swap: remove two, add two
            if time.time() < time_budget - 2.0:
                best_two_swap = None
                best_two_swap_gain = 0.0
                
                best_list = list(best)
                n_best = len(best_list)
                
                # Limit the search space for two-swaps
                for idx1 in range(min(n_best, 30)):
                    if time.time() > time_budget:
                        break
                    i1 = best_list[idx1]
                    for idx2 in range(idx1 + 1, min(n_best, 30)):
                        i2 = best_list[idx2]
                        
                        # Candidate removals: i1, i2
                        # Find best two additions
                        best_adds = []
                        best_adds_val = 0.0
                        
                        temp_best = best - {i1, i2}
                        
                        # Try adding pairs
                        candidates = [j for j in range(n) if j not in temp_best and is_valid_add(temp_best, j)]
                        candidates.sort(key=lambda j: -1.0 / nums[j])
                        
                        # Try top candidates for first add
                        for j1 in candidates[:15]:
                            temp_best2 = temp_best | {j1}
                            for j2 in candidates[:15]:
                                if j2 == j1:
                                    continue
                                if is_valid_add(temp_best2, j2):
                                    val = 1.0 / nums[j1] + 1.0 / nums[j2]
                                    if val > best_adds_val:
                                        best_adds_val = val
                                        best_adds = [j1, j2]
                        
                        if len(best_adds) == 2:
                            remove_val = 1.0 / nums[i1] + 1.0 / nums[i2]
                            gain = best_adds_val - remove_val
                            if gain > best_two_swap_gain:
                                best_two_swap_gain = gain
                                best_two_swap = (i1, i2, best_adds[0], best_adds[1])
                
                if best_two_swap is not None:
                    i1, i2, j1, j2 = best_two_swap
                    best.remove(i1)
                    best.remove(i2)
                    best.add(j1)
                    best.add(j2)
                    best_score += best_two_swap_gain
                    improved = True
            
            # 4. Try remove one element (if it blocks better additions)
            if time.time() < time_budget - 1.0:
                best_remove = None
                best_remove_gain = 0.0
                best_remove_add = -1
                
                for i in best:
                    if time.time() > time_budget:
                        break
                    
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
    
    # Phase 1: Try multiple exclusion prefixes and chain-aware greedy
    prefix_candidates = [0, 2, 5, 10, 20, 40, 80, 120]
    for prefix_skip in prefix_candidates:
        if time.time() > deadline - 15.0:
            break
        
        greedy_set = greedy_with_prefix(prefix_skip)
        if not greedy_set:
            continue
        
        ls_set, ls_score = local_search(greedy_set, max_iters=10)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
    
    # Phase 1b: Chain-aware greedy
    if time.time() < deadline - 15.0:
        chain_set = greedy_chain_aware()
        if chain_set:
            ls_set, ls_score = local_search(chain_set, max_iters=15)
            if ls_score > best_score:
                best_set = ls_set
                best_score = ls_score
    
    # Phase 2: Random restarts with perturbation
    random.seed(42)
    remaining_time = deadline - 3.0
    restart_count = 0
    max_restarts = 100
    
    while time.time() < remaining_time and restart_count < max_restarts:
        restart_count += 1
        
        strategy = restart_count % 5
        
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
        elif strategy == 3:
            # Greedy with random noise in ordering
            current_set = set()
            rand_order = list(range(n))
            random.shuffle(rand_order)
            rand_order.sort(key=lambda i: nums[i] + random.random() * 20)
            for i in rand_order:
                if is_valid_add(current_set, i):
                    current_set.add(i)
        else:
            # Chain-aware with random perturbation
            current_set = greedy_chain_aware()
            if current_set:
                num_remove = random.randint(2, 5)
                for _ in range(num_remove):
                    if current_set:
                        victim = random.choice(list(current_set))
                        current_set.remove(victim)
                for i in range(n):
                    if i not in current_set and is_valid_add(current_set, i):
                        current_set.add(i)
        
        if not current_set:
            continue
        
        ls_set, ls_score = local_search(current_set, max_iters=8)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
    
    # Phase 3: Final polish with more iterations on best
    if time.time() < deadline - 1.0:
        ls_set, ls_score = local_search(best_set, max_iters=40)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
    
    # Phase 4: Very fine-grained polish if time remains
    if time.time() < deadline - 0.5:
        # Try targeted swaps on the best solution
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            improved = False
            best_list = list(best_set)
            for i in best_list:
                if time.time() > deadline - 0.5:
                    break
                for j in range(n):
                    if j in best_set:
                        continue
                    if nums[j] >= nums[i]:
                        continue
                    inter = conflict[j] & best_set
                    if i in inter:
                        if len(inter) > 1:
                            continue
                    else:
                        if len(inter) > 0:
                            continue
                    gain = 1.0 / nums[j] - 1.0 / nums[i]
                    if gain > 1e-12:
                        best_set.remove(i)
                        best_set.add(j)
                        best_score += gain
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    return [nums[i] for i in best_set]
# EVOLVE-BLOCK-END
