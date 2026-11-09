"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = 512
    k = 4  # we want no 4-AP
    
    # Precompute all 4-APs in [1, n]
    # A 4-AP is (a, a+d, a+2d, a+3d) with d >= 1 and a+3d <= n
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Build index: for each element, which APs contain it
    element_to_aps = {i: [] for i in range(1, n + 1)}
    for idx, ap in enumerate(ap_list):
        for elem in ap:
            element_to_aps[elem].append(idx)
    
    # Greedy construction: start with a good seed
    # Strategy: use base-4 digit construction as initial seed, then local search
    
    def base4_seed():
        """Construct a 4-AP-free set using base-4 digits."""
        # The classic construction: numbers whose base-4 representation uses only digits 0 and 1
        # But we need to map to [1, n]. Let's use a different approach.
        # Actually, the standard result is that numbers with base-4 digits in {0,1} form a 4-AP-free set.
        # For [1, 512] = [1, 4^4], we can take numbers 1..512 whose base-4 representation
        # (padded to 4 digits) uses only 0 and 1.
        result = set()
        for x in range(1, n + 1):
            digits = []
            v = x
            for _ in range(4):
                digits.append(v % 4)
                v //= 4
            if all(d in (0, 1) for d in digits):
                result.add(x)
        return result
    
    def is_4ap_free(selected_set):
        """Check if selected_set contains no 4-AP."""
        selected_arr = np.array(sorted(selected_set))
        selected_arr_set = set(selected_arr.tolist())
        for ap in ap_list:
            if all(elem in selected_arr_set for elem in ap):
                return False
        return True
    
    def count_4aps(selected_set):
        """Count number of 4-APs in selected_set."""
        selected_set_arr = np.array(sorted(selected_set))
        selected_arr_set = set(selected_set_arr.tolist())
        count = 0
        for ap in ap_list:
            if all(elem in selected_arr_set for elem in ap):
                count += 1
        return count
    
    # Start with base-4 seed
    current_set = base4_seed()
    current_size = len(current_set)
    
    # Local search: try to add elements, and if that fails, try swaps
    # First, try greedy addition
    def try_add_greedy(selected_set, remaining_elements):
        """Try to greedily add elements from remaining_elements."""
        improved = True
        while improved:
            improved = False
            for elem in sorted(remaining_elements):
                if elem in selected_set:
                    continue
                # Check if adding elem creates a 4-AP
                new_set = selected_set | {elem}
                # Quick check: only need to check APs containing elem
                conflict = False
                for ap_idx in element_to_aps[elem]:
                    ap = ap_list[ap_idx]
                    if all(e in new_set for e in ap):
                        conflict = True
                        break
                if not conflict:
                    selected_set.add(elem)
                    improved = True
        return selected_set
    
    # Greedy add
    remaining = set(range(1, n + 1)) - current_set
    current_set = try_add_greedy(current_set, remaining)
    current_size = len(current_set)
    
    # Local search with swaps
    def local_search_swap(selected_set, max_iters=10000):
        """Try to improve by swapping elements out and in."""
        selected_arr = sorted(selected_set)
        selected_set_copy = set(selected_arr)
        not_in_set = set(range(1, n + 1)) - selected_set_copy
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Try random swaps
            if not not_in_set or not selected_set_copy:
                break
            
            elem_out = random.choice(list(selected_set_copy))
            elem_in = random.choice(list(not_in_set))
            
            # Check if swap is valid
            new_set = (selected_set_copy - {elem_out}) | {elem_in}
            
            # Check 4-APs involving elem_in
            conflict = False
            for ap_idx in element_to_aps[elem_in]:
                ap = ap_list[ap_idx]
                if all(e in new_set for e in ap):
                    conflict = True
                    break
            
            if not conflict:
                # Also need to check that removing elem_out doesn't matter (it shouldn't create issues)
                # But we need to ensure no new 4-AP is formed. Since we're only adding elem_in and removing elem_out,
                # the only possible new 4-APs are those containing elem_in.
                selected_set_copy = new_set
                not_in_set = set(range(1, n + 1)) - selected_set_copy
                return selected_set_copy, True
        
        return selected_set_copy, False
    
    # Run local search
    best_set = set(current_set)
    best_size = current_size
    
    for iteration in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple random restarts of local search
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Start from current best
            trial_set = set(best_set)
            
            # Try swaps
            trial_set, improved = local_search_swap(trial_set, max_iters=5000)
            
            if len(trial_set) > best_size:
                best_set = trial_set
                best_size = len(trial_set)
    
    # Final greedy pass
    remaining = set(range(1, n + 1)) - best_set
    best_set = try_add_greedy(best_set, remaining)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
