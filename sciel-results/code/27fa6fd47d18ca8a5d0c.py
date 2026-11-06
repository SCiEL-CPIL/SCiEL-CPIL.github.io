import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    # Precompute sum indices for a given element x against a set of chosen elements
    # For a set S, adding x creates sums: 2x, and x+y for all y in S
    # We maintain a numpy array 'sum_counts' where sum_counts[s] = number of representations of s
    
    def is_valid_addition_bitset(x, chosen_mask, sum_counts):
        """
        Check if x can be added.
        chosen_mask: boolean array of length N+1
        sum_counts: numpy array of length 2*N+1
        Returns True if valid.
        """
        # Check 2x
        if sum_counts[2*x] >= 2:
            return False
        
        # Check x + y for all y in chosen
        # Sums are x + y where y is in chosen
        # The relevant sums range from x + min(chosen) to x + max(chosen)
        # We can just check the specific sums.
        # To be fast, we can use the fact that chosen is sparse, but with numpy we can do vectorized checks.
        # However, iterating over chosen is O(K).
        # A faster way: 
        # The sums x+y are at indices x + chosen_indices.
        # We need to check if any sum_counts[x + chosen_indices] >= 2.
        
        chosen_indices = np.nonzero(chosen_mask)[0]
        if len(chosen_indices) == 0:
            return True
        
        sums_to_check = x + chosen_indices
        # Filter sums that are within bounds (max sum is 2000)
        valid_sums = sums_to_check <= 2*N
        if np.any(valid_sums):
            if np.any(sum_counts[sums_to_check[valid_sums]] >= 2):
                return False
        
        return True

    def get_tightness_bitset(x, chosen_mask, sum_counts):
        """
        Return tightness score (number of sums that would reach count 2 if x is added).
        Returns None if invalid.
        """
        # Check 2x
        tight = 0
        if sum_counts[2*x] >= 2:
            return None
        if sum_counts[2*x] == 1:
            tight += 1
        
        chosen_indices = np.nonzero(chosen_mask)[0]
        if len(chosen_indices) > 0:
            sums_to_check = x + chosen_indices
            valid_sums = sums_to_check <= 2*N
            if np.any(valid_sums):
                counts_at_sums = sum_counts[sums_to_check[valid_sums]]
                # If any count >= 2, invalid
                if np.any(counts_at_sums >= 2):
                    return None
                # Count how many are == 1 (these would become 2)
                tight += np.sum(counts_at_sums == 1)
        
        return tight

    def add_element(x, chosen_mask, sum_counts, chosen_list):
        """Add x to the set."""
        chosen_mask[x] = True
        sum_counts[2*x] += 1
        chosen_indices = np.nonzero(chosen_mask)[0]
        # Remove x itself from the indices for pairwise sums? No, x+x is handled.
        # We need to add x+y for all y in chosen, y != x.
        # Since we just set chosen_mask[x]=True, x is in chosen_indices.
        # We need to add x+y for y in chosen_indices where y != x.
        # Or simply: for y in chosen_list (before adding x), sum_counts[x+y] += 1.
        # But chosen_list is a Python list. Let's use the mask.
        # Actually, it's easier to just iterate over the current chosen list.
        for y in chosen_list:
            sum_counts[x + y] += 1
        chosen_list.append(x)

    def remove_element(x, chosen_mask, sum_counts, chosen_list):
        """Remove x from the set."""
        # Remove x+y for all y in chosen_list, y != x
        for y in chosen_list:
            if y != x:
                sum_counts[x + y] -= 1
        sum_counts[2*x] -= 1
        chosen_mask[x] = False
        chosen_list.remove(x)

    def run_greedy_bitset(noise=0.0, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        
        chosen_mask = np.zeros(N + 1, dtype=bool)
        sum_counts = np.zeros(2*N + 1, dtype=np.int32)
        chosen_list = []
        candidates = list(range(1, N + 1))
        
        while candidates:
            if time.time() > time_limit:
                break
            
            best_x = None
            best_score = float('inf')
            
            for x in candidates:
                score = get_tightness_bitset(x, chosen_mask, sum_counts)
                if score is None:
                    continue
                total = score + (random.random() * noise if noise > 0 else 0)
                if total < best_score:
                    best_score = total
                    best_x = x
            
            if best_x is None:
                break
                
            add_element(best_x, chosen_mask, sum_counts, chosen_list)
            candidates.remove(best_x)
            
        return chosen_list

    def local_search_bitset(initial_set, time_limit):
        chosen_mask = np.zeros(N + 1, dtype=bool)
        sum_counts = np.zeros(2*N + 1, dtype=np.int32)
        chosen_list = list(initial_set)
        
        # Initialize sum_counts
        for x in chosen_list:
            sum_counts[2*x] += 1
        for i in range(len(chosen_list)):
            for j in range(i+1, len(chosen_list)):
                sum_counts[chosen_list[i] + chosen_list[j]] += 1
        for x in chosen_list:
            chosen_mask[x] = True
            
        candidates = set(range(1, N + 1))
        for x in chosen_list:
            candidates.discard(x)
            
        if not candidates:
            return chosen_list
            
        while time.time() < time_limit:
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            added_any = False
            for x in cand_list:
                if time.time() > time_limit:
                    break
                if is_valid_addition_bitset(x, chosen_mask, sum_counts):
                    add_element(x, chosen_mask, sum_counts, chosen_list)
                    candidates.discard(x)
                    added_any = True
                    break
            
            if added_any:
                continue
            
            # Try swaps
            if len(candidates) == 0:
                break
                
            swap_success = False
            for _ in range(10):
                if time.time() > time_limit:
                    break
                    
                x_to_try = random.choice(list(candidates))
                chosen_copy = list(chosen_list)
                random.shuffle(chosen_copy)
                
                for y in chosen_copy[:20]:
                    # Tentatively remove y
                    remove_element(y, chosen_mask, sum_counts, chosen_list)
                    
                    if is_valid_addition_bitset(x_to_try, chosen_mask, sum_counts):
                        add_element(x_to_try, chosen_mask, sum_counts, chosen_list)
                        candidates.discard(x_to_try)
                        candidates.add(y)
                        swap_success = True
                        break
                    else:
                        # Revert: add y back
                        add_element(y, chosen_mask, sum_counts, chosen_list)
                    
                if swap_success:
                    break
            
            if not swap_success:
                # Kick
                if time.time() > time_limit:
                    break
                kick_size = random.choice([1, 2, 3])
                to_remove = random.sample(chosen_list, min(kick_size, len(chosen_list)))
                for y in to_remove:
                    remove_element(y, chosen_mask, sum_counts, chosen_list)
                    candidates.add(y)
                
                # Refill greedily
                cand_list = list(candidates)
                random.shuffle(cand_list)
                for x in cand_list:
                    if time.time() > time_limit:
                        break
                    if is_valid_addition_bitset(x, chosen_mask, sum_counts):
                        add_element(x, chosen_mask, sum_counts, chosen_list)
                        candidates.discard(x)
                        
        return chosen_list

    # Main loop
    round_num = 0
    while time.time() < deadline:
        round_num += 1
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        noise = random.uniform(0, 3.0)
        greedy_time = min(remaining * 0.2, 8.0)
        baseline = run_greedy_bitset(noise=noise, time_limit=time.time() + greedy_time)
        
        if len(baseline) > best_size:
            best_size = len(baseline)
            best_set = list(baseline)
        
        ls_time = min(remaining * 0.6, 30.0)
        start_ls_set = best_set if len(best_set) > len(baseline) else baseline
        refined = local_search_bitset(start_ls_set, time.time() + ls_time)
        
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = list(refined)
            
        if time.time() < deadline - 10:
            noise2 = random.uniform(0, 5.0)
            baseline2 = run_greedy_bitset(noise=noise2, time_limit=min(deadline - 5, time.time() + 5))
            if len(baseline2) > best_size:
                best_size = len(baseline2)
                best_set = list(baseline2)
                
            if time.time() < deadline - 3:
                refined2 = local_search_bitset(best_set, time.time() + min(remaining * 0.1, 5.0))
                if len(refined2) > best_size:
                    best_size = len(refined2)
                    best_set = list(refined2)
            
    return best_set
