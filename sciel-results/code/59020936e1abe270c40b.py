import time
from math import gcd
from functools import reduce
from bisect import bisect_right

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 90s limit

    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 2..48
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index i corresponds to denominator i+2
    # weight[i] = L // (i+2)
    weights = [L // d for d in denominators]
    
    # Meet-in-the-Middle Strategy
    # Split the list of items into two halves.
    # n=47. Split into 23 and 24.
    # 2^23 = 8,388,608. 2^24 = 16,777,216.
    # This is manageable if we use the DP-like subset sum generation and filter.
    
    def solve_mitm():
        mid = n // 2 # 23
        left_indices = list(range(mid))
        right_indices = list(range(mid, n))
        
        left_weights = weights[:mid]
        right_weights = weights[mid:]
        
        # Generate subset sums for left half
        # We store (sum, mask) for all valid subsets (sum < L)
        # To be memory efficient and fast, we generate all 2^23 sums first?
        # 2^23 is ~8.4M. Storing 8.4M tuples is ~60MB. Okay.
        # But generating them by iterating mask 0..2^23-1 and summing bits is O(2^23 * 23) ~ 190M ops. Too slow in Python.
        
        # Better: Iterative expansion.
        # start with [(0, 0)]
        # for each weight w:
        #   new = [(s+w, m|bit) for s, m in current if s+w < L]
        #   current += new
        # This is O(2^m) total if no pruning, but with pruning it's less.
        # However, Python list comprehensions are fast.
        
        def gen_subsets(w_list):
            results = [(0, 0)]
            for i, w in enumerate(w_list):
                bit = 1 << i
                # We need to iterate over the CURRENT size of results, not the growing one
                # to avoid using items added in this step.
                # Copying the list or using range(len) is needed.
                current_len = len(results)
                new_items = []
                for k in range(current_len):
                    s, m = results[k]
                    ns = s + w
                    if ns < L:
                        new_items.append((ns, m | bit))
                results.extend(new_items)
            return results

        # Generate left subsets
        t0 = time.time()
        left_subsets = gen_subsets(left_weights)
        t1 = time.time()
        # Sort by sum
        left_subsets.sort(key=lambda x: x[0])
        
        if time.time() - start_time > timeout:
            return None

        # Generate right subsets
        right_subsets = gen_subsets(right_weights)
        # Sort by sum
        right_subsets.sort(key=lambda x: x[0])
        
        if time.time() - start_time > timeout:
            return None

        # Now find best pair (l, r) such that l.sum + r.sum < L
        # Maximize l.sum + r.sum
        # Two-pointer technique
        # left is sorted ascending. right is sorted ascending.
        # i points to largest left sum (index len-1)
        # j points to smallest right sum (index 0)
        
        i = len(left_subsets) - 1
        j = 0
        best_sum = 0
        best_l_mask = 0
        best_r_mask = 0
        
        # We want max sum < L.
        while i >= 0 and j < len(right_subsets):
            s_l = left_subsets[i][0]
            s_r = right_subsets[j][0]
            total = s_l + s_r
            if total < L:
                if total > best_sum:
                    best_sum = total
                    best_l_mask = left_subsets[i][1]
                    best_r_mask = right_subsets[j][1]
                # Since total < L, we can try to increase right sum (j+1) to get closer to L
                # while keeping the same left sum.
                j += 1
            else:
                # total >= L. We must decrease left sum to make room for larger right sums.
                i -= 1
                
        if best_sum > 0:
            # Convert masks to indices
            indices = []
            # Left mask corresponds to indices 0..mid-1
            for k in range(mid):
                if best_l_mask & (1 << k):
                    indices.append(k)
            # Right mask corresponds to indices mid..n-1
            for k in range(len(right_weights)):
                if best_r_mask & (1 << k):
                    indices.append(mid + k)
                    
            return indices
            
        return None

    result = solve_mitm()
    if result:
        return result
        
    # Fallback to B&B if MITM fails or times out
    def lcm_func(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm_func, range(2, 49))
    weights = [L // d for d in denominators]
    n = len(weights)
    
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
    best_sum_weight = 0
    best_indices = []
    
    def update_best(sum_w, indices):
        nonlocal best_sum_weight, best_indices
        if sum_w > best_sum_weight:
            best_sum_weight = sum_w
            best_indices = list(indices)
            
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum_weight, best_indices
        
        if time.time() - start_time > timeout:
            return
            
        if current_sum + suffix_sum[idx] <= best_sum_weight:
            return
            
        if idx == n:
            if current_sum > best_sum_weight and current_sum < L:
                update_best(current_sum, current_indices)
            return
            
        if current_sum + weights[idx] < L:
            dfs(idx + 1, current_sum + weights[idx], current_indices + [idx])
            
        if time.time() - start_time > timeout:
            return
            
        dfs(idx + 1, current_sum, current_indices)
        
    dfs(0, 0, [])
    
    if not best_indices:
        return [0]
        
    return best_indices
