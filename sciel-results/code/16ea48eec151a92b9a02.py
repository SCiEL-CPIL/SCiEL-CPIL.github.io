import time
from math import isqrt

def construct():
    start_time = time.time()
    
    # 1. Generate primes up to 256
    N = 256
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(N) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    primes = [i for i, prime in enumerate(sieve) if prime]
    
    # 2. Identify primes that can be part of a composite number <= 256
    # Any prime p > 128 must be a singleton because 2*p > 256.
    # We only need to optimize the selection for primes <= 128.
    # Primes > 128 are always included as singletons.
    
    large_primes = [p for p in primes if p > 128]
    small_primes = [p for p in primes if p <= 128]
    
    # The problem is to partition small_primes into groups.
    # Each group is a set of primes whose product is <= 256.
    # The value of a group is the product.
    # We want to maximize the sum of values.
    # Note: A singleton prime p has value p.
    # Combining p and q into pq is beneficial if pq > p + q.
    
    # We use Branch and Bound to find the optimal partition of small_primes.
    # State: (index in small_primes, used_primes_mask, current_sum)
    # Since len(small_primes) is 31, we can't use a simple bitmask of 31 bits easily in recursion without being slow if not pruned.
    # But most primes > 42 can only be singleton or paired with 2.
    
    # Let's sort small_primes in descending order to make decisions on larger primes first?
    # Or ascending?
    # Ascending allows us to use small primes to "absorb" larger ones?
    # Actually, let's just iterate through small_primes.
    
    # Optimization: Precompute valid combinations for each prime p in small_primes
    # using available small primes.
    
    # We will use a recursive function with memoization or simple B&B.
    # Given the small size, simple B&B with a good upper bound is sufficient.
    
    # Upper bound: Sum of all remaining primes (assuming we can keep them all as singletons)
    # This is a weak bound.
    # Better bound: Sum of max(p, best_possible_combination_with_p) for remaining?
    
    # Let's try a greedy + local search or just simple B&B.
    # With 31 items, B&B is fast if we prune well.
    
    # Sort small_primes descending.
    small_primes.sort(reverse=True)
    
    # Precompute suffix sums for upper bound
    n_sp = len(small_primes)
    suffix_sums = [0] * (n_sp + 1)
    for i in range(n_sp - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + small_primes[i]
    
    best_sum = 0
    best_selection = [] # List of lists of primes, or just the resulting numbers
    
    # We track which primes from the "small" set (<= 42) are still available to form composites.
    # Primes > 42 can only be singleton or paired with 2, 3, 5 (if product <= 256).
    # Actually, any prime p in small_primes can be combined with any subset of *other* primes in small_primes
    # as long as the product <= 256.
    
    # To simplify, we can just treat all small_primes as items to be assigned.
    # We process them one by one.
    
    def solve(idx, available_mask, current_sum, selected_composites):
        nonlocal best_sum, best_selection
        
        if time.time() - start_time > 50:
            return
            
        if idx == n_sp:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = selected_composites[:]
            return
            
        # Upper bound: current_sum + sum of all remaining primes
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        p = small_primes[idx]
        
        # Option 1: Take p as a singleton
        # This is always valid.
        # We "use" p, so it's removed from available.
        # But wait, if we take p as singleton, we can't use it to form a composite later.
        # Since we are processing sequentially, "using" it means we decide its fate now.
        # If we decide it's a singleton, it's done.
        
        # However, we might want to combine p with *previous* primes?
        # No, we process sequentially. If we combined p with a previous prime, we would have decided that when processing the previous prime.
        # So, at step idx, p is either:
        # 1. A singleton.
        # 2. Part of a composite with some subset of *earlier* primes?
        # No, that logic is tricky.
        
        # Better approach:
        # Decide the group for each prime.
        # Since groups are disjoint, we can assign each prime to a group.
        # To avoid permutations, we can say:
        # The smallest prime in a group "leads" the group.
        # So, when processing prime p (in ascending order), we decide which *larger* primes (already processed? No, if ascending, larger are not processed yet) 
        # Let's process in ASCENDING order.
        # When we are at prime p, we can form a group containing p and any subset of *remaining* primes (larger ones).
        # This ensures each group is generated exactly once (by its smallest prime).
        
        pass # Placeholder, will implement below
    
    # Let's restart the B&B logic with Ascending Order
    small_primes.sort()
    n_sp = len(small_primes)
    suffix_sums = [0] * (n_sp + 1)
    for i in range(n_sp - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + small_primes[i]
    
    best_sum = 0
    best_result = []
    
    # available_mask: bitmask of primes in small_primes that are still available to be added to a group.
    # Initially all 1s.
    # When we form a group with p and some subset S of available primes (excluding p), we remove p and S from available.
    
    def dfs(idx, available_mask, current_sum, current_groups):
        nonlocal best_sum, best_result
        
        if time.time() - start_time > 55:
            return

        if idx == n_sp:
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = current_groups[:]
            return
            
        # Upper bound
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        p = small_primes[idx]
        
        # If p is not available, it means it was consumed by a previous group (as a larger member).
        # In that case, we just skip to the next index.
        if not (available_mask & (1 << idx)):
            dfs(idx + 1, available_mask, current_sum, current_groups)
            return
            
        # Option 1: p is a singleton.
        # We remove p from available.
        new_mask = available_mask & ~(1 << idx)
        dfs(idx + 1, new_mask, current_sum + p, current_groups + [p])
        
        # Option 2: p is the smallest member of a composite group.
        # We can add any subset of available primes j > idx to this group.
        # The product must be <= 256.
        
        # Generate all valid subsets of available primes j > idx
        # To do this efficiently, we can iterate through submasks or just backtrack.
        # Given the small size, backtracking is fine.
        
        # We will build the group incrementally.
        # Start with product = p.
        # Try adding j = idx + 1, idx + 2, ...
        
        def build_group(start_idx, current_product, current_group_list, current_mask):
            nonlocal best_sum, best_result
            
            if time.time() - start_time > 55:
                return
                
            # Try to extend the group with a larger prime
            for j in range(start_idx, n_sp):
                if current_mask & (1 << j):
                    p_j = small_primes[j]
                    new_product = current_product * p_j
                    if new_product <= 256:
                        new_mask = current_mask & ~(1 << j)
                        new_group = current_group_list + [p_j]
                        
                        # Recurse to add more primes to this same group
                        build_group(j + 1, new_product, new_group, new_mask)
                        
                    else:
                        # Since small_primes is sorted, if p_j is too big, all subsequent are too big.
                        break
            
            # Also, the group can end here.
            # If we end here, we commit this group.
            # The group value is current_product.
            # The primes used are current_group_list.
            # We need to pass the state to the next DFS level (idx + 1).
            # But wait, the group formation consumes primes j > idx.
            # The next DFS call should start at idx + 1, with the mask updated to remove all primes in current_group_list.
            
            # We must ensure we don't double count or miss states.
            # The "commit" step happens after exploring all extensions.
            # But we need to know the final mask and sum.
            
            # This structure is slightly awkward for simple recursion.
            # Let's use a helper to generate all valid groups for p.
            
        # Simpler: Iterate through all valid subsets of available primes > idx
        # that can be added to p.
        
        # Since the number of primes is small, we can just enumerate subsets of the available primes > idx.
        # But we must respect the product limit.
        
        # Let's collect indices of available primes > idx
        candidates = [j for j in range(idx + 1, n_sp) if (available_mask & (1 << j))]
        
        # If no candidates, only singleton is possible (already handled)
        
        # We want to find all subsets S of candidates such that p * prod(S) <= 256.
        # And for each such S, we commit the group {p} U S.
        
        # To avoid generating too many subsets, we can prune based on product.
        
        # We'll do a DFS to generate valid subsets
        def gen_subsets(k, current_prod, current_subset, current_mask):
            nonlocal best_sum, best_result
            
            if time.time() - start_time > 55:
                return

            # Option: Stop adding to this group. Commit the group.
            # The group is [p] + current_subset.
            # Value is current_prod.
            # Mask update: remove p and all in current_subset.
            final_mask = available_mask & ~(1 << idx)
            for j in current_subset:
                final_mask &= ~(1 << j)
                
            dfs(idx + 1, final_mask, current_sum + current_prod, current_groups + [current_prod])
            
            # Option: Add more primes from candidates
            for j in range(k, len(candidates)):
                idx_j = candidates[j]
                p_j = small_primes[idx_j]
                new_prod = current_prod * p_j
                if new_prod <= 256:
                    gen_subsets(j + 1, new_prod, current_subset + [idx_j], current_mask & ~(1 << idx_j))
                else:
                    break # Sorted, so break
        
        if candidates:
            gen_subsets(0, p, [], available_mask)
            
    # Run the solver
    initial_mask = (1 << n_sp) - 1
    dfs(0, initial_mask, 0, [])
    
    # Construct final result
    # best_result contains the values of the composites/singletons from small_primes.
    # large_primes are always included.
    
    final_values = best_result + large_primes
    final_values.append(1)
    
    # Sort and return
    final_values.sort()
    
    # Ensure types are Python int
    return [int(v) for v in final_values]

if __name__ == "__main__":
    print(construct())
