import math
import time

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # We search for minimal M such that there exists a subset of {13, ..., M} 
    # with product N.
    # Since factors must be distinct, subset implies distinct elements.
    
    # Upper bound for M: 
    # If we take k=1, M=N.
    # If we take k=2, M approx sqrt(N) ~ 22000.
    # But we want minimal M, so we start from small M.
    
    # How large can M be?
    # If we use 13, 14, ..., 13+k-1, product grows fast.
    # For k=7, product of 13..19 is 2.5e8.
    # For k=8, product of 13..20 is 5e9 > N.
    # So k is at most 7.
    # For k=7, min product is 2.5e8.
    # Max product for M=20 (using 14..20) is 3.9e8.
    # Max product for M=21 (using 15..21) is 15*16*17*18*19*20*21 = 1.6e9?
    # 14*...*20 = 3.9e8.
    # 15*...*21 = 3.9e8 / 14 * 21 = 3.9e8 * 1.5 = 5.85e8.
    # This is > N.
    # So for M=21, a subset of size 7 might exist.
    
    # Let's iterate M from 13 upwards.
    # For each M, we check if there is a subset of {13, ..., M} with product N.
    # We can use DFS to find such a subset.
    
    def find_subset(target, nums):
        """
        Find a subset of nums with product target.
        nums is sorted.
        Returns the subset or None.
        """
        n = len(nums)
        # Pruning: if product of all nums < target, impossible.
        # If product of smallest k nums > target, impossible for that k.
        
        # We don't know k.
        # We can iterate k from 1 to n?
        # Or just DFS and track product.
        
        # To be efficient, we can use the fact that product must be exactly target.
        # We can divide target by each number and recurse.
        
        # Better: Backtracking with index.
        def backtrack(start, current_product, path):
            if current_product == target:
                return path
            if current_product > target:
                return None
            
            # Pruning: even if we multiply by all remaining numbers, 
            # can we reach target?
            # product_remaining_min = 1
            # for i in range(start, n): product_remaining_min *= nums[i]
            # if current_product * product_remaining_min < target: return None
            
            # Actually, just check if current_product * product(nums[start:]) < target
            # This is expensive to compute every time.
            # Precompute suffix products?
            # Suffix product of nums from i to end.
            
            for i in range(start, n):
                if current_product * nums[i] > target:
                    break # Since nums are sorted, further nums will be larger
                new_prod = current_product * nums[i]
                path.append(nums[i])
                res = backtrack(i + 1, new_prod, path)
                if res:
                    return res
                path.pop()
            return None
            
        # Optimization: 
        # If current_product * (product of all remaining) < target, prune.
        # Precompute suffix_products[i] = product(nums[i:])
        
        suffix_products = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_products[i] = suffix_products[i + 1] * nums[i]
            
        def backtrack_opt(start, current_product, path):
            if current_product == target:
                return path
            if current_product > target:
                return None
            
            # Prune: if current_product * suffix_products[start] < target, 
            # it's impossible to reach target with remaining numbers.
            if current_product * suffix_products[start] < target:
                return None
                
            for i in range(start, n):
                if current_product * nums[i] > target:
                    break
                new_prod = current_product * nums[i]
                path.append(nums[i])
                res = backtrack_opt(i + 1, new_prod, path)
                if res:
                    return res
                path.pop()
            return None

        return backtrack_opt(0, 1, [])

    # Iterate M
    # Max M? 
    # If no solution found for M up to 100, we might need larger.
    # But k is small.
    # For k=2, min M is such that 13*M >= N => M >= N/13 ~ 37M.
    # But we want minimal M.
    # The minimal M will likely be small because we can use many factors.
    # Max k is 7.
    # For k=7, M is around 20-30.
    # For k=1, M=N.
    # We want to minimize M.
    # So we check M=13, 14, ...
    
    # How high to go?
    # If we go up to 1000, 2^988 is huge.
    # But DFS prunes heavily.
    # However, for large M, the number of candidates is large.
    # But if M is large, we can pick small numbers.
    # Actually, if M is large, we can always pick {13, ..., 19, X} where X = N / (13*...*19).
    # If X is an integer > 19 and X <= M, we found a solution.
    # So for any M >= X, we have a solution.
    # So the minimal M is the minimum over all valid subsets of max(subset).
    
    # We can just iterate M from 13 to, say, 1000.
    # If we find a solution, return it.
    
    # To speed up, we can generate subsets more intelligently.
    # But for M < 50, the set size is < 38.
    # 2^38 is too big for brute force, but DFS with pruning is fast.
    
    for M in range(13, 1000):
        if time.time() - start_time > 85:
            break
        nums = list(range(13, M + 1))
        sol = find_subset(N, nums)
        if sol:
            return sol
            
    # Fallback
    return [N]
