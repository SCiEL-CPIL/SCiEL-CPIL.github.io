import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin, well under 90s
    
    N = factorial(12)
    
    # Generate all divisors of N
    divisors = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
        i += 1
    divisors.sort()
    
    # We want to minimize the largest factor M.
    # Iterate M from 14 upwards (since 13 is not a divisor of 12!)
    # Note: The problem states factors > 12.
    # 13 is prime and does not divide 12!, so it cannot be a factor.
    # So the smallest possible factor is 14.
    
    # Pre-compute divisors > 12
    # We will iterate M and filter divisors <= M on the fly or maintain a list.
    # Since divisors is sorted, we can find the index.
    
    # For each M, candidates are divisors d such that 12 < d <= M.
    # We need to check if a subset of these candidates multiplies to N.
    
    # DFS function
    # To avoid re-computing candidates for each M, we can just iterate M and 
    # check if the current set of candidates (which grows) has a solution.
    # However, the set of candidates changes.
    # It's better to iterate M and run a fresh DFS for each M, 
    # because the candidate set is different.
    # But the candidate set is small for small M.
    
    # Optimization: 
    # If a solution exists with max factor M, it also exists with max factor M' > M.
    # We want the MINIMUM M.
    # So we iterate M from 14 up.
    # The first M for which a solution is found is the answer.
    
    # To speed up DFS, we can sort candidates in descending order?
    # If we pick large numbers, we reduce the target faster.
    # But we also need to ensure we don't miss combinations.
    # Standard subset product DFS:
    # def dfs(remaining, candidates, idx):
    #   if remaining == 1: return True
    #   if idx >= len(candidates): return False
    #   # Pruning: if remaining is prime and not in candidates[idx:], fail?
    #   # Simpler: iterate candidates from idx to end
    #   for i in range(idx, len(candidates)):
    #       d = candidates[i]
    #       if d > remaining: break (since sorted asc? no, if sorted desc, d < remaining)
    #       if remaining % d == 0:
    #           if dfs(remaining // d, candidates, i+1):
    #               return True
    #   return False
    
    # We need to return the factors, not just True/False.
    
    # Let's implement the search.
    
    # To handle the "distinct" constraint, passing `i+1` ensures we don't reuse `d`.
    # Since candidates are unique divisors, this is correct.
    
    # We should sort candidates in descending order to prune `d > remaining` faster?
    # If sorted descending, `d` starts large. If `d > remaining`, we can't use it.
    # But we can use smaller ones.
    # If sorted ascending, `d` starts small. `d <= remaining` is usually true.
    # Descending is better for product reduction?
    # Let's try descending.
    
    for M in range(14, N + 1):
        if time.time() > deadline:
            break
            
        # Get candidates <= M and > 12
        # Since divisors is sorted, we can find the upper bound for M.
        # But for small M, we can just filter.
        # To be efficient, we can maintain a pointer or just list comprehension.
        # Number of divisors is ~800, so list comprehension is fast.
        
        candidates = [d for d in divisors if 12 < d <= M]
        
        if not candidates:
            continue
            
        # Sort descending for DFS
        candidates.sort(reverse=True)
        
        # Check if we can form N
        # We use a helper to find the factors
        def find_factors(remaining, candidates, idx, current_factors):
            if remaining == 1:
                return list(current_factors)
            
            if idx >= len(candidates):
                return None
            
            # Pruning: 
            # If the product of all remaining candidates is less than remaining, fail?
            # Hard to compute quickly.
            # If remaining is prime and not in candidates[idx:], fail.
            # Simple check: if no candidate in candidates[idx:] divides remaining, fail.
            # We can check this in the loop.
            
            for i in range(idx, len(candidates)):
                d = candidates[i]
                if d > remaining:
                    # Since sorted descending, all subsequent are smaller? No, descending means larger to smaller.
                    # So if d > remaining, we skip d. 
                    # But smaller d might be <= remaining.
                    # So we cannot break. We must continue.
                    continue
                
                if remaining % d != 0:
                    continue
                
                current_factors.append(d)
                res = find_factors(remaining // d, candidates, i + 1, current_factors)
                if res:
                    return res
                current_factors.pop()
                
            return None
        
        solution = find_factors(N, candidates, 0, [])
        
        if solution:
            # Verify
            prod = 1
            for x in solution:
                prod *= x
            if prod == N and len(set(solution)) == len(solution) and all(x > 12 for x in solution) and len(solution) <= 12:
                return solution
        
    # Fallback
    return [N]
