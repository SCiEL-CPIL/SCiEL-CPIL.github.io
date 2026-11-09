import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    
    # Known solution with max denominator 248
    # 1/8 + 1/31 + 1/248 = 5/31
    best_solution = [8, 31, 248]
    best_max_den = 248
    
    # Try to find a solution with max denominator < 248
    # Use a limited DFS with max depth 4 and strict time limit
    search_deadline = start_time + 5.0
    
    # We search for solutions with max denominator M < 248
    # To be efficient, we can just search in the range [7, 247] with depth limit
    # But iterating M downwards is slow if we restart DFS.
    # Better: Single DFS on range [7, 247] with depth limit, tracking max denominator used.
    # However, we want to minimize the MAX denominator.
    # So we want to find ANY solution in [7, 247]. If found, we can try to reduce M further?
    # Or just find the one with smallest max den?
    # A single DFS that explores smaller denominators first might find a solution with small max den quickly.
    # But it doesn't guarantee the minimum max den.
    # However, if we find one with max den 100, that's better than 248.
    # Let's just find *any* solution with max den < 248.
    # If we find one, we can try to optimize further? No, time limit.
    # Just finding one is an improvement.
    
    # Precompute suffix sums for pruning?
    # For a fixed range [7, 247], suffix sums are static.
    # But we are limiting depth.
    
    max_den_limit = 247
    min_den = 7
    
    # Suffix sums for max possible sum of k terms?
    # Let's just use the simple suffix sum of all reciprocals for "max possible sum" pruning.
    # suffix[i] = sum(1/j for j in i..247)
    suffix = [Fraction(0)] * (max_den_limit + 2)
    for i in range(max_den_limit, min_den - 1, -1):
        suffix[i] = suffix[i+1] + Fraction(1, i)
        
    # Also prefix min sums? 
    # Min sum of k terms from i..247 is 1/247 + ... ?
    # Actually, if we pick k terms, the smallest sum is picking the largest denominators.
    # min_suffix[i][k] = sum of 1/j for the k largest j in [i, 247].
    # This is expensive to precompute for all i, k.
    # Approximation: min sum of k terms >= k / 247.
    # Max sum of k terms <= k / i.
    
    found_solution = [None]
    
    def dfs(start, remaining, depth, path):
        if time.time() > search_deadline:
            return
        if found_solution[0]:
            return
            
        if remaining == 0:
            found_solution[0] = list(path)
            return
            
        if depth == 0:
            return
            
        if start > max_den_limit:
            return
            
        # Pruning 1: Max possible sum
        # Max sum of 'depth' terms starting from 'start' is 1/start + 1/(start+1) + ...
        # We can approximate or compute on fly? 
        # Using suffix[start] is the sum of ALL terms from start to end.
        # If remaining > suffix[start], we can't reach it.
        if remaining > suffix[start]:
            return
            
        # Pruning 2: Min possible sum
        # Min sum of 'depth' terms starting from 'start' (picking largest denoms)
        # Approx: depth / max_den_limit
        if remaining < depth * Fraction(1, max_den_limit):
            return
            
        # Option 1: Pick 'start'
        # Check if 1/start is not too big
        if remaining >= Fraction(1, start):
            path.append(start)
            dfs(start + 1, remaining - Fraction(1, start), depth - 1, path)
            path.pop()
            
            if found_solution[0]:
                return
                
        # Option 2: Do not pick 'start'
        dfs(start + 1, remaining, depth, path)
        
    # Search for solution with up to 4 terms
    # Why 4? 3 terms we checked 7, 8. 8 gave 248. 7 gave 434.
    # Maybe 4 terms can give < 248?
    dfs(min_den, target, 4, [])
    
    if found_solution[0]:
        # Verify
        s = sum(Fraction(1, d) for d in found_solution[0])
        if s == target and len(found_solution[0]) > 0:
            # Check max denominator
            max_d = max(found_solution[0])
            if max_d < best_max_den:
                best_solution = found_solution[0]
                best_max_den = max_d
    
    return best_solution
