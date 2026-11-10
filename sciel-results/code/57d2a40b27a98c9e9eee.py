import time
from collections import deque

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    # LCM of divisors
    LCM = 30030
    
    # Precompute for each divisor d, the set of residues mod LCM that are multiples of d
    # Actually, we need: for a window starting at L, the multiples of d in [L, L+W]
    # are k*d where ceil(L/d) <= k <= floor((L+W)/d)
    
    # Strategy: For each window size W (starting from small), check if there exists
    # a starting position L such that we can assign distinct multiples.
    # 
    # Key insight: Instead of trying all L in [0, LCM), we can use a smarter approach.
    # For a fixed W, the number of candidates for divisor d in window [L, L+W] is
    # approximately (W+1)/d. For small W, large divisors have very few candidates.
    #
    # Better approach: Use constraint propagation / backtracking with MRV heuristic.
    # But even better: since the pattern repeats every LCM, we only need to check
    # L in [0, LCM). For each L, we do a quick feasibility check.
    
    # Alternative structural approach: Model as bipartite matching.
    # Left side: divisors (10 nodes)
    # Right side: candidate integers in the window
    # Edge from divisor d to integer x if d divides x and L <= x <= L+W
    # We need a perfect matching.
    # 
    # For efficiency, we can use the Hopcroft-Karp algorithm or simple BFS-based
    # augmenting path matching since n=10 is small.
    
    # Even better structural change: Instead of iterating W and L, use the fact that
    # the optimal solution has a specific structure. Let's try a different approach:
    # 
    # For each divisor d, the multiples are d, 2d, 3d, ...
    # We want to pick one multiple for each divisor such that all are distinct and
    # the span is minimized.
    # 
    # Key observation: The span max-min is at least as large as the maximum gap
    # between consecutive chosen values. 
    #
    # Let's try: For each possible "base" value b (the minimum), find the optimal
    # assignment. But b can be up to 1000000, so this is too large.
    #
    # Better: Since the pattern repeats every LCM=30030, we only need to consider
    # b mod LCM. For each residue r in [0, LCM), the actual value is b = r + k*LCM
    # for some k >= 0. The relative positions of multiples within a window of size W
    # depend only on r and W, not on k (as long as all values are positive).
    
    # Let me try a cleaner approach:
    # 1. For W from 0 upward:
    # 2. For L from 0 to LCM-1:
    # 3. Build bipartite graph: divisor i connects to multiples of divisors[i] in [L, L+W]
    # 4. Check if perfect matching exists using augmenting paths
    
    def has_perfect_matching(available):
        """Check if there's a perfect matching using augmenting path algorithm."""
        n_left = len(available)
        match_right = {}  # right_node -> left_node
        
        def find_augmenting(left_node, visited):
            for right_node in available[left_node]:
                if right_node in visited:
                    continue
                visited.add(right_node)
                if right_node not in match_right or find_augmenting(match_right[right_node], visited):
                    match_right[right_node] = left_node
                    return True
            return False
        
        for left_node in range(n_left):
            visited = set()
            if not find_augmenting(left_node, visited):
                return False
        return True
    
    # Sort divisors by size (larger divisors are more constrained for small W)
    # Actually, let's keep the original order but sort indices by constraint level
    
    for W in range(0, 500):
        if time.time() > deadline:
            # Return best found so far or fallback
            break
        
        for L in range(LCM):
            if time.time() > deadline:
                break
            
            # Build available candidates for each divisor
            available = []
            valid = True
            for i, d in enumerate(divisors):
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                if k_min > k_max:
                    valid = False
                    break
                multiples = [k * d for k in range(k_min, k_max + 1)]
                available.append(multiples)
            
            if not valid:
                continue
            
            # Sort by number of candidates (most constrained first) for better pruning
            # But for matching, we just need to check existence
            
            # Quick check: each divisor must have at least 1 candidate
            if any(len(available[i]) == 0 for i in range(n)):
                continue
            
            # Use augmenting path matching
            if has_perfect_matching(available):
                # Found a valid assignment - now extract it
                # We need to redo the matching to get the actual assignment
                match_right = {}
                
                def find_augmenting(left_node, visited):
                    for right_node in available[left_node]:
                        if right_node in visited:
                            continue
                        visited.add(right_node)
                        if right_node not in match_right or find_augmenting(match_right[right_node], visited):
                            match_right[right_node] = left_node
                            return True
                    return False
                
                for left_node in range(n):
                    visited = set()
                    find_augmenting(left_node, visited)
                
                # Extract assignment
                result = [None] * n
                for right_node, left_node in match_right.items():
                    result[left_node] = int(right_node)
                
                # Verify
                if all(x is not None for x in result):
                    if len(set(result)) == n:
                        if all(result[i] % divisors[i] == 0 for i in range(n)):
                            if all(0 < x <= 1000000 for x in result):
                                return result
    
    # Fallback: return the divisors themselves
    return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
