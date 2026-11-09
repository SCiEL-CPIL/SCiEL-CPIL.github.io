import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    L = 30030 # LCM of divisors
    
    # We will search for the minimum W
    # W can be 0, 1, ...
    # Since they must be distinct, W >= n-1 = 9?
    # Not necessarily, they don't have to be consecutive.
    # But min/max difference is W.
    
    # Iterate W from 0 upwards
    # Max W is L (actually less, but L is safe upper bound)
    for W in range(0, L + 1):
        # Iterate start A
        for A in range(0, L):
            # Generate candidates in [A, A+W]
            # Filter x > 0
            # Note: A can be 0, so 0 might be in range.
            
            # Optimization: 
            # If the number of candidates is less than n, no matching possible.
            # Candidates are integers x such that A <= x <= A+W and x > 0.
            
            # Let's just generate the list.
            # To speed up, we can check if for any i, there is NO multiple of d_i in [A, A+W] (with x>0).
            
            # Check feasibility for each i
            valid = True
            for i in range(n):
                d = divisors[i]
                # Find if there is a multiple of d in [A, A+W] that is > 0
                # Smallest multiple >= A:
                # k = ceil(A/d)
                # x = k*d
                # if x <= A+W and x > 0: OK
                # else: check if (k+1)*d ... no, if smallest is > A+W, then none.
                
                # Calculate first multiple >= A
                if A == 0:
                    # Multiples are 0, d, 2d...
                    # 0 is not > 0.
                    # So first candidate is d.
                    first_mult = d
                else:
                    k = (A + d - 1) // d
                    first_mult = k * d
                    if first_mult < A: # shouldn't happen with ceil
                        first_mult += d
                    
                if first_mult > A + W:
                    valid = False
                    break
                # Also need to ensure first_mult > 0
                if first_mult <= 0:
                    # This happens if A=0 and we considered 0, but we filtered x>0.
                    # If A=0, first_mult=d>0.
                    # If A>0, first_mult>=A>0.
                    pass
                    
            if not valid:
                continue
                
            # Now check if we can assign distinct multiples.
            # Build adjacency list for bipartite matching
            # Left: 0..n-1
            # Right: candidates (values)
            
            # We can just use backtracking
            # Since n=10, we can represent the used candidates as a set or bitmask?
            # Candidates can be large (up to 1000000+30000), so bitmask of values is not possible.
            # But the number of candidates is W+1.
            # If W is small, we can map candidate values to indices 0..W.
            
            candidates = []
            for x in range(A, A + W + 1):
                if x > 0:
                    candidates.append(x)
            
            if len(candidates) < n:
                continue
                
            # Map value to index
            val_to_idx = {v: i for i, v in enumerate(candidates)}
            
            # Adjacency list for left nodes
            adj = [[] for _ in range(n)]
            for i in range(n):
                d = divisors[i]
                for x in candidates:
                    if x % d == 0:
                        adj[i].append(val_to_idx[x])
                        
            # Check if any adj[i] is empty
            if any(len(a) == 0 for a in adj):
                continue
                
            # Backtracking to find perfect matching
            # match_right[j] = i if candidate j is matched to i, else -1
            match_right = [-1] * len(candidates)
            
            def dfs(i, visited):
                for j in adj[i]:
                    if visited[j]:
                        continue
                    visited[j] = True
                    if match_right[j] == -1 or dfs(match_right[j], visited):
                        match_right[j] = i
                        return True
                return False
            
            found = True
            for i in range(n):
                visited = [False] * len(candidates)
                if not dfs(i, visited):
                    found = False
                    break
                    
            if found:
                # Construct result
                result = [0] * n
                for j in range(len(candidates)):
                    if match_right[j] != -1:
                        i = match_right[j]
                        result[i] = candidates[j]
                        
                # Verify distinctness and positivity
                # result contains the integers
                return result
                
    # Fallback (should not happen)
    return divisors[:]
