import time

# Global cache for the result
_RESULT = None

def _check_validity(S, K):
    # K is list of sets, K[c] is set of sums of subsets of S of size c
    # This is just for initial setup or if we don't pass K
    pass

def _try_add(S, x, K):
    # S: current set (list)
    # x: candidate
    # K: list of sets of sums for current S, K[c] contains sums of subsets of size c
    # Returns new K if valid, else None
    
    n = len(S)
    # Max cardinality in S is n
    # New max cardinality is n+1
    
    # New sums for each cardinality c in 1..n+1
    # NewSums[c] = { x + s | s in K[c-1] }
    
    # We need to check collisions.
    # 1. NewSums[c] vs Old K[c'] for c' != c
    # 2. NewSums[c] vs NewSums[c'] for c' != c
    
    # Note: K[0] = {0}
    # NewSums[1] = {x}
    # NewSums[c] for c>1 depends on K[c-1]
    
    # Let's construct NewSums
    new_K = [set() for _ in range(n + 2)]
    
    # Copy old K
    for c in range(n + 1):
        new_K[c].update(K[c])
        
    # Calculate new sums and check validity on the fly
    # We iterate c from 1 to n+1
    # NewSums[c] comes from K[c-1]
    
    # To check collisions efficiently:
    # For each c, let New_c = { x + s for s in K[c-1] }
    # Check New_c against all existing sums in new_K[c'] for c' != c.
    # But new_K is being built.
    # Better: 
    # 1. Compute all New_c sets.
    # 2. Check New_c vs Old K[c'] (c' != c).
    # 3. Check New_c vs New_c' (c' != c).
    # 4. If valid, merge into new_K.
    
    New_Sums = [set() for _ in range(n + 2)]
    for c in range(1, n + 2):
        if c - 1 <= n:
            # K[c-1] exists
            for s in K[c-1]:
                New_Sums[c].add(x + s)
        else:
            # c-1 > n, so K[c-1] is empty, New_Sums[c] is empty
            pass
            
    # Check 1: New_Sums[c] vs Old K[c']
    for c in range(1, n + 2):
        if not New_Sums[c]:
            continue
        for c_prime in range(n + 1):
            if c_prime == c:
                continue
            # Intersection of New_Sums[c] and K[c_prime]
            if New_Sums[c] & K[c_prime]:
                return None
                
    # Check 2: New_Sums[c] vs New_Sums[c_prime]
    # Note: If New_Sums[c] and New_Sums[c_prime] intersect, it means there exist T1, T2 subsets of S
    # such that x + sum(T1) = x + sum(T2) => sum(T1) = sum(T2).
    # Since T1 has size c-1 and T2 has size c_prime-1.
    # If c != c_prime, then |T1| != |T2|.
    # If sum(T1) = sum(T2) and |T1| != |T2|, then S was already invalid?
    # No, S is valid, so K_S[|T1|] and K_S[|T2|] are disjoint.
    # So sum(T1) cannot be in K_S[|T2|].
    # Thus New_Sums[c] and New_Sums[c_prime] should be disjoint automatically if S is valid?
    # Let's verify.
    # New_Sums[c] = { x + s | s in K[c-1] }
    # New_Sums[c'] = { x + s' | s' in K[c'-1] }
    # Intersection implies x + s = x + s' => s = s'.
    # s in K[c-1] and s in K[c'-1].
    # Since S is valid, K[c-1] and K[c'-1] are disjoint if c-1 != c'-1.
    # So s can only be in both if c-1 == c'-1 => c == c'.
    # So Check 2 is redundant if S is valid.
    
    # So we only need Check 1.
    
    # If valid, return new_K
    # We can just return the merged K
    for c in range(n + 2):
        new_K[c].update(New_Sums[c])
        
    return new_K

def _dfs(S, K, candidates, current_size, max_size, deadline):
    global _RESULT
    if time.time() > deadline:
        return
    
    if current_size > max_size:
        # Update best
        _RESULT = list(S)
        # We can't just return, we want to find even larger?
        # The problem asks to maximize.
        # If we find size 10, we should try to find 11.
        # So we continue searching.
        pass
        
    if not candidates:
        return
        
    # Heuristic: try smaller numbers first? 
    # Superincreasing starts with small numbers.
    # Let's iterate through candidates.
    
    for i, x in enumerate(candidates):
        if time.time() > deadline:
            return
            
        new_K = _try_add(S, x, K)
        if new_K is not None:
            # Add x
            S.add(x)
            _dfs(S, new_K, candidates[i+1:], current_size + 1, max_size, deadline)
            S.remove(x)
            
        # Option to exclude x is implicit by moving to next i

def construct():
    global _RESULT
    if _RESULT is not None:
        return _RESULT
        
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Initial state
    S = set()
    # K[0] = {0}, K[c] = empty for c>0
    # Max possible size is small, let's say 20
    K = [set() for _ in range(20)]
    K[0].add(0)
    
    candidates = list(range(1, 257))
    
    # We want to maximize size.
    # A simple DFS might get stuck in a local optimum or take too long if not pruned well.
    # However, the constraint is very tight.
    # Let's try to find the maximum possible size.
    
    _dfs(S, K, candidates, 0, 0, deadline)
    
    if _RESULT is None:
        # Fallback to known good solution
        _RESULT = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    # Ensure types are correct
    return [int(x) for x in _RESULT]

# Run search on import to populate cache?
# "Keep the function names and signatures of the seed program."
# "Each evaluation must finish within 60 s."
# If I run it on import, it might be faster for multiple calls?
# But usually construct() is called once.
# I will run it inside construct() but cache the result globally.
