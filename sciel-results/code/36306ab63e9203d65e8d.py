import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 256
    
    # Helper to check if adding x to S (set) is valid
    # S is a set of ints
    # Returns True if S U {x} is 3-AP free
    def is_valid_addition(S, x):
        # Check if x forms a 3-AP with any pair in S
        # Or if x is the midpoint of a pair in S
        # Or if x is an endpoint of a pair in S
        
        # Case 1: x is midpoint. y, x, z. y+z=2x.
        # y, z in S.
        # For each y in S, check if 2x-y is in S.
        # Case 2: x is endpoint. x, y, z. x+z=2y.
        # y, z in S.
        # For each y in S, check if 2y-x is in S.
        # Note: if 2y-x == x, then y=x, but x not in S.
        # If 2y-x == z, we check if z in S.
        
        # Optimization:
        # Iterate y in S.
        # 1. mid = (x+y)/2. If integer and mid in S -> invalid.
        # 2. end1 = 2*x - y. If in range and end1 in S -> invalid.
        # 3. end2 = 2*y - x. If in range and end2 in S -> invalid.
        
        # Wait, if we check "mid in S", that covers the case where x and y are endpoints.
        # If we check "end1 in S", that covers the case where x is midpoint?
        # Let's trace:
        # AP: a, b, c. a+c=2b.
        # If x is added:
        # If x=b (midpoint): a, x, c. a,c in S. a+c=2x.
        #   Check: for a in S, is 2x-a in S?
        # If x=a (endpoint): x, b, c. b,c in S. x+c=2b -> c=2b-x.
        #   Check: for b in S, is 2b-x in S?
        # If x=c (endpoint): a, b, x. a,b in S. a+x=2b -> a=2b-x.
        #   Same as above.
        
        # So we just need:
        # For each y in S:
        #   z1 = 2*x - y. If 1 <= z1 <= N and z1 in S: return False
        #   z2 = 2*y - x. If 1 <= z2 <= N and z2 in S: return False
        
        # Note: z1 == z2 implies 2x-y = 2y-x => 3x=3y => x=y. But x not in S.
        # So we check both.
        
        for y in S:
            z1 = 2 * x - y
            if 1 <= z1 <= N:
                if z1 in S:
                    return False
            z2 = 2 * y - x
            if 1 <= z2 <= N:
                if z2 in S:
                    return False
        return True

    # Greedy construction
    S = set()
    # Prioritize small numbers for higher reciprocal weight
    for x in range(1, N + 1):
        if is_valid_addition(S, x):
            S.add(x)
    
    def get_score(S):
        return sum(1.0 / x for x in S)
    
    best_S = set(S)
    best_score = get_score(best_S)
    
    # Local Search: Hill Climbing
    # Try swaps
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try single swaps: remove x, add y
        for x in list(best_S):
            if time.time() > deadline - 5.0:
                break
            S_minus_x = best_S - {x}
            score_minus_x = best_score - 1.0 / x
            for y in range(1, N + 1):
                if y in best_S:
                    continue
                # Check if S_minus_x U {y} is valid
                if is_valid_addition(S_minus_x, y):
                    new_score = score_minus_x + 1.0 / y
                    if new_score > best_score + 1e-12:
                        best_S = S_minus_x | {y}
                        best_score = new_score
                        improved = True
                        break # Restart loop with new best_S
        
        # Try double swaps if time permits?
        # Might be too slow for N=256 if not optimized.
        # Single swaps are O(N^3) roughly. 256^3 is ~16M.
        # With Python, 16M iterations might take 10-20s.
        # We have 85s budget.
        # Let's try to optimize or limit double swaps.
        
        if time.time() < deadline - 10.0 and not improved:
            # Try 2-swaps
            # Remove x1, x2. Add y1, y2.
            # This is complex. Let's stick to single swaps first and see.
            # If single swaps don't improve, try random restarts or more aggressive search.
            pass

    # Final verification
    def is_valid(S):
        for x in S:
            for y in S:
                if x >= y: continue
                z1 = 2*x - y
                if 1 <= z1 <= N and z1 in S: return False
                z2 = 2*y - x
                if 1 <= z2 <= N and z2 in S: return False
        return True

    if not is_valid(best_S):
        # Fallback to greedy
        best_S = set()
        for x in range(1, N + 1):
            if is_valid_addition(best_S, x):
                best_S.add(x)
    
    if not best_S:
        best_S = {1}
        
    return sorted(best_S)
