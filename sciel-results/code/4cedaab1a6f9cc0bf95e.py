"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    p = 1009
    N = p - 1  # 1008
    
    # Precompute modular inverses using Fermat's little theorem (p is prime)
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    # We need to find the longest interval [s, e] such that no pair {x, inv[x]} 
    # is fully contained in [s, e].
    # Let L_x = min(x, inv[x]) and R_x = max(x, inv[x]).
    # The interval [s, e] is invalid if there exists x such that s <= L_x and R_x <= e.
    # So for a fixed s, the maximum valid e is min{ R_x | L_x >= s } - 1.
    
    # min_R_at_L[L] stores the minimum R_x for all x such that L_x == L.
    min_R_at_L = [float('inf')] * (p + 2)
    
    for x in range(1, p):
        y = inv[x]
        L = min(x, y)
        R = max(x, y)
        if R < min_R_at_L[L]:
            min_R_at_L[L] = R
            
    # Compute M(s) = min{ R_x | L_x >= s } using suffix minimums.
    # M(s) = min( min_R_at_L[s], M(s+1) )
    M = [float('inf')] * (p + 2)
    # M[p] is infinity, M[p+1] is infinity
    # We iterate from p-1 down to 1
    for s in range(p - 1, 0, -1):
        if min_R_at_L[s] < M[s + 1]:
            M[s] = min_R_at_L[s]
        else:
            M[s] = M[s + 1]
            
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        # Check time just in case, though this loop is O(N)
        if time.time() > deadline - 2.0:
            break
            
        max_e = M[s] - 1
        
        # Ensure max_e is within bounds [s, p-1]
        if max_e > p - 1:
            max_e = p - 1
            
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    return [best_start, best_len]

# EVOLVE-BLOCK-END
