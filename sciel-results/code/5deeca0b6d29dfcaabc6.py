import time
from fractions import Fraction
# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
    # Deadline guard
    _start_time = time.time()
    _deadline = 140.0  # 140s out of 150s budget
    
    def _time_ok():
        return (time.time() - _start_time) < _deadline
    
    # Schoolbook fallback (Rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U_frac[i * m + a]
                    v_row = V_frac[a * p + k]
                    w_row = W_frac[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    S_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    S_W = [
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]

    # Build the 28-term Strassen basis for the 2x2x8 part (a=0,1)
    rank_base = 28
    U_base = [[0] * rank_base for _ in range(n * m)]
    V_base = [[0] * rank_base for _ in range(m * p)]
    W_base = [[0] * rank_base for _ in range(p * n)]
    
    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val != 0:
                        U_base[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val != 0:
                        V_base[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val != 0:
                        W_base[(k0 + lk) * n + li][col] = w_val
            col += 1

    # Construct Rank 44 candidate: Strassen for a=0,1 + Schoolbook for a=2
    rank_44 = 44
    U = [[0] * rank_44 for _ in range(n * m)]
    V = [[0] * rank_44 for _ in range(m * p)]
    W = [[0] * rank_44 for _ in range(p * n)]
    
    # Copy base
    for r in range(rank_base):
        for i in range(n):
            for a in range(2):
                U[i*m+a][r] = U_base[i*m+a][r]
        for a in range(2):
            for k in range(p):
                V[a*p+k][r] = V_base[a*p+k][r]
        for k in range(p):
            for i in range(n):
                W[k*n+i][r] = W_base[k*n+i][r]
                
    # Add a=2 part
    col = rank_base
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    candidate_44 = {"U": U, "V": V, "W": W}
    
    if _validate(candidate_44):
        if not _time_ok():
            return candidate_44
        
        # Laser Method: Try to find a lower-rank decomposition by searching
        # for linear dependencies that allow rank reduction
        # Strategy: Try rank 43, 42, 41 by finding if any rank-1 term can be
        # expressed as a linear combination of others while preserving the tensor
        
        best_candidate = candidate_44
        best_rank = 44
        
        # Try to reduce rank by checking if we can merge or eliminate terms
        # For each pair of terms, check if their sum can be replaced by a single term
        # This is equivalent to checking if the tensor has rank < 44
        
        # More systematic approach: Try random linear combinations of the 44 terms
        # and check if we can get a valid rank-43 decomposition
        
        import random
        random.seed(42)
        
        # Try a few different approaches to reduce rank
        for target_rank in [43, 42, 41]:
            if not _time_ok():
                break
            
            # Try to find a valid decomposition of rank target_rank
            # by modifying the rank-44 decomposition
            success = False
            
            # Approach 1: Try to set one column to zero and adjust others
            # This won't work directly, so try linear combinations
            
            # Approach 2: Use the fact that for matrix multiplication, there are
            # known lower bounds. For 2x3x8, the rank is known to be at most 44
            # but may be lower.
            
            # Try a structured search: modify the a=2 schoolbook part
            # The a=2 part has 16 terms. Can we do better than schoolbook for 2x1x8?
            # 2x1x8 multiplication has rank 16 (schoolbook is optimal for 1xn or nx1)
            
            # So the 44 rank is likely optimal given the block structure.
            # But maybe we can mix the a=0,1 and a=2 parts more cleverly.
            
            # Try: Use Strassen-like ideas on the full 2x3x8 tensor
            # This is complex, so try a heuristic search
            
            # For now, return the rank-44 solution if no better is found
            pass
        
        return best_candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
