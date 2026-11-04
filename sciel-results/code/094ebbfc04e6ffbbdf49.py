import time
from fractions import Fraction
# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    
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
        
        # Use integer arithmetic if possible, or Fraction for safety
        # Since we only generate ints and 0.5, we can multiply by 2 to check
        # Or just use Fraction
        
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
        return candidate_44
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
