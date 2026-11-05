import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
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
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2
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

    # Build 44-term decomposition
    rank_44 = 44
    U_44 = [[0] * rank_44 for _ in range(nm)]
    V_44 = [[0] * rank_44 for _ in range(mp)]
    W_44 = [[0] * rank_44 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(n):
        for k in range(p):
            U_44[i * m + 2][col] = 1
            V_44[2 * p + k][col] = 1
            W_44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    if not _validate(candidate_44):
        return _schoolbook()

    # Try to find a better decomposition for 2x3x2 (rank < 9)
    # If we find rank 8 for 2x3x2, we can get rank 32 for 2x3x8
    def _find_23x2_decomposition():
        """Try to find a rank-8 decomposition for 2x3x2."""
        # U: (6, rank), V: (6, rank), W: (4, rank)
        # Indices: U[i*3+a], V[a*2+k], W[k*2+i]
        # Need: sum_r U[i*3+a,r]*V[a*2+k,r]*W[k*2+i,r] = 1 for all i,a,k
        
        # Try rank 8
        rank = 8
        # Initialize with a heuristic
        U = [[0]*rank for _ in range(6)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(4)]
        
        # Try a structured approach
        # For each of the 8 terms, assign which (i,a,k) tuples it covers
        # This is complex, so let's try a simple pattern first
        
        # Pattern: use 4 terms for a=0 (2x2x2 Strassen-like), 
        # and 4 terms for a=1,2 combined
        
        # Actually, let's try: 3 terms for each a value, but that's 9.
        # Let's try to share terms between a values.
        
        # Try this pattern:
        # Term 0: U[0,0]=1, U[3,0]=1; V[0,0]=1; W[0,0]=1, W[3,0]=1 -> covers (0,0,0),(1,0,0) with V[0,0]
        # This is getting complex. Let me try a direct numerical search.
        
        # Set up as a system of equations
        # Variables: U[0..5, 0..7], V[0..5, 0..7], W[0..3, 0..7]
        # Constraints: for each (i,a,k), sum_r U[i*3+a,r]*V[a*2+k,r]*W[k*2+i,r] = 1
        
        # This is a system of 12 equations with 12*8 = 96 unknowns (but bilinear)
        # Let's try to solve it iteratively
        
        import random
        random.seed(42)
        
        # Initialize randomly
        for r in range(rank):
            for idx in range(6):
                U[idx][r] = random.choice([-1, 0, 1])
                V[idx][r] = random.choice([-1, 0, 1])
            for idx in range(4):
                W[idx][r] = random.choice([-1, 0, 1])
        
        # Check if it works
        def _check(U, V, W):
            for i in range(2):
                for a in range(3):
                    for k in range(2):
                        acc = 0
                        for r in range(rank):
                            acc += U[i*3+a][r] * V[a*2+k][r] * W[k*2+i][r]
                        if acc != 1:
                            return False
            return True
        
        if _check(U, V, W):
            return U, V, W
        
        # Try a more structured initialization
        # Use the fact that we can separate the a dimension
        # For a=0: need 2x2x2 decomposition (rank 7)
        # For a=1: need 2x1x2 decomposition (rank 4)  
        # For a=2: need 2x1x2 decomposition (rank 4)
        # Total: 15, but we can share the U and W parts
        
        # Actually, let's just try the 44-term solution and see if we can prune it
        # For now, return None to indicate no better solution found
        return None
    
    result = _find_23x2_decomposition()
    if result is not None:
        U_232, V_232, W_232 = result
        rank_232 = len(U_232[0])
        # Extend to 2x3x8 by repeating 4 times
        rank_new = rank_232 * 4
        U_new = [[0]*rank_new for _ in range(nm)]
        V_new = [[0]*rank_new for _ in range(mp)]
        W_new = [[0]*rank_new for _ in range(pn)]
        
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(rank_232):
                for i in range(2):
                    for a in range(3):
                        u_val = U_232[i*3+a][r]
                        if u_val != 0:
                            U_new[i*m+a][block_idx*rank_232 + r] = u_val
                for a in range(3):
                    for lk in range(2):
                        v_val = V_232[a*2+lk][r]
                        if v_val != 0:
                            V_new[a*p + (k0+lk)][block_idx*rank_232 + r] = v_val
                for lk in range(2):
                    for i in range(2):
                        w_val = W_232[lk*2+i][r]
                        if w_val != 0:
                            W_new[(k0+lk)*n + i][block_idx*rank_232 + r] = w_val
        
        candidate_new = {"U": U_new, "V": V_new, "W": W_new}
        if _validate(candidate_new):
            return candidate_new
    
    return candidate_44
