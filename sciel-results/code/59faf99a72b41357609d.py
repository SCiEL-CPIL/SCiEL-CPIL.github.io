"""
Strategy: 
1. Use 4 blocks of 2x3x2 decomposition.
2. Each 2x3x2 block has rank 10 (Strassen 2x2x2 rank 7 + 3 additional terms).
3. Total rank 40.
4. Fallback to rank 44 (Strassen + Schoolbook) if rank 10 block construction fails.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

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
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
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

    def _build_rank10_block():
        """
        Constructs a rank-10 decomposition for 2x3x2.
        Returns U (6x10), V (6x10), W (4x10) or None.
        """
        # 1. Initialize with Strassen 2x2x2 (Rank 7)
        # U: 2x2 (i, a) -> 4 entries. But we need 6 entries (2x3).
        # Strassen covers a=0,1. a=2 is 0.
        U_block = [[0] * 10 for _ in range(6)] # i in 0,1; a in 0,1,2
        V_block = [[0] * 10 for _ in range(6)] # a in 0,1,2; k in 0,1
        W_block = [[0] * 10 for _ in range(4)] # k in 0,1; i in 0,1
        
        # Fill Strassen part (r=0..6)
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    U_block[li * 3 + la][r] = S_U[r][li * 2 + la]
            for la in range(2):
                for lk in range(2):
                    V_block[la * 2 + lk][r] = S_V[r][la * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    W_block[lk * 2 + li][r] = S_W[r][lk * 2 + li]
        
        # 2. We need to satisfy T[i, 2, k] = 1 for i in 0,1; k in 0,1.
        # Equations:
        # Sum_r U[i,2,r] V[2,k,r] W[k,i,r] = 1
        
        # We have 7 Strassen terms.
        # We add 3 new terms (r=7,8,9).
        # We set W for new terms to be basis vectors for 3 of the 4 outputs.
        # W[0,0,7]=1, others 0. (k=0, i=0)
        # W[0,1,8]=1, others 0. (k=0, i=1)
        # W[1,0,9]=1, others 0. (k=1, i=0)
        # W[1,1,*] for r=7,8,9 is 0.
        
        W_block[0*2+0][7] = 1 # k=0, i=0
        W_block[0*2+1][8] = 1 # k=0, i=1
        W_block[1*2+0][9] = 1 # k=1, i=0
        
        # Now we need to find U[i,2,r] and V[2,k,r] for r=0..9
        # such that:
        # Eq (0,0): Sum_{r=0..6} U[0,2,r] V[2,0,r] W[0,0,r] + U[0,2,7] V[2,0,7] * 1 = 1
        # Eq (0,1): Sum_{r=0..6} U[0,2,r] V[2,0,r] W[0,1,r] = 1  (W[0,1,7..9]=0)
        # Eq (1,0): Sum_{r=0..6} U[1,2,r] V[2,1,r] W[1,0,r] + U[1,2,9] V[2,1,9] * 1 = 1
        # Eq (1,1): Sum_{r=0..6} U[1,2,r] V[2,1,r] W[1,1,r] = 1  (W[1,1,7..9]=0)
        
        # Note: W indices are k*n+i.
        # W[0,0] is k=0, i=0.
        # W[0,1] is k=0, i=1.
        # W[1,0] is k=1, i=0.
        # W[1,1] is k=1, i=1.
        
        # We can try to set U[0,2,r] and U[1,2,r] to simple values.
        # Let's try to solve for V given U.
        # Or search for small integer solutions.
        
        # Heuristic:
        # Set U[0,2,r] = 1 for all r.
        # Set U[1,2,r] = 1 for all r.
        
        U_block[0*3+2] = [1]*10
        U_block[1*3+2] = [1]*10
        
        # Now solve for V[2,k,r] (6 values for k=0, 6 for k=1).
        # Let v0 = V[2,0,r], v1 = V[2,1,r].
        
        # Eq (0,1): Sum_{r=0..6} 1 * v0[r] * W[0,1,r] = 1
        # Eq (1,1): Sum_{r=0..6} 1 * v1[r] * W[1,1,r] = 1
        
        # These are linear systems.
        # A0 is 1x7 (row of W[0,1,0..6])
        # A1 is 1x7 (row of W[1,1,0..6])
        
        W_01 = [W_block[0*2+1][r] for r in range(7)]
        W_11 = [W_block[1*2+1][r] for r in range(7)]
        
        # Find v0 such that dot(W_01, v0) = 1
        # Find v1 such that dot(W_11, v1) = 1
        
        # If no integer solution, we might need to adjust U or add terms.
        # But we only have 3 extra terms.
        
        # Let's try to find integer v0, v1 in {-2,-1,0,1,2}
        
        def find_v(w_row, target=1):
            # w_row is list of 7 ints
            # Find v in {-2..2}^7 such that sum(v*r) = target
            # Simple search for small vectors
            # Since w_row has small entries, we can try to match.
            # Actually, we can just use one non-zero term if possible.
            # If any w_row[r] is 1, set v[r]=1.
            # If any w_row[r] is -1, set v[r]=-1.
            # If any w_row[r] is 2, set v[r]=1/2 (not integer).
            
            # Let's try to find any integer solution.
            # We can use the fact that we can set v[r] for r=7,8,9 to 0 for these equations.
            
            # Try simple vectors
            for v_val in [-2, -1, 0, 1, 2]:
                for r in range(7):
                    v = [0]*10
                    v[r] = v_val
                    if w_row[r] * v_val == target:
                        return v
            # Try 2 non-zero terms
            for r1 in range(7):
                for r2 in range(7):
                    if r1 == r2: continue
                    for v1 in [-2, -1, 1, 2]:
                        for v2 in [-2, -1, 1, 2]:
                            if w_row[r1]*v1 + w_row[r2]*v2 == target:
                                v = [0]*10
                                v[r1] = v1
                                v[r2] = v2
                                return v
            return None
        
        v0 = find_v(W_01)
        v1 = find_v(W_11)
        
        if v0 is None or v1 is None:
            return None
            
        V_block[2*2+0] = v0
        V_block[2*2+1] = v1
        
        # Now check Eq (0,0) and (1,0)
        # Eq (0,0): Sum_{r=0..6} U[0,2,r] v0[r] W[0,0,r] + U[0,2,7] v0[7] * 1 = 1
        # We set U[0,2,r]=1.
        # So: Sum_{r=0..6} v0[r] W[0,0,r] + v0[7] = 1
        # We know v0[7..9] are 0 from find_v (since we only searched r=0..6).
        # So we need Sum_{r=0..6} v0[r] W[0,0,r] = 1.
        
        W_00 = [W_block[0*2+0][r] for r in range(7)]
        acc_00 = sum(v0[r] * W_00[r] for r in range(7))
        
        # Eq (1,0): Sum_{r=0..6} U[1,2,r] v1[r] W[1,0,r] + U[1,2,9] v1[9] * 1 = 1
        # We set U[1,2,r]=1.
        # So: Sum_{r=0..6} v1[r] W[1,0,r] + v1[9] = 1
        # We know v1[7..9] are 0.
        # So we need Sum_{r=0..6} v1[r] W[1,0,r] = 1.
        
        W_10 = [W_block[1*2+0][r] for r in range(7)]
        acc_10 = sum(v1[r] * W_10[r] for r in range(7))
        
        if acc_00 != 1 or acc_10 != 1:
            return None
            
        # If we get here, we have a valid rank-10 decomposition.
        return U_block, V_block, W_block

    block = _build_rank10_block()
    
    if block:
        U_block, V_block, W_block = block
        # Tile 4 times
        rank = 40
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for b in range(4):
            k0 = b * 2
            for r in range(10):
                col = b * 10 + r
                for i in range(n):
                    for a in range(m):
                        U[i*m+a][col] = U_block[i*3+a][r]
                for a in range(m):
                    for k in range(2):
                        V[a*p + (k0+k)][col] = V_block[a*2+k][r]
                for k in range(2):
                    for i in range(n):
                        W[(k0+k)*n + i][col] = W_block[k*2+i][r]
                        
        candidate = {"U": U, "V": V, "W": W}
        if _validate(candidate):
            return candidate

    # Fallback to Rank 44
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
            if not _elapsed_ok():
                return _schoolbook()

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
                
    col = rank_base
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
