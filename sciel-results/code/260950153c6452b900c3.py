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

    # Schoolbook fallback
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

    # Build 28-term Strassen basis for 2x2x8
    rank_s = 28
    U_s = [[0] * rank_s for _ in range(n * m)]
    V_s = [[0] * rank_s for _ in range(m * p)]
    W_s = [[0] * rank_s for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_s[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_s[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_s[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Try to absorb 2x1x8 part: find V[2,k,r] for r in 0..27
    # such that sum_r V[2,k,r] * W_s[k*n+i, r] = 1 for all i,k
    # with U[i,2,r] = 1 for all i,r
    
    U_ext = [[1] * rank_s for _ in range(n)]
    V_ext = [[0] * rank_s for _ in range(p)]
    
    success = True
    for k in range(p):
        # A_k: n x rank_s, A_k[i,r] = W_s[k*n+i, r]
        # We want to solve A_k * v_k = 1
        # Using exact arithmetic
        A_k = [[Fraction(W_s[k * n + i][r]) for r in range(rank_s)] for i in range(n)]
        b_k = [Fraction(1) for _ in range(n)]
        
        # Solve A_k v = b_k
        # Since it's underdetermined, we can set free variables to 0 or 1/2
        # Let's try to find a solution with half-integer entries
        # We can use a simple greedy approach or Gaussian elimination with Fractions
        
        # Gaussian elimination on [A_k | b_k]
        # Augmented matrix
        aug = [row[:] + [b_k[i]] for i, row in enumerate(A_k)]
        
        # Perform elimination
        rows = len(aug)
        cols = rank_s
        pivot = 0
        for r in range(rows):
            if pivot >= cols:
                break
            # Find pivot
            found = False
            for i in range(r, rows):
                if aug[i][pivot] != 0:
                    found = True
                    break
            if not found:
                continue
            # Swap
            if i != r:
                aug[r], aug[i] = aug[i], aug[r]
            # Normalize
            div = aug[r][pivot]
            for j in range(pivot, cols + 1):
                aug[r][j] = aug[r][j] / div
            # Eliminate
            for i in range(rows):
                if i != r:
                    factor = aug[i][pivot]
                    if factor != 0:
                        for j in range(pivot, cols + 1):
                            aug[i][j] = aug[i][j] - factor * aug[r][j]
            pivot += 1
        
        # Check for consistency
        for i in range(rows):
            # Check if row is [0 ... 0 | b] with b != 0
            if all(aug[i][j] == 0 for j in range(cols)) and aug[i][cols] != 0:
                success = False
                break
        if not success:
            break
            
        # Construct a particular solution
        # Set free variables to 0
        v_k = [Fraction(0) for _ in range(rank_s)]
        for i in range(rows):
            # Find the pivot column for this row
            for j in range(cols):
                if aug[i][j] != 0:
                    v_k[j] = aug[i][cols]
                    break
        
        # Check if v_k has half-integer entries
        valid = True
        for val in v_k:
            # val is a Fraction
            # Check if 2*val is an integer
            if (val * 2).denominator != 1:
                valid = False
                break
        if not valid:
            # Try to adjust by adding null space vectors?
            # For now, mark as failed
            success = False
            break
        
        V_ext[k] = [float(val) for val in v_k]
    
    if success:
        # Build rank-28 candidate
        U_final = [row[:] for row in U_s]
        V_final = [row[:] for row in V_s]
        W_final = [row[:] for row in W_s]
        
        for i in range(n):
            for r in range(rank_s):
                U_final[i * m + 2][r] = U_ext[i][r]
        for k in range(p):
            for r in range(rank_s):
                V_final[2 * p + k][r] = V_ext[k][r]
        
        candidate = {"U": U_final, "V": V_final, "W": W_final}
        if _validate(candidate):
            return candidate
    
    # Fallback: 44-term
    rank_44 = 44
    U44 = [[0] * rank_44 for _ in range(n * m)]
    V44 = [[0] * rank_44 for _ in range(m * p)]
    W44 = [[0] * rank_44 for _ in range(p * n)]
    
    for r in range(rank_s):
        for i in range(n):
            for a in range(2):
                U44[i * m + a][r] = U_s[i * m + a][r]
        for a in range(2):
            for k in range(p):
                V44[a * p + k][r] = V_s[a * p + k][r]
        for k in range(p):
            for i in range(n):
                W44[k * n + i][r] = W_s[k * n + i][r]
    
    col = rank_s
    for i in range(n):
        for k in range(p):
            U44[i * m + 2][col] = 1
            V44[2 * p + k][col] = 1
            W44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
    
    candidate = {"U": U44, "V": V44, "W": W44}
    if _validate(candidate):
        return candidate
    return _schoolbook()

# EVOLVE-BLOCK-END
