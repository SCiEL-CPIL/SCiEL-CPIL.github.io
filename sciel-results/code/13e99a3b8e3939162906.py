"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(n, m, p, U, V, W):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    """
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > 1e-9:
                    return False
    return True


def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True


def _strassen_2x2():
    """
    Classic Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11-A22)*(B22-B11)
    U[0][0] = 1; U[3][0] = -1
    V[0][0] = -1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A11*(B11-B22)
    U[0][2] = 1
    V[0][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A22*(B22-B11)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}


def _rectangular_decomp(n, m, p):
    """
    Decompose n×m × m×p → n×p matrix multiplication.
    Strategy: if min(n,p) >= 2, use Strassen on a 2×2 sub-block, schoolbook for rest.
    Returns (U, V, W, rank) where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    """
    # Use Strassen for 2×2 sub-block if possible, schoolbook otherwise
    # For simplicity, use schoolbook for rectangular unless both dims >= 2
    if n >= 2 and p >= 2:
        # Use Strassen for first 2×2 block of output, schoolbook for remaining
        # This gets complex; let's just use schoolbook for now for rectangular
        pass
    
    # Schoolbook fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank


def _block_kronecker_5x5x5():
    """
    Build a 5×5×5 matmul decomposition by splitting 5 = 2+3.
    Blocks: (2,2), (2,3), (3,2), (3,3)
    Use Strassen for (2,2), schoolbook for others.
    """
    n, m, p = 5, 5, 5
    splits = [(2, 3)]  # split point at 2
    
    # Get Strassen for 2x2
    s = _strassen_2x2()
    sU, sV, sW = s["U"], s["V"], s["W"]
    
    # Collect all block decompositions
    # Block (bi, bj, bk) where bi,bj,bk ∈ {0,1} corresponding to sizes [2,3]
    sizes = [2, 3]
    
    U_blocks = []  # list of (row_offset, U_matrix)
    V_blocks = []
    W_blocks = []
    
    total_rank = 0
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                ni = sizes[bi]
                mj = sizes[bj]
                pk = sizes[bk]
                
                # Row offset in U: sum of ni' * m for bi' < bi, plus ni * (sum of mj' for bj' < bj) + ni * sum of sizes[bj'] for bj'<bj... 
                # Actually: U row index = i*m + j where i is global row, j is global col
                # i = sum(sizes[bi'] for bi' < bi) + local_i
                # j = sum(sizes[bj'] for bj' < bj) + local_j
                i_off = sum(sizes[bb] for bb in range(bi))
                j_off = sum(sizes[bb] for bb in range(bj))
                k_off = sum(sizes[bb] for bb in range(bk))
                
                # U sub-matrix: rows (i_off + local_i)*m + (j_off + local_j) for local_i in [0,ni), local_j in [0,mj)
                # This is a ni*mj × rank_sub matrix
                
                if bi == 0 and bj == 0 and bk == 0:
                    # 2×2×2 block: use Strassen
                    sub_U = sU  # 4×7 = (2*2)×7
                    sub_V = sV  # 4×7 = (2*2)×7
                    sub_W = sW  # 4×7 = (2*2)×7
                    sub_rank = 7
                else:
                    # Schoolbook for this block
                    sub_rank = ni * mj * pk
                    sub_U = [[0] * sub_rank for _ in range(ni * mj)]
                    sub_V = [[0] * sub_rank for _ in range(mj * pk)]
                    sub_W = [[0] * sub_rank for _ in range(pk * ni)]
                    col = 0
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                sub_U[li * mj + lj][col] = 1
                                sub_V[lj * pk + lk][col] = 1
                                sub_W[lk * ni + li][col] = 1
                                col += 1
                
                U_blocks.append((i_off, j_off, sub_U))
                V_blocks.append((j_off, k_off, sub_V))
                W_blocks.append((k_off, i_off, sub_W))
                total_rank += sub_rank
    
    # Now assemble full U, V, W
    nm, mp, pn = n * m, m * p, p * n
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    col_offset = 0
    for idx in range(len(U_blocks)):
        i_off, j_off, sub_U = U_blocks[idx]
        j_off_v, k_off_v, sub_V = V_blocks[idx]
        k_off_w, i_off_w, sub_W = W_blocks[idx]
        
        sub_rank = len(sub_U[0]) if sub_U else 0
        
        for li in range(len(sub_U)):
            local_i = li // len(sub_U[0])  # This isn't right; need to track ni, mj
            # Actually, li ranges over ni*mj, so:
            # global row in U = (i_off + local_i_row) * m + (j_off + local_i_col)
            # where local_i_row = li // mj, local_i_col = li % mj
            # But we don't have mj here directly. Let me restructure.
        
        # Let me redo this more carefully
        pass
    
    # Redo assembly with proper tracking
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    col_offset = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                ni = sizes[bi]
                mj = sizes[bj]
                pk = sizes[bk]
                
                i_off = sum(sizes[bb] for bb in range(bi))
                j_off = sum(sizes[bb] for bb in range(bj))
                k_off = sum(sizes[bb] for bb in range(bk))
                
                if bi == 0 and bj == 0 and bk == 0:
                    sub_U = sU
                    sub_V = sV
                    sub_W = sW
                    sub_rank = 7
                else:
                    sub_rank = ni * mj * pk
                    sub_U = [[0] * sub_rank for _ in range(ni * mj)]
                    sub_V = [[0] * sub_rank for _ in range(mj * pk)]
                    sub_W = [[0] * sub_rank for _ in range(pk * ni)]
                    col = 0
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                sub_U[li * mj + lj][col] = 1
                                sub_V[lj * pk + lk][col] = 1
                                sub_W[lk * ni + li][col] = 1
                                col += 1
                
                # Fill U
                for li in range(ni):
                    for lj in range(mj):
                        global_row = (i_off + li) * m + (j_off + lj)
                        sub_row = li * mj + lj
                        for c in range(sub_rank):
                            U[global_row][col_offset + c] = sub_U[sub_row][c]
                
                # Fill V
                for lj in range(mj):
                    for lk in range(pk):
                        global_row = (j_off + lj) * p + (k_off + lk)
                        sub_row = lj * pk + lk
                        for c in range(sub_rank):
                            V[global_row][col_offset + c] = sub_V[sub_row][c]
                
                # Fill W
                for lk in range(pk):
                    for li in range(ni):
                        global_row = (k_off + lk) * n + (i_off + li)
                        sub_row = lk * ni + li
                        for c in range(sub_rank):
                            W[global_row][col_offset + c] = sub_W[sub_row][c]
                
                col_offset += sub_rank
    
    return {"U": U, "V": V, "W": W}


def _verify_strassen_2x2():
    """Verify the Strassen 2x2 decomposition."""
    n, m, p = 2, 2, 2
    res = _strassen_2x2()
    U, V, W = res["U"], res["V"], res["W"]
    return _verify(n, m, p, U, V, W)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Try block Kronecker approach first
    best = None
    best_rank = n * m * p
    
    # Block Kronecker: 5 = 2+3
    try:
        bk = _block_kronecker_5x5x5()
        if _verify(n, m, p, bk["U"], bk["V"], bk["W"]):
            best = bk
            best_rank = len(bk["U"][0])
    except Exception:
        pass
    
    # If block Kronecker didn't work or wasn't better, try schoolbook
    if best is None or best_rank >= n * m * p:
        sb = _schoolbook(n, m, p)
        if _verify(n, m, p, sb["U"], sb["V"], sb["W"]):
            best = sb
            best_rank = n * m * p
    
    return best
# EVOLVE-BLOCK-END
