"""Matrix multiplication 4,5,6 - rank-minimal tensor decomposition."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np
from fractions import Fraction
import itertools

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _decompose_2x2x3():
    """Decompose 2x2 * 2x3. Rank 11 via Strassen(2x2x2) + 2x1*1x3 outer."""
    U_s, V_s, W_s = _strassen_2x2x2()
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(6)]
    
    for r in range(7):
        for i in range(4):
            if U_s[i][r] != 0:
                U[i][r] = U_s[i][r]
        for j in range(4):
            if V_s[j][r] != 0:
                V[j][r] = V_s[j][r]
        for k in range(4):
            if W_s[k][r] != 0:
                W[k][r] = W_s[k][r]
    
    for i in range(2):
        for k in range(3):
            r = 7 + i*3 + k
            U[i*2][r] = 1
            V[2+k][r] = 1
            W[k*2+i][r] = 1
    
    return U, V, W

def _decompose_2x2x4():
    """Decompose 2x2 * 2x4. Rank 15 via Strassen(2x2x2) + Strassen(2x2x2) + 2x1*1x2 outer."""
    U_s, V_s, W_s = _strassen_2x2x2()
    U = [[0]*15 for _ in range(4)]
    V = [[0]*15 for _ in range(8)]
    W = [[0]*15 for _ in range(8)]
    
    for r in range(7):
        for i in range(4):
            if U_s[i][r] != 0:
                U[i][r] = U_s[i][r]
        for j in range(4):
            if V_s[j][r] != 0:
                V[j][r] = V_s[j][r]
        for k in range(4):
            if W_s[k][r] != 0:
                W[k][r] = W_s[k][r]
    
    for r in range(7):
        rr = 7 + r
        for i in range(4):
            if U_s[i][r] != 0:
                U[i][rr] = U_s[i][r]
        for j in range(4):
            if V_s[j][r] != 0:
                V[4+j][rr] = V_s[j][r]
        for k in range(4):
            if W_s[k][r] != 0:
                W[4+k][rr] = W_s[k][r]
    
    for i in range(2):
        for k in range(2):
            r = 14
            U[i*2][r] = 1
            V[2+k][r] = 1
            W[k*2+i][r] = 1
    
    return U, V, W

def _block_decompose(n, m, p):
    """
    General block decomposition for n x m * m x p.
    Uses Strassen for 2x2x2 blocks, specialized decompositions for other 2x2xk blocks,
    and outer products for small blocks.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Precompute small decompositions
    decomp_223 = _decompose_2x2x3()
    decomp_224 = _decompose_2x2x4()
    
    # Try all ways to split into 2x2, 2x3, 2x4, etc.
    # For simplicity, try splitting each dimension into 2s and 1s
    # and use the best available decomposition for each block
    
    best_rank = n * m * p
    best_result = None
    
    # Generate all splits into 1s and 2s
    def gen_splits(total):
        if total == 0:
            return [[]]
        if total == 1:
            return [[1]]
        if total == 2:
            return [[2], [1, 1]]
        if total == 3:
            return [[2, 1], [1, 2]]
        if total == 4:
            return [[2, 2], [2, 1, 1], [1, 2, 1], [1, 1, 2], [1, 1, 1, 1]]
        return []
    
    a_splits = gen_splits(n)
    b_col_splits = gen_splits(m)
    c_splits = gen_splits(p)
    
    # For each combination of splits
    for a_sp in a_splits:
        for b_col_sp in b_col_splits:
            for c_sp in c_sp:
                # b_row_splits must match b_col_splits (since middle dimension is m)
                # Actually b_row_splits is for the rows of B, which is m
                # and b_col_splits is for the columns of B, which is p
                # Wait, the structure is:
                # A is n x m, B is m x p, C is n x p
                # A blocks: n x m, split n into a_sp, m into b_col_sp
                # B blocks: m x p, split m into b_row_sp, p into c_sp
                # C blocks: n x p, split n into c_row_sp, p into c_col_sp
                # For consistency: b_col_sp (from A) == b_row_sp (from B)
                # and c_row_sp (from C) == a_sp (from A)
                # and c_col_sp (from C) == c_sp (from B)
                
                # So we need:
                # a_sp (split of n)
                # b_col_sp (split of m)
                # b_row_sp = b_col_sp (same split of m)
                # c_sp (split of p)
                # c_row_sp = a_sp (same split of n)
                # c_col_sp = c_sp (same split of p)
                
                # Compute total rank
                total_rank = 0
                valid = True
                for i in range(len(a_sp)):
                    a_h = a_sp[i]
                    for j in range(len(b_col_sp)):
                        m_w = b_col_sp[j]
                        for k in range(len(c_sp)):
                            p_w = c_sp[k]
                            # Block is a_h x m_w * m_w x p_w -> a_h x p_w
                            if a_h == 2 and m_w == 2 and p_w == 2:
                                total_rank += 7
                            elif a_h == 2 and m_w == 2 and p_w == 3:
                                total_rank += 11
                            elif a_h == 2 and m_w == 2 and p_w == 4:
                                total_rank += 15
                            elif a_h == 2 and m_w == 1 and p_w == 2:
                                total_rank += 4
                            elif a_h == 2 and m_w == 1 and p_w == 3:
                                total_rank += 6
                            elif a_h == 2 and m_w == 1 and p_w == 4:
                                total_rank += 8
                            elif a_h == 1 and m_w == 2 and p_w == 2:
                                total_rank += 4
                            elif a_h == 1 and m_w == 2 and p_w == 3:
                                total_rank += 6
                            elif a_h == 1 and m_w == 2 and p_w == 4:
                                total_rank += 8
                            elif a_h == 1 and m_w == 1 and p_w == 1:
                                total_rank += 1
                            elif a_h == 1 and m_w == 1 and p_w == 2:
                                total_rank += 2
                            elif a_h == 1 and m_w == 1 and p_w == 3:
                                total_rank += 3
                            elif a_h == 1 and m_w == 1 and p_w == 4:
                                total_rank += 4
                            else:
                                total_rank += a_h * m_w * p_w
                
                if total_rank >= best_rank:
                    continue
                
                # Build the decomposition
                U = [[0] * total_rank for _ in range(n * m)]
                V = [[0] * total_rank for _ in range(m * p)]
                W = [[0] * total_rank for _ in range(p * n)]
                
                # Precompute offsets
                a_off = [0]
                for s in a_sp[:-1]:
                    a_off.append(a_off[-1] + s)
                b_col_off = [0]
                for s in b_col_sp[:-1]:
                    b_col_off.append(b_col_off[-1] + s)
                c_off = [0]
                for s in c_sp[:-1]:
                    c_off.append(c_off[-1] + s)
                
                col = 0
                for i in range(len(a_sp)):
                    a_h = a_sp[i]
                    a_start = a_off[i]
                    for j in range(len(b_col_sp)):
                        m_w = b_col_sp[j]
                        m_start = b_col_off[j]
                        for k in range(len(c_sp)):
                            p_w = c_sp[k]
                            p_start = c_off[k]
                            
                            # Block: A[a_start:a_start+a_h, m_start:m_start+m_w] * B[m_start:m_start+m_w, p_start:p_start+p_w]
                            # -> C[a_start:a_start+a_h, p_start:p_start+p_w]
                            
                            if a_h == 2 and m_w == 2 and p_w == 2:
                                for r in range(7):
                                    for ii in range(2):
                                        for jj in range(2):
                                            val = U_s[ii*2+jj][r]
                                            if val != 0:
                                                U[(a_start+ii)*m + (m_start+jj)][col] += val
                                    for jj in range(2):
                                        for kk in range(2):
                                            val = V_s[jj*2+kk][r]
                                            if val != 0:
                                                V[(m_start+jj)*p + (p_start+kk)][col] += val
                                    for kk in range(2):
                                        for ii in range(2):
                                            val = W_s[kk*2+ii][r]
                                            if val != 0:
                                                W[(p_start+kk)*n + (a_start+ii)][col] += val
                                    col += 1
                            elif a_h == 2 and m_w == 2 and p_w == 3:
                                U_d, V_d, W_d = decomp_223
                                for r in range(11):
                                    for ii in range(2):
                                        for jj in range(2):
                                            val = U_d[ii*2+jj][r]
                                            if val != 0:
                                                U[(a_start+ii)*m + (m_start+jj)][col] += val
                                    for jj in range(2):
                                        for kk in range(3):
                                            val = V_d[jj*3+kk][r]
                                            if val != 0:
                                                V[(m_start+jj)*p + (p_start+kk)][col] += val
                                    for kk in range(3):
                                        for ii in range(2):
                                            val = W_d[kk*2+ii][r]
                                            if val != 0:
                                                W[(p_start+kk)*n + (a_start+ii)][col] += val
                                    col += 1
                            elif a_h == 2 and m_w == 2 and p_w == 4:
                                U_d, V_d, W_d = decomp_224
                                for r in range(15):
                                    for ii in range(2):
                                        for jj in range(2):
                                            val = U_d[ii*2+jj][r]
                                            if val != 0:
                                                U[(a_start+ii)*m + (m_start+jj)][col] += val
                                    for jj in range(2):
                                        for kk in range(4):
                                            val = V_d[jj*4+kk][r]
                                            if val != 0:
                                                V[(m_start+jj)*p + (p_start+kk)][col] += val
                                    for kk in range(4):
                                        for ii in range(2):
                                            val = W_d[kk*2+ii][r]
                                            if val != 0:
                                                W[(p_start+kk)*n + (a_start+ii)][col] += val
                                    col += 1
                            elif a_h == 2 and m_w == 1 and p_w == 2:
                                # Outer product: 2x1 * 1x2 -> 2x2
                                for ii in range(2):
                                    for kk in range(2):
                                        U[(a_start+ii)*m + m_start][col] = 1
                                        V[m_start*p + (p_start+kk)][col] = 1
                                        W[(p_start+kk)*n + (a_start+ii)][col] = 1
                                        col += 1
                            elif a_h == 2 and m_w == 1 and p_w == 3:
                                # Outer product: 2x1 * 1x3 -> 2x3
                                for ii in range(2):
                                    for kk in range(3):
                                        U[(a_start+ii)*m + m_start][col] = 1
                                        V[m_start*p + (p_start+kk)][col] = 1
                                        W[(p_start+kk)*n + (a_start+ii)][col] = 1
                                        col += 1
                            elif a_h == 2 and m_w == 1 and p_w == 4:
                                # Outer product: 2x1 * 1x4 -> 2x4
                                for ii in range(2):
                                    for kk in range(4):
                                        U[(a_start+ii)*m + m_start][col] = 1
                                        V[m_start*p + (p_start+kk)][col] = 1
                                        W[(p_start+kk)*n + (a_start+ii)][col] = 1
                                        col += 1
                            elif a_h == 1 and m_w == 2 and p_w == 2:
                                # Outer product: 1x2 * 2x2 -> 1x2
                                for jj in range(2):
                                    for kk in range(2):
                                        U[(a_start)*m + (m_start+jj)][col] = 1
                                        V[(m_start+jj)*p + (p_start+kk)][col] = 1
                                        W[(p_start+kk)*n + a_start][col] = 1
                                        col += 1
                            elif a_h == 1 and m_w == 2 and p_w == 3:
                                # Outer product: 1x2 * 2x3 -> 1x3
                                for jj in range(2):
                                    for kk in range(3):
                                        U[(a_start)*m + (m_start+jj)][col] = 1
                                        V[(m_start+jj)*p + (p_start+kk)][col] = 1
                                        W[(p_start+kk)*n + a_start][col] = 1
                                        col += 1
                            elif a_h == 1 and m_w == 2 and p_w == 4:
                                # Outer product: 1x2 * 2x4 -> 1x4
                                for jj in range(2):
                                    for kk in range(4):
                                        U[(a_start)*m + (m_start+jj)][col] = 1
                                        V[(m_start+jj)*p + (p_start+kk)][col] = 1
                                        W[(p_start+kk)*n + a_start][col] = 1
                                        col += 1
                            elif a_h == 1 and m_w == 1 and p_w == 1:
                                U[a_start*m + m_start][col] = 1
                                V[m_start*p + p_start][col] = 1
                                W[p_start*n + a_start][col] = 1
                                col += 1
                            elif a_h == 1 and m_w == 1 and p_w == 2:
                                for kk in range(2):
                                    U[a_start*m + m_start][col] = 1
                                    V[m_start*p + (p_start+kk)][col] = 1
                                    W[(p_start+kk)*n + a_start][col] = 1
                                    col += 1
                            elif a_h == 1 and m_w == 1 and p_w == 3:
                                for kk in range(3):
                                    U[a_start*m + m_start][col] = 1
                                    V[m_start*p + (p_start+kk)][col] = 1
                                    W[(p_start+kk)*n + a_start][col] = 1
                                    col += 1
                            elif a_h == 1 and m_w == 1 and p_w == 4:
                                for kk in range(4):
                                    U[a_start*m + m_start][col] = 1
                                    V[m_start*p + (p_start+kk)][col] = 1
                                    W[(p_start+kk)*n + a_start][col] = 1
                                    col += 1
                            else:
                                # Fallback: schoolbook
                                for ii in range(a_h):
                                    for jj in range(m_w):
                                        for kk in range(p_w):
                                            U[(a_start+ii)*m + (m_start+jj)][col] = 1
                                            V[(m_start+jj)*p + (p_start+kk)][col] = 1
                                            W[(p_start+kk)*n + (a_start+ii)][col] = 1
                                            col += 1
                
                if col == total_rank:
                    result = {"U": U, "V": V, "W": W}
                    if _verify(result["U"], result["V"], result["W"], n, m, p):
                        best_rank = total_rank
                        best_result = result
    
    if best_result is not None:
        return best_result, best_rank
    
    return _schoolbook(n, m, p), n*m*p

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    
    try:
        result, rank = _block_decompose(n, m, p)
        if time.time() < deadline:
            return result
    except Exception:
        pass
    
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
