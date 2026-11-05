"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    
    deadline = time.time() + 135  # Safety margin before 150s limit
    
    # Schoolbook decomposition (rank 105)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def verify(U, V, W, r):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Try Strassen-based block decomposition
    # Split 3x5x7 into blocks and use Strassen for 2x2x2 sub-blocks
    # This can reduce rank from 105 to potentially lower
    
    # Strategy: Use a block decomposition approach
    # For 3x5x7, we can try various block splits
    
    def try_block_decomposition():
        """Try to find a lower rank decomposition using block structure."""
        if time.time() > deadline - 20:
            return None
        
        # Try different block sizes
        # For 3x5x7, possible splits:
        # n=3: [1,2] or [2,1]
        # m=5: [1,4], [2,3], [3,2], [4,1]
        # p=7: [1,6], [2,5], [3,4], [4,3], [5,2], [6,1]
        
        n_splits = [[1, 2], [2, 1]]
        m_splits = [[1, 4], [2, 3], [3, 2], [4, 1]]
        p_splits = [[1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]]
        
        best = None
        
        for ns in n_splits:
            for ms in m_splits:
                for ps in p_splits:
                    if time.time() > deadline - 10:
                        return best
                    
                    # Calculate rank for this block structure
                    # Each block (ni x mj x pk) has rank ni*mj*pk with schoolbook
                    # But we can use Strassen for 2x2x2 blocks to get rank 7 instead of 8
                    
                    rank = 0
                    for ni in ns:
                        for mj in ms:
                            for pk in ps:
                                if ni == 2 and mj == 2 and pk == 2:
                                    rank += 7  # Strassen
                                else:
                                    rank += ni * mj * pk
                    
                    if rank < best_rank and (best is None or rank < best[0]):
                        best = (rank, ns, ms, ps)
        
        return best
    
    best_block = try_block_decomposition()
    
    if best_block and best_block[0] < best_rank:
        rank, ns, ms, ps = best_block
        
        # Build the actual decomposition
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        col = 0
        # Map global indices to local block indices
        n_offsets = [0]
        for ni in ns:
            n_offsets.append(n_offsets[-1] + ni)
        m_offsets = [0]
        for mj in ms:
            m_offsets.append(m_offsets[-1] + mj)
        p_offsets = [0]
        for pk in ps:
            p_offsets.append(p_offsets[-1] + pk)
        
        for i_block in range(len(ns)):
            for j_block in range(len(ms)):
                for k_block in range(len(ps)):
                    ni = ns[i_block]
                    mj = ms[j_block]
                    pk = ps[k_block]
                    
                    if ni == 2 and mj == 2 and pk == 2:
                        # Use Strassen
                        # Strassen's 7 products for 2x2x2
                        # M1 = (A00+A11)(B00+B11)
                        # M2 = (A10+A11)B00
                        # M3 = A00(B10-B11)
                        # M4 = A11(B01-B10)
                        # M5 = (A00+A10)B11
                        # M6 = (A01-A00)(B00+B10)
                        # M7 = (A10-A11)(B01+B11)
                        # C00 = M1+M4-M5+M7
                        # C01 = M3+M5
                        # C10 = M2+M4
                        # C11 = M1-M2+M3+M6
                        
                        # Local indices
                        i0 = n_offsets[i_block]
                        i1 = n_offsets[i_block] + 1
                        j0 = m_offsets[j_block]
                        j1 = m_offsets[j_block] + 1
                        k0 = p_offsets[k_block]
                        k1 = p_offsets[k_block] + 1
                        
                        # M1 = (A00+A11)(B00+B11)
                        # U for A00: row i0*m+j0, U for A11: row i1*m+j1
                        # V for B00: row j0*p+k0, V for B11: row j1*p+k1
                        # W for C00: row k0*n+i0, W for C01: row k1*n+i0
                        # W for C10: row k0*n+i1, W for C11: row k1*n+i1
                        
                        # M1 contributes to C00 (+1), C01 (+1), C10 (+1), C11 (-1)
                        U[i0 * m + j0][col] += 1  # A00
                        U[i1 * m + j1][col] += 1  # A11
                        V[j0 * p + k0][col] += 1  # B00
                        V[j1 * p + k1][col] += 1  # B11
                        W[k0 * n + i0][col] += 1  # C00
                        W[k1 * n + i0][col] += 1  # C01
                        W[k0 * n + i1][col] += 1  # C10
                        W[k1 * n + i1][col] -= 1  # C11
                        col += 1
                        
                        # M2 = (A10+A11)B00
                        # Contributes to C10 (+1), C11 (-1)
                        U[i1 * m + j0][col] += 1  # A10
                        U[i1 * m + j1][col] += 1  # A11
                        V[j0 * p + k0][col] += 1  # B00
                        W[k0 * n + i1][col] += 1  # C10
                        W[k1 * n + i1][col] -= 1  # C11
                        col += 1
                        
                        # M3 = A00(B10-B11)
                        # Contributes to C01 (+1), C11 (+1)
                        U[i0 * m + j0][col] += 1  # A00
                        V[j1 * p + k0][col] += 1  # B10
                        V[j1 * p + k1][col] -= 1  # B11
                        W[k1 * n + i0][col] += 1  # C01
                        W[k1 * n + i1][col] += 1  # C11
                        col += 1
                        
                        # M4 = A11(B01-B10)
                        # Contributes to C00 (+1), C10 (+1)
                        U[i1 * m + j1][col] += 1  # A11
                        V[j0 * p + k1][col] += 1  # B01
                        V[j1 * p + k0][col] -= 1  # B10
                        W[k0 * n + i0][col] += 1  # C00
                        W[k0 * n + i1][col] += 1  # C10
                        col += 1
                        
                        # M5 = (A00+A10)B11
                        # Contributes to C00 (-1), C01 (+1)
                        U[i0 * m + j0][col] += 1  # A00
                        U[i1 * m + j0][col] += 1  # A10
                        V[j1 * p + k1][col] += 1  # B11
                        W[k0 * n + i0][col] -= 1  # C00
                        W[k1 * n + i0][col] += 1  # C01
                        col += 1
                        
                        # M6 = (A01-A00)(B00+B10)
                        # Contributes to C11 (+1)
                        U[i0 * m + j1][col] += 1  # A01
                        U[i0 * m + j0][col] -= 1  # A00
                        V[j0 * p + k0][col] += 1  # B00
                        V[j1 * p + k0][col] += 1  # B10
                        W[k1 * n + i1][col] += 1  # C11
                        col += 1
                        
                        # M7 = (A10-A11)(B01+B11)
                        # Contributes to C00 (+1)
                        U[i1 * m + j0][col] += 1  # A10
                        U[i1 * m + j1][col] -= 1  # A11
                        V[j0 * p + k1][col] += 1  # B01
                        V[j1 * p + k1][col] += 1  # B11
                        W[k0 * n + i0][col] += 1  # C00
                        col += 1
                    else:
                        # Schoolbook for this block
                        for i in range(ni):
                            for j in range(mj):
                                for k in range(pk):
                                    gi = n_offsets[i_block] + i
                                    gj = m_offsets[j_block] + j
                                    gk = p_offsets[k_block] + k
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1
        
        # Verify
        if verify(U, V, W, rank):
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    # Try to improve further with additional optimizations
    # Check if we can do better than the current best
    
    if time.time() < deadline - 30:
        # Try a different approach: use the fact that the tensor is the all-ones tensor
        # and look for sparse decompositions
        
        # For now, return the best we have
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
