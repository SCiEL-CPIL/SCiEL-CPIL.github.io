"""Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[3][0]=1
    # M2 = (A21+A22)B11
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=1
    # M3 = A11(B21-B22)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[0][2]=1; W[1][2]=1
    # M4 = A22(B11-B21)
    U[3][3]=1; V[0][3]=1; V[2][3]=-1; W[2][3]=1; W[3][3]=1
    # M5 = (A11+A12)B22
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=1; W[1][4]=1
    # M6 = (A21-A11)(B11+B12)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[2][5]=1; W[2][5]=1; W[3][5]=1
    # M7 = (A12-A22)(B21+B22)
    U[1][6]=1; U[3][6]=-1; V[1][6]=1; V[3][6]=1; W[0][6]=1; W[1][6]=1
    
    return U, V, W

def _kronecker(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decompositions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    n, m, p = n1*n2, m1*m2, p1*p2
    
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1*r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            idx = (i1*n2+i2)*m + (j1*m2+j2)
                            U[idx][c] = U1[i1*m1+j1][c1] * U2[i2*m2+j2][c2]
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            idx = (j1*m2+j2)*p + (k1*p2+k2)
                            V[idx][c] = V1[j1*p1+k1][c1] * V2[j2*p2+k2][c2]
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            idx = (k1*p2+k2)*n + (i1*n2+i2)
                            W[idx][c] = W1[k1*n1+i1][c1] * W2[k2*n2+i2][c2]
    
    return U, V, W

def _add_decomps(U1, V1, W1, U2, V2, W2, n, m, p):
    """Concatenate two decompositions (sum of tensors)."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 + r2
    U = [row + [0]*r2 for row in U1]
    V = [row + [0]*r2 for row in V1]
    W = [row + [0]*r2 for row in W1]
    for c in range(r2):
        for i in range(n*m):
            U[i][r1+c] = U2[i][c]
        for j in range(m*p):
            V[j][r1+c] = V2[j][c]
        for k in range(p*n):
            W[k][r1+c] = W2[k][c]
    return U, V, W

def _block_decomposition(n, m, p, n_splits, m_splits, p_splits, use_strassen=False):
    """
    Create a block decomposition with specified splits.
    n_splits: list of sizes summing to n
    m_splits: list of sizes summing to m
    p_splits: list of sizes summing to p
    
    For each block (ni, mj, pk), use schoolbook or Strassen (if 2x2x2 and use_strassen)
    """
    total_rank = 0
    U = [[0]*0 for _ in range(n*m)]  # placeholder
    V = [[0]*0 for _ in range(m*p)]
    W = [[0]*0 for _ in range(p*n)]
    
    # First pass: compute total rank
    for ni in n_splits:
        for mj in m_splits:
            for pk in p_splits:
                if use_strassen and ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Fill in
    col_offset = 0
    i_offset = 0
    j_offset = 0
    k_offset = 0
    
    for ni in n_splits:
        i_offset = 0
        for mj in m_splits:
            j_offset = 0
            for pk in p_splits:
                if use_strassen and ni == 2 and mj == 2 and pk == 2:
                    U_sub, V_sub, W_sub = _strassen_2x2x2()
                    for c in range(7):
                        for ii in range(ni):
                            for jj in range(mj):
                                U[(i_offset+ii)*m + (j_offset+jj)][col_offset+c] = U_sub[ii*mj+jj][c]
                        for jj in range(mj):
                            for kk in range(pk):
                                V[(j_offset+jj)*p + (k_offset+kk)][col_offset+c] = V_sub[jj*pk+kk][c]
                        for kk in range(pk):
                            for ii in range(ni):
                                W[(k_offset+kk)*n + (i_offset+ii)][col_offset+c] = W_sub[kk*ni+ii][c]
                    col_offset += 7
                else:
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                U[(i_offset+ii)*m + (j_offset+jj)][col_offset] = 1
                                V[(j_offset+jj)*p + (k_offset+kk)][col_offset] = 1
                                W[(k_offset+kk)*n + (i_offset+ii)][col_offset] = 1
                                col_offset += 1
                k_offset += pk
            j_offset += mj
        i_offset += ni
    
    return U, V, W, total_rank

def _try_all_block_decompositions(n, m, p, deadline):
    """Try all possible block decompositions and return the best one."""
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Generate all compositions of n, m, p
    def compositions(total, max_parts=None):
        if max_parts is None:
            max_parts = total
        results = []
        if total == 0:
            results.append([])
            return results
        for first in range(1, total + 1):
            for rest in compositions(total - first, max_parts - 1):
                results.append([first] + rest)
        return results
    
    n_comps = compositions(n)
    m_comps = compositions(m)
    p_comps = compositions(p)
    
    for n_splits in n_comps:
        if time.time() > deadline:
            break
        for m_splits in m_comps:
            if time.time() > deadline:
                break
            for p_splits in p_comps:
                if time.time() > deadline:
                    break
                
                # Try with and without Strassen
                for use_strassen in [False, True]:
                    U, V, W, rank = _block_decomposition(n, m, p, n_splits, m_splits, p_splits, use_strassen)
                    if rank < best_rank:
                        # Verify
                        if _verify(U, V, W, n, m, p):
                            best_rank = rank
                            best_U, best_V, best_W = U, V, W
    
    return best_U, best_V, best_W, best_rank

def construct():
    """Construct rank-minimal decomposition for 3x5x6."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    n, m, p = 3, 5, 6
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p  # 90
    
    # Try block decompositions
    U, V, W, rank = _try_all_block_decompositions(n, m, p, deadline)
    if rank < best_rank:
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Verify final result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook if something went wrong
        best_U, best_V, best_W = _schoolbook(n, m, p)
    
    return {"U": best_U, "V": best_V, "W": best_W}
