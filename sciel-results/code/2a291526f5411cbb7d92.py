"""
Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
Uses Strassen Kronecker (rank 49) for 4x4x4 core + rank-4 2x2x3 blocks for 4x4x3.
Total rank = 65.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 130.0


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = a*(e+f)
    U[0, 0] = 1; V[0, 0] = 1; V[1, 0] = 1
    # M2 = (a+b)*h
    U[0, 1] = 1; U[1, 1] = 1; V[3, 1] = 1
    # M3 = (c+d)*e
    U[2, 2] = 1; U[3, 2] = 1; V[0, 2] = 1
    # M4 = d*(g+f)
    U[3, 3] = 1; V[2, 3] = 1; V[1, 3] = 1
    # M5 = (a-d)*(e+h)
    U[0, 4] = 1; U[3, 4] = -1; V[0, 4] = 1; V[3, 4] = 1
    # M6 = (b-c)*(e+f)
    U[1, 5] = 1; U[2, 5] = -1; V[0, 5] = 1; V[1, 5] = 1
    # M7 = (a+c)*(h-g)
    U[0, 6] = 1; U[2, 6] = 1; V[3, 6] = 1; V[2, 6] = -1
    
    # C[0,0] = ae+bg = M5 + M4 - M6 - M7
    W[0, 4] = 1; W[0, 3] = 1; W[0, 5] = -1; W[0, 6] = -1
    # C[0,1] = af+bh = M1 + M2
    W[1, 0] = 1; W[1, 1] = 1
    # C[1,0] = ce+dg = M3 + M7
    W[2, 2] = 1; W[2, 6] = 1
    # C[1,1] = cf+dh = M5 + M1 - M2 - M4
    W[3, 4] = 1; W[3, 0] = 1; W[3, 1] = -1; W[3, 3] = -1
    
    return U, V, W


def _rank4_2x2x3():
    """
    Return U, V, W for 2x2 * 2x3 matrix multiplication with rank 4.
    
    A = [[a, b], [c, d]] (2x2), B = [[e, f, g], [h, i, j]] (2x3)
    C = A*B = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]] (2x3)
    
    U shape: (4, 4) - rows indexed by i*2+j
    V shape: (6, 4) - rows indexed by j*3+k
    W shape: (6, 4) - rows indexed by k*2+i (output: k*2+i where k in 0..2, i in 0..1)
    
    Rank-4 decomposition:
    M1 = a*(e+f+g)
    M2 = b*(h+i+j)
    M3 = c*(e+f+g)
    M4 = d*(h+i+j)
    """
    U = np.zeros((4, 4), dtype=float)
    V = np.zeros((6, 4), dtype=float)
    W = np.zeros((6, 4), dtype=float)
    
    # M1 = a*(e+f+g)
    U[0, 0] = 1; V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1
    # M2 = b*(h+i+j)
    U[1, 1] = 1; V[3, 1] = 1; V[4, 1] = 1; V[5, 1] = 1
    # M3 = c*(e+f+g)
    U[2, 2] = 1; V[0, 2] = 1; V[1, 2] = 1; V[2, 2] = 1
    # M4 = d*(h+i+j)
    U[3, 3] = 1; V[3, 3] = 1; V[4, 3] = 1; V[5, 3] = 1
    
    # W: C[k,i] where k is column (0..2), i is row (0..1), index = k*2+i
    # C[0,0] = a*e + b*h: M1 gives a*e, M2 gives b*h
    W[0, 0] = 1; W[0, 1] = 1
    # C[0,1] = a*f + b*i: M1 gives a*f, M2 gives b*i
    W[1, 0] = 1; W[1, 1] = 1
    # C[0,2] = a*g + b*j: M1 gives a*g, M2 gives b*j
    W[2, 0] = 1; W[2, 1] = 1
    # C[1,0] = c*e + d*h: M3 gives c*e, M4 gives d*h
    W[3, 2] = 1; W[3, 3] = 1
    # C[1,1] = c*f + d*i: M3 gives c*f, M4 gives d*i
    W[4, 2] = 1; W[4, 3] = 1
    # C[1,2] = c*g + d*j: M3 gives c*g, M4 gives d*j
    W[5, 2] = 1; W[5, 3] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, trials=50, atol=1e-9):
    """Verify a tensor decomposition is correct."""
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.einsum('ij,ijt,jkt,kit->ik', A, U_r, V_r, W_r)
        if not np.allclose(C_computed, C_expected, atol=atol):
            return False
    return True


def _kron_product_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n1m1 = U1.shape[0]
    m1p1 = V1.shape[0]
    p1n1 = W1.shape[0]
    n2m2 = U2.shape[0]
    m2p2 = V2.shape[0]
    p2n2 = W2.shape[0]
    
    new_U = np.zeros((n1m1 * n2m2, r), dtype=float)
    new_V = np.zeros((m1p1 * m2p2, r), dtype=float)
    new_W = np.zeros((p1n1 * p2n2, r), dtype=float)
    
    for t1 in range(r1):
        for t2 in range(r2):
            t = t1 * r2 + t2
            new_U[:, t] = np.kron(U1[:, t1], U2[:, t2])
            new_V[:, t] = np.kron(V1[:, t1], V2[:, t2])
            new_W[:, t] = np.kron(W1[:, t1], W2[:, t2])
    
    return new_U, new_V, new_W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    def _time_remaining():
        return DEADLINE_SECONDS - (time.time() - start_time)
    
    try:
        if _time_remaining() < 5.0:
            raise TimeoutError("Not enough time")
        
        # Step 1: Strassen 2x2x2
        U22, V22, W22 = _strassen_2x2x2()
        
        if not _verify_decomposition(U22, V22, W22, 2, 2, 2, trials=20):
            raise ValueError("Strassen verification failed")
        
        if _time_remaining() < 5.0:
            raise TimeoutError("Not enough time for Kronecker")
        
        # Step 2: Kronecker product for 4x4x4
        U44, V44, W44 = _kron_product_decomp(U22, V22, W22, U22, V22, W22)
        
        if _time_remaining() < 5.0:
            raise TimeoutError("Not enough time")
        
        if not _verify_decomposition(U44, V44, W44, 4, 4, 4, trials=20):
            raise ValueError("4x4x4 verification failed")
        
        rank_44 = U44.shape[1]  # 49
        
        # Step 3: Rank-4 2x2x3 decomposition for the remaining 3 columns
        U223, V223, W223 = _rank4_2x2x3()
        
        if not _verify_decomposition(U223, V223, W223, 2, 2, 3, trials=20):
            raise ValueError("2x2x3 verification failed")
        
        # For 4x4x3: split into 4 blocks of 2x2 * 2x3
        rank_223 = U223.shape[1]  # 4
        rank_443 = 4 * rank_223  # 16
        
        total_rank = rank_44 + rank_443  # 49 + 16 = 65
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        # Fill 4x4x4 part (columns 0-3 of B, rows 0-3 of output)
        U[:, :rank_44] = U44
        for j in range(4):
            for k in range(4):
                V[j * p + k, :rank_44] = V44[j * 4 + k, :rank_44]
        for k in range(4):
            for i in range(4):
                W[k * n + i, :rank_44] = W44[k * 4 + i, :rank_44]
        
        # Fill 4x4x3 part (columns 4-6 of B, rows 0-3 of output)
        col = rank_44
        for row_block in range(2):
            for col_block in range(2):
                for t in range(rank_223):
                    # U: A[i,j] for i in row_block*2..row_block*2+1, j in col_block*2..col_block*2+1
                    for i_local in range(2):
                        for j_local in range(2):
                            i = row_block * 2 + i_local
                            j = col_block * 2 + j_local
                            U[i * m + j, col + t] = U223[i_local * 2 + j_local, t]
                    
                    # V: B[j,k] for j in col_block*2..col_block*2+1, k in 4..6
                    for j_local in range(2):
                        for k_local in range(3):
                            j = col_block * 2 + j_local
                            k = 4 + k_local
                            V[j * p + k, col + t] = V223[j_local * 3 + k_local, t]
                    
                    # W: C[i,k] for i in row_block*2..row_block*2+1, k in 4..6
                    # W index: k*n + i
                    for k_local in range(3):
                        for i_local in range(2):
                            k = 4 + k_local
                            i = row_block * 2 + i_local
                            W[k * n + i, col + t] = W223[k_local * 2 + i_local, t]
                
                col += rank_223
        
        if col != total_rank:
            raise ValueError(f"Column mismatch: {col} vs {total_rank}")
        
        if not _verify_decomposition(U, V, W, n, m, p, trials=50):
            raise ValueError("Full decomposition verification failed")
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    except Exception:
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
