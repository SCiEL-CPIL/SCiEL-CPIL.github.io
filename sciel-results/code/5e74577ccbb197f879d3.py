"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill


def _build_schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    
    # Try a few random seeds to verify correctness
    for seed in [42, 123, 456, 789, 1024]:
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            # M_r[i,k] = sum_j u_mat[i,j] * v_mat[j,k]
            M_r = u_mat @ v_mat
            # C_decomp[i,k] += w_mat[k,i] * M_r[i,k]
            C_decomp += w_mat.T * M_r
        
        if not np.allclose(C_decomp, C_expected, atol=tol):
            return False
    
    return True


def _build_strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for 4x4 * 4x8.
    
    The key insight: C[i,k] = sum_j A[i,j]*B[j,k]
    Split i into (i1, i0) where i = 2*i1 + i0, i1 in {0,1}, i0 in {0,1}
    Split k into (k1, k0) where k = 2*k1 + k0, k1 in {0,1,2,3}, k0 in {0,1}
    Split j into (j1, j0) where j = 2*j1 + j0, j1 in {0,1}, j0 in {0,1}
    
    For each (i1, k1) block pair:
    C_block[i0, k0] = sum_{j1=0}^{1} sum_{j0=0}^{1} A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    
    This is a (2, 4, 2) matrix multiplication: M (2x4) * N (4x2) = C (2x2)
    where M[i0, j1*2+j0] = A[2*i1+i0, 2*j1+j0]
          N[j1*2+j0, k0] = B[2*j1+j0, 2*k1+k0]
    
    For the (2,4,2) sub-problem, use Strassen:
    M = [M1, M2] where M1 = M[:, 0:2], M2 = M[:, 2:4] (each 2x2)
    N = [N1; N2] where N1 = N[0:2, :], N2 = N[2:4, :] (each 2x2)
    C = M1*N1 + M2*N2
    
    Strassen for M1*N1: 7 rank-1 terms
    Strassen for M2*N2: 7 rank-1 terms
    Total for this sub-problem: 14 rank-1 terms
    
    There are 2 (i1) * 4 (k1) = 8 sub-problems
    Total rank: 8 * 14 = 112
    """
    rank = 112
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i1 in range(2):
        for k1 in range(4):
            # M[i0, j1*2+j0] = A[2*i1+i0, 2*j1+j0]
            # N[j1*2+j0, k0] = B[2*j1+j0, 2*k1+k0]
            # C[i0, k0] = sum_{j1,j0} M[i0, j1*2+j0] * N[j1*2+j0, k0]
            
            # M1 = M[:, 0:2] = A[2*i1+i0, 0:2] for i0 in {0,1}
            # M2 = M[:, 2:4] = A[2*i1+i0, 2:4] for i0 in {0,1}
            # N1 = N[0:2, :] = B[0:2, 2*k1+k0] for k0 in {0,1}
            # N2 = N[2:4, :] = B[2:4, 2*k1+k0] for k0 in {0,1}
            
            # Strassen for M1*N1:
            # M1 = [[m11, m12], [m21, m22]] where m11 = A[2*i1, 0], m12 = A[2*i1, 1], m21 = A[2*i1+1, 0], m22 = A[2*i1+1, 1]
            # N1 = [[n11, n12], [n21, n22]] where n11 = B[0, 2*k1], n12 = B[0, 2*k1+1], n21 = B[1, 2*k1], n22 = B[1, 2*k1+1]
            
            m11_idx = (2*i1) * m + 0
            m12_idx = (2*i1) * m + 1
            m21_idx = (2*i1+1) * m + 0
            m22_idx = (2*i1+1) * m + 1
            
            n11_idx = 0 * p + (2*k1)
            n12_idx = 0 * p + (2*k1+1)
            n21_idx = 1 * p + (2*k1)
            n22_idx = 1 * p + (2*k1+1)
            
            # Strassen products for M1*N1:
            # P1 = (m11+m22)*(n11+n22)
            # P2 = (m21+m22)*n11
            # P3 = m11*(n12-n22)
            # P4 = m22*(n21-n11)
            # P5 = (m11+m12)*n22
            # P6 = (m21-m11)*(n11+n12)
            # P7 = (m12-m22)*(n21+n22)
            
            strassen_terms_M1N1 = [
                # (u_indices, v_indices, w_coefficients)
                # P1 = (m11+m22)*(n11+n22)
                ([m11_idx, m22_idx], [n11_idx, n22_idx], [(0,0,1), (0,1,1), (1,0,1), (1,1,-1)]),
                # P2 = (m21+m22)*n11
                ([m21_idx, m22_idx], [n11_idx], [(0,0,0), (0,1,0), (1,0,-1), (1,1,0)]),
                # P3 = m11*(n12-n22)
                ([m11_idx], [n12_idx, n22_idx], [(0,0,0), (0,1,1), (1,0,0), (1,1,0)]),
                # P4 = m22*(n21-n11)
                ([m22_idx], [n21_idx, n11_idx], [(0,0,0), (0,1,0), (1,0,1), (1,1,-1)]),
                # P5 = (m11+m12)*n22
                ([m11_idx, m12_idx], [n22_idx], [(0,0,-1), (0,1,0), (1,0,0), (1,1,0)]),
                # P6 = (m21-m11)*(n11+n12)
                ([m21_idx, m11_idx], [n11_idx, n12_idx], [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]),
                # P7 = (m12-m22)*(n21+n22)
                ([m12_idx, m22_idx], [n21_idx, n22_idx], [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]),
            ]
            
            # Strassen for M2*N2:
            # M2 = [[M2_11, M2_12], [M2_21, M2_22]] where M2_11 = A[2*i1, 2], M2_12 = A[2*i1, 3], M2_21 = A[2*i1+1, 2], M2_22 = A[2*i1+1, 3]
            # N2 = [[N2_11, N2_12], [N2_21, N2_22]] where N2_11 = B[2, 2*k1], N2_12 = B[2, 2*k1+1], N2_21 = B[3, 2*k1], N2_22 = B[3, 2*k1+1]
            
            M2_11_idx = (2*i1) * m + 2
            M2_12_idx = (2*i1) * m + 3
            M2_21_idx = (2*i1+1) * m + 2
            M2_22_idx = (2*i1+1) * m + 3
            
            N2_11_idx = 2 * p + (2*k1)
            N2_12_idx = 2 * p + (2*k1+1)
            N2_21_idx = 3 * p + (2*k1)
            N2_22_idx = 3 * p + (2*k1+1)
            
            strassen_terms_M2N2 = [
                # Q1 = (M2_11+M2_22)*(N2_11+N2_22)
                ([M2_11_idx, M2_22_idx], [N2_11_idx, N2_22_idx], [(0,0,1), (0,1,1), (1,0,1), (1,1,-1)]),
                # Q2 = (M2_21+M2_22)*N2_11
                ([M2_21_idx, M2_22_idx], [N2_11_idx], [(0,0,0), (0,1,0), (1,0,-1), (1,1,0)]),
                # Q3 = M2_11*(N2_12-N2_22)
                ([M2_11_idx], [N2_12_idx, N2_22_idx], [(0,0,0), (0,1,1), (1,0,0), (1,1,0)]),
                # Q4 = M2_22*(N2_21-N2_11)
                ([M2_22_idx], [N2_21_idx, N2_11_idx], [(0,0,0), (0,1,0), (1,0,1), (1,1,-1)]),
                # Q5 = (M2_11+M2_12)*N2_22
                ([M2_11_idx, M2_12_idx], [N2_22_idx], [(0,0,-1), (0,1,0), (1,0,0), (1,1,0)]),
                # Q6 = (M2_21-M2_11)*(N2_11+N2_12)
                ([M2_21_idx, M2_11_idx], [N2_11_idx, N2_12_idx], [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]),
                # Q7 = (M2_12-M2_22)*(N2_21+N2_22)
                ([M2_12_idx, M2_22_idx], [N2_21_idx, N2_22_idx], [(0,0,0), (0,1,0), (1,0,0), (1,1,1)]),
            ]
            
            all_terms = strassen_terms_M1N1 + strassen_terms_M2N2
            
            for u_indices, v_indices, w_coeffs in all_terms:
                # u_indices: list of indices into U (n*m dimension)
                # v_indices: list of indices into V (m*p dimension)
                # w_coeffs: list of (i0, k0, coeff) for W (p*n dimension)
                
                for ui in u_indices:
                    U[ui, col] = 1.0
                for vi in v_indices:
                    V[vi, col] = 1.0
                for i0, k0, coeff in w_coeffs:
                    # W index: k*n + i where k = 2*k1+k0, i = 2*i1+i0
                    k_global = 2*k1 + k0
                    i_global = 2*i1 + i0
                    w_idx = k_global * n + i_global
                    W[w_idx, col] += coeff
                
                col += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # First try the Strassen-based decomposition (rank 112)
    U, V, W = _build_strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if Strassen fails
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
