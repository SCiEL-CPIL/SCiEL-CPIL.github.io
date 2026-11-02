"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    if U.shape != (n * m, r):
        return False
    if V.shape != (m * p, r):
        return False
    if W.shape != (p * n, r):
        return False
    
    max_err = 0.0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0.0
                for rr in range(r):
                    val += U[i * m + k, rr] * V[k * p + j, rr] * W[j * n + i, rr]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    # Also check that non-matmul entries are zero
    for i in range(n):
        for j in range(p):
            for k in range(m):
                for rr in range(r):
                    pass  # The above loop already checks all matmul entries
    
    # Check that the decomposition is zero for non-matmul entries
    # Actually, the matmul tensor has T[i*m+k, k*p+j, j*n+i] = 1 for all i,j,k
    # and T[a,b,c] = 0 otherwise
    # We need to verify all 4*8*8 = 256 entries
    
    max_err = 0.0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if this is a matmul entry
                i = a // m
                k = a % m
                expected_j = b % p
                expected_i = c % n
                expected_k = b // p
                
                if (k == expected_k and i == expected_i and j == expected_j if False else True):
                    # This check is complex, let me use a simpler approach
                    pass
    
    # Simpler verification: compute the full tensor and compare
    computed = np.zeros((n * m, m * p, p * n), dtype=float)
    for rr in range(r):
        u_col = U[:, rr]
        v_col = V[:, rr]
        w_col = W[:, rr]
        computed += np.einsum('a,b,c->abc', u_col, v_col, w_col)
    
    # Build expected tensor
    expected = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                expected[i * m + k, k * p + j, j * n + i] = 1.0
    
    max_err = np.max(np.abs(computed - expected))
    return max_err < tol


def _search_rank13(deadline):
    """
    Search for a rank-13 decomposition by trying to find a linear dependency
    among the rank-14 terms that allows rank reduction.
    """
    n, m, p = 2, 2, 4
    rank14 = _build_rank14()
    U14 = np.array(rank14['U'], dtype=float)
    V14 = np.array(rank14['V'], dtype=float)
    W14 = np.array(rank14['W'], dtype=float)
    
    # For rank-13, we need to find a linear combination of columns that is zero
    # in the tensor sense: sum_t alpha_t * (U[:,t] ⊗ V[:,t] ⊗ W[:,t]) = 0
    # This is a nonlinear constraint.
    
    # Alternative approach: try all subsets of 13 terms from the 14
    # and check if they can be adjusted (with different coefficients) to span the tensor
    
    # This is still a nonlinear problem. Let me try a simpler heuristic:
    # Check if any single term is redundant (can be removed without loss)
    
    for remove_idx in range(14):
        if time.time() > deadline:
            break
        
        # Try removing term remove_idx
        U_rem = np.delete(U14, remove_idx, axis=1)
        V_rem = np.delete(V14, remove_idx, axis=1)
        W_rem = np.delete(W14, remove_idx, axis=1)
        
        # Check if the remaining 13 terms still span the tensor
        # This requires solving for whether the tensor is in the span of the 13 rank-1 tensors
        # This is a bilinear problem, which is hard
        
        # Quick check: compute the residual
        computed = np.zeros((n * m, m * p, p * n), dtype=float)
        for rr in range(13):
            u_col = U_rem[:, rr]
            v_col = V_rem[:, rr]
            w_col = W_rem[:, rr]
            computed += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        
        expected = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    expected[i * m + k, k * p + j, j * n + i] = 1.0
        
        max_err = np.max(np.abs(computed - expected))
        if max_err < 1e-10:
            # Found a rank-13 solution!
            return {"U": U_rem.tolist(), "V": V_rem.tolist(), "W": W_rem.tolist()}
    
    # Try a more sophisticated approach: allow rescaling of terms
    # For each pair of terms to remove, try to find coefficients that compensate
    
    # Actually, the known result is that the rank is at least 14 for 2x2x4,
    # so rank-13 may not exist. Let me verify this is the case.
    
    # For now, return None
    return None


def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    # First, try to find a rank-13 decomposition
    rank13 = _search_rank13(_DEADLINE)
    if rank13 is not None and _verify_decomposition(rank13['U'], rank13['V'], rank13['W']):
        return rank13
    
    # Fallback to verified rank-14 Strassen-based decomposition
    return _build_rank14()

# EVOLVE-BLOCK-END
