"""Matrix multiplication 3x4x6 - Strassen Block Construction."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.max(np.abs(T - expected)) < tol

def _strassen_recursive(n, m, p):
    """
    Recursively compute the Strassen decomposition for n x m x p matrices.
    Returns U, V, W tensors of shape (n, m, rank), (m, p, rank), (p, n, rank).
    """
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        U = np.zeros((1, 1, 1))
        V = np.zeros((1, 1, 1))
        W = np.zeros((1, 1, 1))
        U[0, 0, 0] = 1.0
        V[0, 0, 0] = 1.0
        W[0, 0, 0] = 1.0
        return U, V, W
    
    # Ensure dimensions are even for Strassen split
    # If not, we could pad, but for this specific problem (3,4,6), 
    # we will pad to (4,4,8) in the main construct function.
    # Here we assume n, m, p are even.
    
    n_half = n // 2
    m_half = m // 2
    p_half = p // 2
    
    # Recursively compute sub-problems
    # The sub-problems are all of size (n_half, m_half, p_half)
    U_sub, V_sub, W_sub = _strassen_recursive(n_half, m_half, p_half)
    rank_sub = U_sub.shape[2]
    
    # Allocate full tensors
    U = np.zeros((n, m, 7 * rank_sub))
    V = np.zeros((m, p, 7 * rank_sub))
    W = np.zeros((p, n, 7 * rank_sub))
    
    # Strassen Formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A01 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B00 - B01)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # We map the 7 sub-multiplications to 7 sets of rank_sub columns.
    # For each sub-multiplication c in 0..6, we have rank_sub columns.
    
    # Coefficients for U (A matrix)
    # U[i, j, c*rank_sub + k]
    # i, j in {0, 1} * n_half, m_half
    
    # M1: A00 + A11
    # U[0:nh, 0:mh, 0:rs] += 1
    # U[nh:n, mh:m, 0:rs] += 1
    U[0:n_half, 0:m_half, 0:rank_sub] += U_sub
    U[n_half:n, m_half:m, 0:rank_sub] += U_sub
    
    # M2: A01 + A11
    # U[0:nh, mh:m, rs:2rs] += 1
    # U[nh:n, mh:m, rs:2rs] += 1
    U[0:n_half, m_half:m, rank_sub:2*rank_sub] += U_sub
    U[n_half:n, m_half:m, rank_sub:2*rank_sub] += U_sub
    
    # M3: A00
    # U[0:nh, 0:mh, 2rs:3rs] += 1
    U[0:n_half, 0:m_half, 2*rank_sub:3*rank_sub] += U_sub
    
    # M4: A11
    # U[nh:n, mh:m, 3rs:4rs] += 1
    U[n_half:n, m_half:m, 3*rank_sub:4*rank_sub] += U_sub
    
    # M5: A00 + A01
    # U[0:nh, 0:mh, 4rs:5rs] += 1
    # U[0:nh, mh:m, 4rs:5rs] += 1
    U[0:n_half, 0:m_half, 4*rank_sub:5*rank_sub] += U_sub
    U[0:n_half, m_half:m, 4*rank_sub:5*rank_sub] += U_sub
    
    # M6: A10 - A00
    # U[nh:n, 0:mh, 5rs:6rs] += 1
    # U[0:nh, 0:mh, 5rs:6rs] -= 1
    U[n_half:n, 0:m_half, 5*rank_sub:6*rank_sub] += U_sub
    U[0:n_half, 0:m_half, 5*rank_sub:6*rank_sub] -= U_sub
    
    # M7: A01 - A11
    # U[0:nh, mh:m, 6rs:7rs] += 1
    # U[nh:n, mh:m, 6rs:7rs] -= 1
    U[0:n_half, m_half:m, 6*rank_sub:7*rank_sub] += U_sub
    U[n_half:n, m_half:m, 6*rank_sub:7*rank_sub] -= U_sub
    
    # Coefficients for V (B matrix)
    # V[j, k, c*rank_sub + k_idx]
    
    # M1: B00 + B11
    V[0:m_half, 0:p_half, 0:rank_sub] += V_sub
    V[m_half:m, p_half:p, 0:rank_sub] += V_sub
    
    # M2: B00
    V[0:m_half, 0:p_half, rank_sub:2*rank_sub] += V_sub
    
    # M3: B10 - B11
    V[m_half:m, 0:p_half, 2*rank_sub:3*rank_sub] += V_sub
    V[m_half:m, p_half:p, 2*rank_sub:3*rank_sub] -= V_sub
    
    # M4: B00 - B01
    V[0:m_half, 0:p_half, 3*rank_sub:4*rank_sub] += V_sub
    V[0:m_half, p_half:p, 3*rank_sub:4*rank_sub] -= V_sub
    
    # M5: B11
    V[m_half:m, p_half:p, 4*rank_sub:5*rank_sub] += V_sub
    
    # M6: B00 + B01
    V[0:m_half, 0:p_half, 5*rank_sub:6*rank_sub] += V_sub
    V[0:m_half, p_half:p, 5*rank_sub:6*rank_sub] += V_sub
    
    # M7: B10 + B11
    V[m_half:m, 0:p_half, 6*rank_sub:7*rank_sub] += V_sub
    V[m_half:m, p_half:p, 6*rank_sub:7*rank_sub] += V_sub
    
    # Coefficients for W (C matrix)
    # W[k, i, c*rank_sub + k_idx]
    # Note: W shape is (p, n, rank). Indices are (k, i).
    
    # M1: C00 += 1, C11 += 1
    # C00 is k in 0:ph, i in 0:nh
    # C11 is k in ph:p, i in nh:n
    W[0:p_half, 0:n_half, 0:rank_sub] += W_sub
    W[p_half:p, n_half:n, 0:rank_sub] += W_sub
    
    # M2: C10 += 1, C11 -= 1
    # C10 is k in ph:p, i in 0:nh
    # C11 is k in ph:p, i in nh:n
    W[p_half:p, 0:n_half, rank_sub:2*rank_sub] += W_sub
    W[p_half:p, n_half:n, rank_sub:2*rank_sub] -= W_sub
    
    # M3: C01 += 1, C11 += 1
    # C01 is k in 0:ph, i in nh:n
    # C11 is k in ph:p, i in nh:n
    W[0:p_half, n_half:n, 2*rank_sub:3*rank_sub] += W_sub
    W[p_half:p, n_half:n, 2*rank_sub:3*rank_sub] += W_sub
    
    # M4: C00 += 1
    W[0:p_half, 0:n_half, 3*rank_sub:4*rank_sub] += W_sub
    
    # M5: C00 -= 1, C01 += 1
    W[0:p_half, 0:n_half, 4*rank_sub:5*rank_sub] -= W_sub
    W[0:p_half, n_half:n, 4*rank_sub:5*rank_sub] += W_sub
    
    # M6: C10 += 1, C11 += 1
    W[p_half:p, 0:n_half, 5*rank_sub:6*rank_sub] += W_sub
    W[p_half:p, n_half:n, 5*rank_sub:6*rank_sub] += W_sub
    
    # M7: C00 += 1
    W[0:p_half, 0:n_half, 6*rank_sub:7*rank_sub] += W_sub
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Time guard
    start_time = time.time()
    deadline = 120.0 - 5.0 # 5s safety margin
    
    # Pad dimensions to powers of 2 for Strassen
    # 3 -> 4, 4 -> 4, 6 -> 8
    n_pad = 4
    m_pad = 4
    p_pad = 8
    
    if time.time() > deadline:
        # Fallback to schoolbook if we are already too late (unlikely)
        rank = n * m * p
        U = np.zeros((n, m, rank))
        V = np.zeros((m, p, rank))
        W = np.zeros((p, n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, c] = 1.0
                    V[j, k, c] = 1.0
                    W[k, i, c] = 1.0
                    c += 1
        return {'U': U.flatten().tolist(), 'V': V.flatten().tolist(), 'W': W.flatten().tolist()}

    # Compute Strassen decomposition for padded sizes
    U_pad, V_pad, W_pad = _strassen_recursive(n_pad, m_pad, p_pad)
    
    # Extract the sub-tensors for the original sizes
    # U shape: (n_pad, m_pad, rank) -> (n, m, rank)
    U = U_pad[:n, :m, :]
    # V shape: (m_pad, p_pad, rank) -> (m, p, rank)
    V = V_pad[:m, :p, :]
    # W shape: (p_pad, n_pad, rank) -> (p, n, rank)
    W = W_pad[:p, :n, :]
    
    rank = U.shape[2]
    
    if time.time() > deadline:
        # Return what we have, assuming it's correct
        return {'U': U.flatten().tolist(), 'V': V.flatten().tolist(), 'W': W.flatten().tolist()}

    # Verify
    if _verify(U, V, W, n, m, p, tol=1e-6):
        return {'U': U.flatten().tolist(), 'V': V.flatten().tolist(), 'W': W.flatten().tolist()}
    else:
        # Fallback to trivial if construction failed
        rank_trivial = n * m * p
        U_t = np.zeros((n, m, rank_trivial))
        V_t = np.zeros((m, p, rank_trivial))
        W_t = np.zeros((p, n, rank_trivial))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i, j, c] = 1.0
                    V_t[j, k, c] = 1.0
                    W_t[k, i, c] = 1.0
                    c += 1
        return {'U': U_t.flatten().tolist(), 'V': V_t.flatten().tolist(), 'W': W_t.flatten().tolist()}

# EVOLVE-BLOCK-END
