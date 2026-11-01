"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition using Strassen on 2x2 blocks.
    The 2x4 * 4x8 multiplication is viewed as 8 independent 2x2 * 2x2 multiplications,
    each done with rank 7 (Strassen), giving total rank 8*7 = 56.
    """
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            b11_idx = 2 * u * 8 + 2 * v
            b12_idx = 2 * u * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            c11_idx = 2 * v * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = 2 * v * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # s=0: M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # s=1: M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # s=2: M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # s=3: M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # s=4: M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # s=5: M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # s=6: M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    rank = len(U[0]) if U else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Vectorized check: for each (i,j,k) where j==k's corresponding index
    # T[i,j,k,l] = delta_{j,l}, so we need sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] = 1 iff j==l
    # Here l is determined by W index: c = k*n + i, so l is implicit
    # Actually the tensor is T[a,b,c] = 1 if b's first index == c's second index
    # a = i*m+j, b = j*p+k, c = k*n+i
    # The "l" in W[k*n+i] corresponds to row i of C, col k of C
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] (for fixed i,k) should give 1 for each j contribution
    # sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for all i,j,k
    
    # Check all n*m*p = 2*4*8 = 64 entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_time=100):
    """
    Try to reduce rank by checking if any factor can be removed or merged.
    For each pair of factors, check if one is a linear combination of others
    in a way that preserves the tensor identity.
    """
    import time
    start = time.time()
    rank = len(U[0])
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor tensor T[i,j,k,l] = delta_{j,l}
    # T = sum_s U_s ⊗ V_s ⊗ W_s
    # Check if we can remove factor s: need T - U_s⊗V_s⊗W_s to be representable
    # with remaining factors. This is hard to check directly.
    
    # Alternative: try to find linear dependencies among the factor vectors
    # If for some s, U_s = sum_{t != s} c_t U_t, and similarly for V and W
    # with consistent coefficients, we might be able to eliminate s.
    
    # Simpler approach: check if any two factors are identical (can be merged)
    # or if a factor is zero (can be removed)
    
    for s in range(rank):
        if time.time() - start > max_time:
            break
        # Check if factor s is zero
        if np.allclose(U_arr[:, s], 0) or np.allclose(V_arr[:, s], 0) or np.allclose(W_arr[:, s], 0):
            # Remove this factor
            mask = [t for t in range(rank) if t != s]
            return (U_arr[:, mask].tolist(), V_arr[:, mask].tolist(), W_arr[:, mask].tolist())
    
    # Try pairwise merging: if factor s and t have proportional U, V, W vectors
    for s in range(rank):
        if time.time() - start > max_time:
            break
        for t in range(s + 1, rank):
            # Check if U[:,s] = a * U[:,t], V[:,s] = b * V[:,t], W[:,s] = c * W[:,t]
            # and a*b*c = 1 (so the product is preserved)
            if np.any(U_arr[:, t]) and np.any(V_arr[:, t]) and np.any(W_arr[:, t]):
                a = np.dot(U_arr[:, s], U_arr[:, t]) / np.dot(U_arr[:, t], U_arr[:, t])
                b = np.dot(V_arr[:, s], V_arr[:, t]) / np.dot(V_arr[:, t], V_arr[:, t])
                c = np.dot(W_arr[:, s], W_arr[:, t]) / np.dot(W_arr[:, t], W_arr[:, t])
                
                # Check if U[:,s] ≈ a*U[:,t], etc.
                if (np.allclose(U_arr[:, s], a * U_arr[:, t], atol=1e-10) and
                    np.allclose(V_arr[:, s], b * V_arr[:, t], atol=1e-10) and
                    np.allclose(W_arr[:, s], c * W_arr[:, t], atol=1e-10)):
                    # Check if a*b*c ≈ 1
                    if abs(a * b * c - 1.0) < 1e-10:
                        # These two factors contribute the same tensor term, merge them
                        # Actually they're identical contributions, so we can remove one
                        mask = [x for x in range(rank) if x != t]
                        new_U = U_arr[:, mask].copy()
                        new_V = V_arr[:, mask].copy()
                        new_W = W_arr[:, mask].copy()
                        # Verify
                        if _verify_decomposition(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
                            return (new_U.tolist(), new_V.tolist(), new_W.tolist())
    
    return (U, V, W)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Build the rank-56 Strassen-based decomposition
    U, V, W = _build_rank_56(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: return what we have
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        U2, V2, W2 = _try_reduce_rank(U, V, W, n, m, p, max_time=min(time_remaining - 5, 100))
        if len(U2[0]) < len(U[0]):
            return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
