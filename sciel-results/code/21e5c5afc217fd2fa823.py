"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

def _build_ntt_decomposition(n, m, p):
    """
    Build a low-rank decomposition using 4-point NTT along the middle dimension.
    
    For matrix multiplication C = A * B where A is n×m and B is m×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We apply a 4-point DFT (m=4) to diagonalize the sum over j.
    The 4-point DFT has entries omega^jk where omega = exp(-2πi/4) = -i.
    
    Using the DFT basis, we can write:
    sum_j A[i,j] * B[j,k] = (1/4) * sum_t DFT_A[i,t] * DFT_B[t,k] * 4
    where DFT_A[i,t] = sum_j A[i,j] * omega^{jt}
    and DFT_B[t,k] = sum_j B[j,k] * omega^{-jt}
    
    This gives us 4 independent scalar multiplications per (i,k) pair.
    Total rank = n * p * 4 = 2 * 8 * 4 = 64... but we can do better.
    
    Actually, let's think more carefully. The tensor T[i,j,k] = delta structure means:
    We need sum_s U[i,j,s] * V[j,k,s] * W[k,i,s] = 1 for all i,j,k.
    
    Using the NTT approach:
    - For each output position (i,k), we have 4 "frequency" components
    - U[i,j,s] depends on i, j, and frequency t
    - V[j,k,s] depends on j, k, and frequency t  
    - W[k,i,s] = 1 for the appropriate combination
    
    Let's use omega = -i (4th root of unity):
    omega^0 = 1, omega^1 = -i, omega^2 = -1, omega^3 = i
    
    For the DFT: X[t] = sum_j x[j] * omega^{jt}
    Inverse: x[j] = (1/4) * sum_t X[t] * omega^{-jt}
    
    So: C[i,k] = sum_j A[i,j] * B[j,k]
    = sum_j A[i,j] * B[j,k]
    
    Let A_hat[i,t] = sum_j A[i,j] * omega^{jt}
    Let B_hat[t,k] = sum_j B[j,k] * omega^{-jt}
    
    Then C[i,k] = (1/4) * sum_t A_hat[i,t] * B_hat[t,k]
    
    This gives rank = n * p * 4 / 4 = n * p = 16? No...
    
    Actually each (i,k,t) triple gives one rank-1 term, so rank = n*p*4 = 64.
    But we can exploit that W[k,i,s] just needs to be 1 for the right s.
    
    Let me reconsider. The factorization is:
    T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k*n+i, s]
    
    Using NTT:
    U[i,j,s] = omega^{j*t} / 4 for s = (i,k,t) encoded somehow
    V[j,k,s] = omega^{-j*t} for s = (i,k,t)
    W[k*n+i, s] = 1 for s = (i,k,t)
    
    But s must be the same for U, V, W. Let's index s by (t, i, k) or similar.
    For a fixed t, we have:
    sum_j (omega^{jt}/4) * (omega^{-jt}) = sum_j 1/4 = 4/4 = 1 ✓
    
    So for each t in {0,1,2,3}, and for each (i,k) pair:
    U[i*m+j, s(i,k,t)] = omega^{jt} / 4
    V[j*p+k, s(i,k,t)] = omega^{-jt}
    W[k*n+i, s(i,k,t)] = 1
    
    Total rank = 4 * n * p = 4 * 2 * 8 = 64.
    
    Hmm, that's not better than 56. Let me think if we can do better.
    
    Actually, the key insight is that we don't need separate factors for each (i,k,t).
    We can share the t-component across all (i,k) pairs!
    
    For a fixed t:
    U[i*m+j, s_t] = omega^{jt} / 4 (depends on i, j, t)
    V[j*p+k, s_t] = omega^{-jt} (depends on j, k, t)
    W[k*n+i, s_t] = 1 (depends on k, i, t)
    
    But U depends on i and j, so we can't have a single factor for all i.
    We need separate factors for each (i, t) pair for U, but V and W can be shared.
    
    Wait, no. The factor s must be consistent. Let's use s = (i, t) for U,
    but then V and W also depend on s, so they depend on i and t.
    
    This is getting complex. Let me just implement the straightforward version
    with rank 64 and see if it verifies, then try to reduce.
    """
    # Use the 4-point DFT with omega = -i
    # omega^0 = 1, omega^1 = -i, omega^2 = -1, omega^3 = i
    
    # For exactness, we'll use the fact that omega^jt cycles with period 4.
    # omega^{jt} for j=0,1,2,3 and t=0,1,2,3:
    # t=0: 1, 1, 1, 1
    # t=1: 1, -i, -1, i
    # t=2: 1, -1, 1, -1
    # t=3: 1, i, -1, -i
    
    # Since we need integer/half-integer entries, complex numbers won't work directly.
    # We need to use real representations.
    
    # Alternative: use a real 4-point transform.
    # The 4-point DFT can be decomposed into real and imaginary parts.
    # But this doubles the rank.
    
    # Let me try a different approach: use the fact that for m=4,
    # we can write the sum as a combination of 4 simple patterns.
    
    # Pattern 0 (t=0): all ones, weight 1/4
    # Pattern 1 (t=1): [1, -i, -1, i], weight 1/4
    # Pattern 2 (t=2): [1, -1, 1, -1], weight 1/4
    # Pattern 3 (t=3): [1, i, -1, -i], weight 1/4
    
    # For real-valued matrices, we can combine t=1 and t=3 (conjugates):
    # t=1 + t=3: [2, 0, -2, 0] -> weight 1/4 each, combined weight 1/2
    # t=1 - t=3: [0, -2i, 0, 2i] -> this is imaginary, problematic
    
    # Let me just use the Strassen approach which is known to work.
    # The NTT approach with complex numbers doesn't give integer/half-integer entries.
    
    # Instead, let me try a different factorization using real transforms.
    # Use the 2x2 block structure with a real 2-point transform.
    
    # For m=4, split into two 2-element groups: j=0,1 and j=2,3
    # Use 2-point DFT on each group:
    # Group 1: A[i,0], A[i,1] -> [A[i,0]+A[i,1], A[i,0]-A[i,1]]
    # Group 2: A[i,2], A[i,3] -> [A[i,2]+A[i,3], A[i,2]-A[i,3]]
    
    # This doesn't directly help with the sum over j.
    
    # Let me fall back to the Strassen decomposition which is known to work.
    return _build_rank_56(n, m, p)

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
    """
    start = time.time()
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check if any factor is zero
    for s in range(rank):
        if time.time() - start > max_time:
            break
        if np.allclose(U_arr[:, s], 0) or np.allclose(V_arr[:, s], 0) or np.allclose(W_arr[:, s], 0):
            mask = [t for t in range(rank) if t != s]
            return (U_arr[:, mask].tolist(), V_arr[:, mask].tolist(), W_arr[:, mask].tolist())
    
    # Try pairwise merging
    for s in range(rank):
        if time.time() - start > max_time:
            break
        for t in range(s + 1, rank):
            if np.any(U_arr[:, t]) and np.any(V_arr[:, t]) and np.any(W_arr[:, t]):
                a = np.dot(U_arr[:, s], U_arr[:, t]) / np.dot(U_arr[:, t], U_arr[:, t])
                b = np.dot(V_arr[:, s], V_arr[:, t]) / np.dot(V_arr[:, t], V_arr[:, t])
                c = np.dot(W_arr[:, s], W_arr[:, t]) / np.dot(W_arr[:, t], W_arr[:, t])
                
                if (np.allclose(U_arr[:, s], a * U_arr[:, t], atol=1e-10) and
                    np.allclose(V_arr[:, s], b * V_arr[:, t], atol=1e-10) and
                    np.allclose(W_arr[:, s], c * W_arr[:, t], atol=1e-10)):
                    if abs(a * b * c - 1.0) < 1e-10:
                        mask = [x for x in range(rank) if x != t]
                        new_U = U_arr[:, mask].copy()
                        new_V = V_arr[:, mask].copy()
                        new_W = W_arr[:, mask].copy()
                        if _verify_decomposition(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
                            return (new_U.tolist(), new_V.tolist(), new_W.tolist())
    
    return (U, V, W)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Try NTT-based decomposition first
    U, V, W = _build_ntt_decomposition(n, m, p)
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        U2, V2, W2 = _try_reduce_rank(U, V, W, n, m, p, max_time=min(time_remaining - 5, 100))
        if len(U2[0]) < len(U[0]):
            return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
