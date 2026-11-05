"""Matrix multiplication 3x5x5 - rank minimization via factor elimination."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 5, 5]
N, M, P = 3, 5, 5


def _schoolbook():
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W):
    """Check U @ diag(1) @ V^T reconstruction matches tensor.
    Tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    Equivalently: for each c, contribution is outer(U_c, V_c) reshaped...
    Actually: T[i,j,k] = sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c]
    """
    r = U.shape[1]
    T = np.zeros((N, M, P))
    for c in range(r):
        uc = U[:, c]  # (N*M,)
        vc = V[:, c]  # (M*P,)
        wc = W[:, c]  # (P*N,)
        # reshape
        u_mat = uc.reshape(N, M)
        v_mat = vc.reshape(M, P)
        w_mat = wc.reshape(P, N)
        # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    # Compare with true tensor T[i,j,k] = delta (the matmul tensor)
    # The matmul tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the middle)
    true_T = np.ones((N, M, P))
    return np.allclose(T, true_T, atol=1e-9)


def _verify_fast(U, V, W):
    """Faster verification using einsum."""
    r = U.shape[1]
    # T[i,j,k] = sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c]
    # Reshape U to (N,M,r), V to (M,P,r), W to (P,N,r)
    u = U.reshape(N, M, r)
    v = V.reshape(M, P, r)
    w = W.reshape(P, N, r)
    T = np.einsum('ijc,jkc,kic->ijk', u, v, w)
    true_T = np.ones((N, M, P))
    return np.allclose(T, true_T, atol=1e-9)


def _try_eliminate(U, V, W, col_to_remove):
    """
    Try to remove column `col_to_remove` by solving for adjustments to other columns.
    
    We want: sum_c U_c * V_c * W_c = 0 for the removed column's contribution,
    i.e., we need to find deltas in the remaining columns that cancel it out.
    
    This is a nonlinear problem (bilinear in U,V and W). But we can linearize:
    Fix two of the three matrices and solve for the third.
    
    Strategy: Fix V and W, solve for U adjustments. Or fix U and V, solve for W.
    Actually, the constraint is:
      sum_c' U_c' * V_c' * W_c' = - U_removed * V_removed * W_removed  (tensor equality)
    
    If we fix V and W for all c' != removed, and only adjust U, then:
      sum_c' U_c' * (V_c' * W_c') = - U_removed * (V_removed * W_removed)
    where (V_c * W_c) is a tensor. This is linear in U_c'!
    
    Similarly, fix U and W, solve for V. Or fix U and V, solve for W.
    
    We try all three linearizations and pick the one that works.
    """
    r = U.shape[1]
    if r <= 1:
        return None
    
    # The tensor to cancel
    u_rm = U[:, col_to_remove].reshape(N, M)
    v_rm = V[:, col_to_remove].reshape(M, P)
    w_rm = W[:, col_to_remove].reshape(P, N)
    
    # Target tensor: -u_rm[i,j] * v_rm[j,k] * w_rm[k,i]
    # We need sum over remaining c of U_c * V_c * W_c = target
    
    # Option 1: Fix V and W, solve for U
    # For each (i,j): sum_c' U[i*M+j, c'] * (sum_k V[j*P+k,c'] * W[k*N+i,c']) = target[i,j]
    # Let's define for each c': f_c'(i,j) = sum_k V[j*P+k, c'] * W[k*N+i, c']
    # Then: sum_c' U[i*M+j, c'] * f_c'(i,j) = target[i,j]
    # This is a linear system: for each (i,j), sum over c' of U[i*M+j,c'] * f_c'(i,j) = target[i,j]
    # Number of equations: N*M, number of unknowns: (r-1)*N*M
    # This is underdetermined (more unknowns than equations), so it should be solvable.
    
    remaining = [c for c in range(r) if c != col_to_remove]
    r_rem = len(remaining)
    
    # Build the linear system for U
    # Unknowns: U[i*M+j, c'] for all i,j and c' in remaining -> (r-1)*N*M unknowns
    # Equations: for each (i,j): sum_{c'} U[i*M+j, c'] * f_c'(i,j) = target[i,j]
    # That's N*M equations.
    
    # Actually, let me think differently. The tensor equation is:
    # sum_c' U_c' * V_c' * W_c' = target_tensor
    # If we fix V_c' and W_c' for all c', and treat U as unknown:
    # For each (i,j,k): sum_c' U[i*M+j, c'] * V[j*P+k, c'] * W[k*N+i, c'] = target[i,j,k]
    # Number of equations: N*M*P = 75
    # Number of unknowns: (r-1)*N*M = (r-1)*15
    # For r=75: 74*15 = 1110 unknowns, 75 equations. Very underdetermined.
    
    # Let's build the matrix A where A[(i,j,k), (i',j',c')] = delta(i,i')*delta(j,j') * V[j'*P+k, c'] * W[k*N+i'*N+j'... 
    # Hmm, this is getting complex. Let me use a different approach.
    
    # Simpler: use the fact that if we fix V and W, the equation for U is:
    # For each (i,j): the row U[i*M+j, :] (length r) dotted with a vector g_{ij} = (sum_k V[j*P+k,c]*W[k*N+i,c] for each c)
    # equals target[i,j]
    
    # Let me compute g_{ij,c} for c in remaining
    # g[i,j,c] = sum_k V[j*P+k, c] * W[k*N+i, c]
    
    # Then for each (i,j): sum_{c in remaining} U[i*M+j, c] * g[i,j,c] = target[i,j]
    # Plus we want U[i*M+j, col_to_remove] = 0
    
    # This is N*M equations with (r-1)*N*M unknowns. Underdetermined.
    # We can find a particular solution using least squares or null space.
    
    # Let's set up the full linear system including the constraint that the removed column is zero.
    # Actually, we just need ANY solution for the remaining columns.
    
    # Build A: shape (N*M, (r-1)*N*M)
    # A[(i,j), (i2,j2,c_idx)] = 1 if i==i2 and j==j2 else 0, times g[i,j,remaining[c_idx]]
    # Wait, that's not right. Let me re-derive.
    
    # Equation for (i,j): sum_{c in remaining} U[i*M+j, c] * g[i,j,c] = target[i,j]
    # Unknowns are U[i2*M+j2, c] for all i2,j2,c in remaining.
    # The coefficient of U[i2*M+j2, c] in equation (i,j) is:
    #   1 if i2==i and j2==j, else 0, and if so, the coefficient is g[i,j,c]
    
    # So A is block diagonal in (i,j) blocks! Each block is 1 x (r-1).
    # Block (i,j): [g[i,j,remaining[0]], g[i,j,remaining[1]], ..., g[i,j,remaining[r-2]]]
    
    # This means each equation only involves the U values for that specific (i,j).
    # So for each (i,j), we have 1 equation with (r-1) unknowns.
    # We can set (r-2) of them to 0 and solve for the last one (if g is nonzero).
    
    # Let's do this: for each (i,j), pick one column c*(i,j) from remaining and solve for U[i*M+j, c*(i,j)],
    # setting all other U[i*M+j, c'] = 0 for c' != c*(i,j).
    
    # But wait - we also need the removed column to be 0, which it already is since we're only adjusting remaining.
    
    # Let me just use least squares on the full system.
    
    # Actually, the simplest approach: since it's underdetermined, use numpy lstsq.
    
    # Build A as (N*M, r-1) where A[i*M+j, idx] = g[i,j, remaining[idx]]
    # And b[i*M+j] = target[i,j]
    # Then solve A @ x = b for x of length (r-1), where x[idx] = U[i*M+j, remaining[idx]]
    # But this only works if we're solving for ONE (i,j) at a time... 
    
    # I think I'm overcomplicating. Let me use a different strategy.
    
    # ALTERNATIVE APPROACH: 
    # The tensor equation sum_c U_c V_c W_c = T can be seen as:
    # For fixed V and W, it's linear in U.
    # For fixed U and W, it's linear in V.
    # For fixed U and V, it's linear in W.
    
    # So: fix V and W (all columns except removed), and solve for U (all columns except removed).
    # The system is: for each (i,j,k): sum_{c in rem} U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] = T[i,j,k]
    
    # Let me vectorize this. Unknowns: U[:, rem] which is (N*M, r-1).
    # Let me flatten: unknown vector x of length N*M*(r-1), where x[(i*M+j)*(r-1) + idx] = U[i*M+j, remaining[idx]]
    
    # Equation for (i,j,k): sum_{idx} x[(i*M+j)*(r-1)+idx] * V[j*P+k, remaining[idx]] * W[k*N+i, remaining[idx]] = T[i,j,k]
    
    # This is a linear system A x = b with N*M*P equations and N*M*(r-1) unknowns.
    
    # For r=75: 75 equations, 15*74=1110 unknowns. Very underdetermined.
    
    # Let's build A and use lstsq.
    
    r_total = r
    rem_idx = np.array(remaining)
    
    # A has shape (N*M*P, N*M*(r-1))
    # This could be large. For r=75: 75 x 1110. That's fine.
    
    A = np.zeros((N * M * P, N * M * (r - 1)))
    b = np.ones(N * M * P)  # target tensor is all ones
    
    for i in range(N):
        for j in range(M):
            for k in range(P):
                eq_idx = i * M * P + j * P + k
                for idx, c in enumerate(rem_idx):
                    var_idx = (i * M + j) * (r - 1) + idx
                    A[eq_idx, var_idx] = V[j * P + k, c] * W[k * N + i, c]
    
    # Solve A x = b using least squares
    try:
        x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    
    # Check if solution is valid
    x_reshaped = x.reshape(N * M, r - 1)
    
    # Build new U
    U_new = np.zeros_like(U)
    for idx, c in enumerate(rem_idx):
        U_new[:, c] = x_reshaped[:, idx]
    # Column col_to_remove stays 0
    
    # Verify
    if _verify_fast(U_new, V, W):
        return U_new, V, W
    
    # Try fixing U and W, solve for V
    # Equation for (i,j,k): sum_{c in rem} V[j*P+k, c] * U[i*M+j, c] * W[k*N+i, c] = T[i,j,k]
    # Unknowns: V[:, rem], shape (M*P, r-1), flattened to M*P*(r-1)
    
    A2 = np.zeros((N * M * P, M * P * (r - 1)))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                eq_idx = i * M * P + j * P + k
                for idx, c in enumerate(rem_idx):
                    var_idx = (j * P + k) * (r - 1) + idx
                    A2[eq_idx, var_idx] = U[i * M + j, c] * W[k * N + i, c]
    
    try:
        x2, _, _, _ = np.linalg.lstsq(A2, b, rcond=None)
    except Exception:
        pass
    else:
        x2_reshaped = x2.reshape(M * P, r - 1)
        V_new = np.zeros_like(V)
        for idx, c in enumerate(rem_idx):
            V_new[:, c] = x2_reshaped[:, idx]
        if _verify_fast(U, V_new, W):
            return U, V_new, W
    
    # Try fixing U and V, solve for W
    # Equation for (i,j,k): sum_{c in rem} W[k*N+i, c] * U[i*M+j, c] * V[j*P+k, c] = T[i,j,k]
    # Unknowns: W[:, rem], shape (P*N, r-1), flattened to P*N*(r-1)
    
    A3 = np.zeros((N * M * P, P * N * (r - 1)))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                eq_idx = i * M * P + j * P + k
                for idx, c in enumerate(rem_idx):
                    var_idx = (k * N + i) * (r - 1) + idx
                    A3[eq_idx, var_idx] = U[i * M + j, c] * V[j * P + k, c]
    
    try:
        x3, _, _, _ = np.linalg.lstsq(A3, b, rcond=None)
    except Exception:
        pass
    else:
        x3_reshaped = x3.reshape(P * N, r - 1)
        W_new = np.zeros_like(W)
        for idx, c in enumerate(rem_idx):
            W_new[:, c] = x3_reshaped[:, idx]
        if _verify_fast(U, V, W_new):
            return U, V, W_new
    
    return None


def construct():
    deadline = time.time() + 120.0
    
    U, V, W = _schoolbook()
    
    # Iteratively try to eliminate columns
    improved = True
    while improved:
        improved = False
        r = U.shape[1]
        
        for col in range(r):
            if time.time() > deadline - 30.0:
                break
            
            result = _try_eliminate(U, V, W, col)
            if result is not None:
                U, V, W = result
                improved = True
                break  # Restart with new rank
        
        if time.time() > deadline - 30.0:
            break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
