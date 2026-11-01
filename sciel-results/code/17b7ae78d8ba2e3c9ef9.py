import numpy as np
import time
import random

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 5
    target_rank = 13
    time_limit = 110
    start_time = time.time()
    
    # Precompute the target tensor T
    # T[i, j, k] = 1 if C[i, k] depends on A[i, j] and B[j, k]
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So the tensor component T_{i, j, k} is 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # In the decomposition C = sum_r W_r (U_r . A) (V_r . B),
    # The condition is that for all i, k:
    # sum_r W[i, k, r] (sum_{i', j'} U[i', j', r] A[i', j']) (sum_{j', k'} V[j', k', r] B[j', k']) == C[i, k]
    
    # We can define the target bilinear form.
    # Let A be a vector of size nm, B be a vector of size mp.
    # C is a vector of size pn.
    # The tensor T is a 3rd order tensor of size (nm, mp, pn)?
    # Actually, the standard tensor for matmul is T_{i, j, k} where i is row of C, j is col of A/row of B, k is col of C.
    # T_{i, j, k} = 1 if A[i, j] and B[j, k] multiply to contribute to C[i, k].
    # Here, C[i, k] = sum_j A[i, j] B[j, k].
    # So for a fixed r, we have vectors u (size nm), v (size mp), w (size pn).
    # The contribution to C[i, k] is w[i, k, r] * (sum_{i', j'} u[i', j', r] A[i', j']) * (sum_{j', k'} v[j', k', r] B[j', k']).
    # This must equal sum_j A[i, j] B[j, k].
    
    # To check validity, we can pick random A and B and check if C matches.
    # Or, we can check the tensor identity directly.
    # The tensor identity is:
    # sum_r w_{i, k, r} u_{i', j', r} v_{j', k', r} = delta_{i, i'} delta_{k, k'} ? No.
    # The bilinear form is:
    # <T, A \otimes B> = C.
    # T is defined by T_{i, j, k} = 1 if j is the contraction index.
    # Specifically, T_{i, j, k} = 1 if we are looking at A[i, j] and B[j, k] contributing to C[i, k].
    # Wait, the tensor T has indices (i, j, k) corresponding to (row of C, col of A, col of C)?
    # No, the standard tensor for matmul $A_{n \times m} B_{m \times p} = C_{n \times p}$ is:
    # $T_{i, j, k} = 1$ if $A_{i, j}$ and $B_{j, k}$ are multiplied.
    # Here $i \in [0, n), j \in [0, m), k \in [0, p)$.
    # The decomposition is:
    # $T_{i, j, k} = \sum_{r=1}^R U_{i, j, r} V_{j, k, r} W_{i, k, r}$?
    # No, $W$ is in the output space.
    # The correct identity is:
    # $\sum_{r} W_{i, k, r} U_{i, j, r} V_{j, k, r} = T_{i, j, k}$
    # where $U_{i, j, r}$ is the coefficient of $A_{i, j}$ in the r-th linear form.
    # $V_{j, k, r}$ is the coefficient of $B_{j, k}$ in the r-th linear form.
    # $W_{i, k, r}$ is the coefficient of the r-th product in $C_{i, k}$.
    
    # So we need to find U, V, W such that:
    # For all i, j, k:
    # sum_r U[i, j, r] V[j, k, r] W[i, k, r] = (1 if j is valid else 0)
    # Actually, $T_{i, j, k} = 1$ for all $i \in [0, n), j \in [0, m), k \in [0, p)$.
    # Because every $A_{i, j}$ and $B_{j, k}$ pair contributes to $C_{i, k}$.
    # Wait, $C_{i, k} = \sum_j A_{i, j} B_{j, k}$.
    # So the tensor $T$ has $T_{i, j, k} = 1$ for all $i, j, k$?
    # No, the tensor is defined on the space $A \otimes B \to C$.
    # The component $T_{i, j, k}$ is 1 if the term $A_{i, j} B_{j, k}$ appears in $C_{i, k}$.
    # Yes, it is 1 for all $i, j, k$.
    # So we need:
    # $\sum_{r} U_{i, j, r} V_{j, k, r} W_{i, k, r} = 1$ for all $i \in [0, n), j \in [0, m), k \in [0, p)$.
    
    # Note: The indices in U, V, W in the problem description are flattened.
    # U has shape (n*m, rank). Index (i, j) is i*m + j.
    # V has shape (m*p, rank). Index (j, k) is j*p + k.
    # W has shape (p*n, rank). Index (k, i) is k*n + i (Trace convention W[k*n+i]).
    # So $W_{i, k, r}$ in my formula corresponds to $W[k \cdot n + i, r]$ in the array.
    
    # We want to minimize the error:
    # E = sum_{i, j, k} (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1)^2
    
    # We will use a randomized local search.
    
    def calculate_error(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # We need to compute sum_r U[i, r] V[j, r] W[k, r] for all i, j, k.
        # This is a tensor contraction.
        # Let's use numpy einsum.
        # U: (i, j, r) -> reshape to (n, m, R)
        # V: (j, k, r) -> reshape to (m, p, R)
        # W: (k, i, r) -> reshape to (p, n, R)
        
        U_3d = U.reshape(n, m, target_rank)
        V_3d = V.reshape(m, p, target_rank)
        W_3d = W.reshape(p, n, target_rank)
        
        # Compute T_recon[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Note: W index is k*n + i. In W_3d, it is W_3d[k, i, r].
        
        # einsum: 'ijr,jkr,ki r->ijk'
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Target is 1 everywhere.
        error = np.sum((T_recon - 1.0) ** 2)
        return error

    # Initialize
    U = np.zeros((n * m, target_rank), dtype=int)
    V = np.zeros((m * p, target_rank), dtype=int)
    W = np.zeros((p * n, target_rank), dtype=int)
    
    # Random initialization
    for r in range(target_rank):
        U[:, r] = np.random.randint(-1, 2, n * m)
        V[:, r] = np.random.randint(-1, 2, m * p)
        W[:, r] = np.random.randint(-1, 2, p * n)
        
    current_error = calculate_error(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error = current_error
    
    # Local Search
    while time.time() - start_time < time_limit:
        if best_error < 1e-6:
            break
            
        # Pick a random matrix and index
        mat_choice = np.random.choice(['U', 'V', 'W'])
        if mat_choice == 'U':
            idx = np.random.randint(0, n * m)
            r = np.random.randint(0, target_rank)
            # Perturb U[idx, r]
            old_val = U[idx, r]
            new_val = old_val + np.random.choice([-1, 1])
            U[idx, r] = new_val
            new_error = calculate_error(U, V, W)
            if new_error < current_error:
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[idx, r] = old_val
        elif mat_choice == 'V':
            idx = np.random.randint(0, m * p)
            r = np.random.randint(0, target_rank)
            old_val = V[idx, r]
            new_val = old_val + np.random.choice([-1, 1])
            V[idx, r] = new_val
            new_error = calculate_error(U, V, W)
            if new_error < current_error:
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[idx, r] = old_val
        else:
            idx = np.random.randint(0, p * n)
            r = np.random.randint(0, target_rank)
            old_val = W[idx, r]
            new_val = old_val + np.random.choice([-1, 1])
            W[idx, r] = new_val
            new_error = calculate_error(U, V, W)
            if new_error < current_error:
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[idx, r] = old_val
        
        # Occasionally restart
        if np.random.random() < 0.01:
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            for r in range(target_rank):
                U[:, r] = np.random.randint(-1, 2, n * m)
                V[:, r] = np.random.randint(-1, 2, m * p)
                W[:, r] = np.random.randint(-1, 2, p * n)
            current_error = calculate_error(U, V, W)
            if current_error < best_error:
                best_error = current_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
    if best_error < 1e-6:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to rank 18 Strassen solution
        return _construct_rank_18()

def _construct_rank_18():
    # Copy of the original logic
    n, m, p = 2, 2, 5
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_strassen(k1, k2):
        nonlocal current_rank
        idx_B_0_k1 = 0 * p + k1
        idx_B_1_k1 = 1 * p + k1
        idx_B_0_k2 = 0 * p + k2
        idx_B_1_k2 = 1 * p + k2
        idx_C_0_k1 = k1 * n + 0
        idx_C_1_k1 = k1 * n + 1
        idx_C_0_k2 = k2 * n + 0
        idx_C_1_k2 = k2 * n + 1
        idx_A_00 = 0
        idx_A_01 = 1
        idx_A_10 = 2
        idx_A_11 = 3
        
        terms = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 1, 0, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 0, 1, -1], 'w': [0, 0, 1, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 1, 0, 0], 'w': [1, 1, 0, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 0, 1, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 0, 1, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 1, 0, 1], 'w': [1, 0, 0, 0]}
        ]
        
        for t in terms:
            u = t['u']
            v = t['v']
            w = t['w']
            U[idx_A_00][current_rank] += u[0]
            U[idx_A_01][current_rank] += u[1]
            U[idx_A_10][current_rank] += u[2]
            U[idx_A_11][current_rank] += u[3]
            V[idx_B_0_k1][current_rank] += v[0]
            V[idx_B_1_k1][current_rank] += v[1]
            V[idx_B_0_k2][current_rank] += v[2]
            V[idx_B_1_k2][current_rank] += v[3]
            W[idx_C_0_k1][current_rank] += w[0]
            W[idx_C_1_k1][current_rank] += w[1]
            W[idx_C_0_k2][current_rank] += w[2]
            W[idx_C_1_k2][current_rank] += w[3]
            current_rank += 1

    def add_schoolbook(k):
        nonlocal current_rank
        idx_B_0_k = 0 * p + k
        idx_B_1_k = 1 * p + k
        idx_C_0_k = k * n + 0
        idx_C_1_k = k * n + 1
        terms = [
            { 'u': [1, 0, 0, 0], 'v_idx': 0, 'w_idx': 0, 'val': 1 },
            { 'u': [0, 1, 0, 0], 'v_idx': 1, 'w_idx': 0, 'val': 1 },
            { 'u': [0, 0, 1, 0], 'v_idx': 0, 'w_idx': 1, 'val': 1 },
            { 'u': [0, 0, 0, 1], 'v_idx': 1, 'w_idx': 1, 'val': 1 }
        ]
        for t in terms:
            u = t['u']
            U[0][current_rank] += u[0]
            U[1][current_rank] += u[1]
            U[2][current_rank] += u[2]
            U[3][current_rank] += u[3]
            if t['v_idx'] == 0:
                V[idx_B_0_k][current_rank] += t['val']
            else:
                V[idx_B_1_k][current_rank] += t['val']
            if t['w_idx'] == 0:
                W[idx_C_0_k][current_rank] += t['val']
            else:
                W[idx_C_1_k][current_rank] += t['val']
            current_rank += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_schoolbook(4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
