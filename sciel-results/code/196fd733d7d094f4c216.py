import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 20
    target_rank = 20
    
    # Fallback: Rank 21 Strassen
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l, val in v_coeffs.items():
                    if l == 0: v_idx = 0 * p + 2 * b
                    elif l == 1: v_idx = 0 * p + 2 * b + 1
                    elif l == 2: v_idx = 1 * p + 2 * b
                    elif l == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    # Try to find Rank 20
    # 1. Get the rank-21 decomposition
    sol_21 = get_rank_21()
    U21 = np.array(sol_21['U'], dtype=float)
    V21 = np.array(sol_21['V'], dtype=float)
    W21 = np.array(sol_21['W'], dtype=float)
    
    # 2. Construct the 21 rank-1 tensors as vectors
    # Tensor T is n*m x m*p x p*n? No, the multiplication tensor is (n*m) x (m*p) x (p*n)
    # Actually, the tensor T_{i,j,k} corresponds to A_{i,m} B_{m,j} -> C_{j,i}?
    # Let's stick to the vectorization of the 21 terms.
    # Term r is u_r \otimes v_r \otimes w_r.
    # We need to flatten these into a single vector of size 4*12*12 = 576.
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    total_dim = dim_U * dim_V * dim_W
    
    tensors = []
    for r in range(21):
        # u_r is U[:, r]
        # v_r is V[:, r]
        # w_r is W[:, r]
        # Tensor product u \otimes v \otimes w
        # We can create a 3D array and flatten it.
        # Or simply use np.einsum or kron.
        # np.kron(np.kron(u, v), w)
        u = U21[:, r]
        v = V21[:, r]
        w = W21[:, r]
        # The order of flattening must be consistent.
        # Let's use row-major (C-order) flatten of the 3D array T[u_idx, v_idx, w_idx]
        # The tensor T is defined such that T[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
        # So the vector is just the vectorization of the tensor.
        # We can compute the vector for term r as the vectorization of the rank-1 tensor u_r v_r^T w_r^T?
        # Actually, the tensor is a 3-index object.
        # Let's form the 3D array for term r: T_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
        # Then flatten it.
        T_r = np.einsum('i,j,k->ijk', u, v, w)
        vec_r = T_r.flatten()
        tensors.append(vec_r)
        
    # Stack into a matrix M of shape (total_dim, 21)
    M = np.column_stack(tensors)
    
    # 3. Find the null space vector
    # Use SVD. The smallest singular value should be close to 0.
    # U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    # The last row of Vt is the null space vector.
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    c = Vt[-1, :]
    
    # Check if the dependency is valid (singular value is small)
    if s[-1] > 1e-6:
        # If the null space is not well-defined, fallback to rank 21
        return sol_21
        
    # 4. Select a pivot index k
    # Choose the index with the largest absolute coefficient to avoid division by small numbers
    k = np.argmax(np.abs(c))
    if abs(c[k]) < 1e-9:
        return sol_21
        
    # 5. Construct the new decomposition
    # New coefficients for i != k: alpha_i = 1 - c[i] / c[k]
    # The term k is removed.
    
    new_U = []
    new_V = []
    new_W = []
    
    for i in range(21):
        if i == k:
            continue
        alpha = 1.0 - c[i] / c[k]
        # If alpha is very small, we could skip, but we want exactly 20 terms if possible.
        # If alpha is 0, the term is zero, but we still count it as a term in the list?
        # The rank is the number of non-zero terms.
        # If alpha is 0, we can effectively drop it, reducing rank to 19?
        # Unlikely, but possible. Let's keep it for now.
        new_U.append(alpha * U21[:, i])
        new_V.append(alpha * V21[:, i])
        new_W.append(alpha * W21[:, i])
        
    # Convert to lists of lists as required
    U_new = [list(col) for col in np.column_stack(new_U).T] if new_U else []
    V_new = [list(col) for col in np.column_stack(new_V).T] if new_V else []
    W_new = [list(col) for col in np.column_stack(new_W).T] if new_W else []
    
    # Verify the result
    # Reconstruct the tensor from the new decomposition
    U_arr = np.array(new_U).T if new_U else np.zeros((0, dim_U))
    V_arr = np.array(new_V).T if new_V else np.zeros((0, dim_V))
    W_arr = np.array(new_W).T if new_W else np.zeros((0, dim_W))
    
    # T_new[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    T_new = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # T_original from sol_21
    U_orig = np.array(sol_21['U']).T
    V_orig = np.array(sol_21['V']).T
    W_orig = np.array(sol_21['W']).T
    T_orig = np.einsum('ir,jr,kr->ijk', U_orig, V_orig, W_orig)
    
    # Check difference
    if np.linalg.norm(T_new - T_orig) > 1e-6:
        # Fallback
        return sol_21
        
    return {"U": U_new, "V": V_new, "W": W_new}
