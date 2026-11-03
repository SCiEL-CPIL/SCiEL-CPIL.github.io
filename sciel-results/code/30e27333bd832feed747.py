import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Fallback: Schoolbook decomposition
    def get_schoolbook():
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    time_limit = 140.0 # Leave 10s margin for 150s limit
    
    # Target ranks to try, starting from just below schoolbook
    target_ranks = list(range(rank_schoolbook - 1, rank_schoolbook - 10, -1))
    
    # ALS Parameters
    max_iter_als = 50
    tol_als = 1e-5
    
    # Build the tensor T as a 3D array T[i, j, k] = delta_{i+k? no, T[i, j, k] is 1 if i,j,k match the product}
    # T[i, j, k] = 1 if the term (i, j, k) is in the decomposition.
    # Actually, T is the multiplication tensor. T[i, j, k] = 1 if i is row of A, j is col of A / row of B, k is col of B.
    # The tensor is defined such that C = A B implies C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has entries T_{i, j, k} = 1 if the product A_{ij} B_{jk} contributes to C_{ik}.
    # In the standard basis, T is the identity-like tensor for the contraction.
    # T[i, j, k] = 1 if the index j is the "middle" index.
    # Wait, the tensor for matrix multiplication is $T \in \mathbb{R}^{n \times m \times p}$? No.
    # The tensor is in $\mathbb{R}^{nm} \otimes \mathbb{R}^{mp} \otimes \mathbb{R}^{pn}$.
    # Let's define T as a 3D array of size (n*m, m*p, p*n).
    # T[a, b, c] = 1 if the triple (a, b, c) corresponds to a valid term.
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # So T[a, b, c] = 1 if a%m == b//p and b%p == c//n and c%n == a//m.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomp(U, V, W):
        """Check if U, V, W exactly decompose T."""
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Reconstruct T_recon
        # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # This is expensive if done naively.
        # Use einsum: np.einsum('ar,br,cr->abc', U, V, W)
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        T_recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(T_recon, T, atol=1e-9)

    def round_to_half_int(arr):
        return np.round(arr * 2) / 2.0

    def als_decompose(target_rank, max_iter, tol, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        prev_err = float('inf')
        for _ in range(max_iter):
            # Update U
            # U @ (V^T * W) ? No.
            # T = sum U[:, r] V[:, r] W[:, r]^T
            # For fixed V, W, T is linear in U.
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # We want to minimize sum_{b,c} (T[a, b, c] - sum_r U[a, r] V[b, r] W[c, r])^2
            # This is a linear least squares problem for each a.
            # Let M_a be a matrix of size (m*p * p*n) x target_rank?
            # Actually, we can reshape.
            # T is (nm, mp, pn).
            # For each a, T[a] is (mp, pn).
            # V is (mp, r), W is (pn, r).
            # Let X = V * W^T? No.
            # sum_r U[a, r] V[b, r] W[c, r] = sum_r U[a, r] (V[:, r] * W[:, r])^T flattened?
            # Let Z_r = V[:, r] * W[:, r] (element-wise product, size mp*pn).
            # Then T[a, :, :] approx sum_r U[a, r] Z_r.
            # So we solve U[a] @ Z^T = T[a] flattened.
            
            # Construct Z: (r, mp*pn)
            Z = (V * W[:, None, :]).reshape(target_rank, -1)
            T_flat = T.reshape(n*m, -1)
            
            # Solve U * Z^T = T_flat
            # U = T_flat @ pinv(Z^T)
            U_new = np.linalg.lstsq(Z.T, T_flat, rcond=None)[0].T
            U = U_new
            
            # Update V
            # Similarly, V = T_flat_v @ pinv(Z_v^T)
            # Z_v = U * W
            Z_v = (U * W[:, None, :]).reshape(target_rank, -1)
            T_flat_v = T.transpose(1, 0, 2).reshape(m*p, -1)
            V_new = np.linalg.lstsq(Z_v.T, T_flat_v, rcond=None)[0].T
            V = V_new
            
            # Update W
            Z_w = (U * V[:, None, :]).reshape(target_rank, -1)
            T_flat_w = T.transpose(2, 0, 1).reshape(p*n, -1)
            W_new = np.linalg.lstsq(Z_w.T, T_flat_w, rcond=None)[0].T
            W = W_new
            
            # Calculate error
            err = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        # Round to half-integers
        U_r = round_to_half_int(U)
        V_r = round_to_half_int(V)
        W_r = round_to_half_int(W)
        
        # Verify
        if verify_decomp(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    # Try to find a decomposition with rank < 90
    for rank in target_ranks:
        if time.time() > time_limit:
            break
            
        # Try a few random restarts for this rank
        attempts = 0
        max_attempts = 100 # Limit attempts to avoid infinite loop
        
        while attempts < max_attempts and time.time() < time_limit:
            attempts += 1
            seed = random.randint(0, 10000)
            result = als_decompose(rank, max_iter_als, tol_als, seed)
            if result is not None:
                U, V, W = result
                # Convert to list of lists of floats (or ints if integer)
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Clean up floats that are integers
                for i in range(len(U_list)):
                    for j in range(len(U_list[i])):
                        if U_list[i][j] == int(U_list[i][j]):
                            U_list[i][j] = int(U_list[i][j])
                for i in range(len(V_list)):
                    for j in range(len(V_list[i])):
                        if V_list[i][j] == int(V_list[i][j]):
                            V_list[i][j] = int(V_list[i][j])
                for i in range(len(W_list)):
                    for j in range(len(W_list[i])):
                        if W_list[i][j] == int(W_list[i][j]):
                            W_list[i][j] = int(W_list[i][j])
                            
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # If no better rank found, return schoolbook
    return get_schoolbook()
