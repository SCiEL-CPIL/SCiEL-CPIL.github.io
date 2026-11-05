import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target ranks to search for, in order of preference
    target_ranks = [18, 19, 20, 21]
    
    # Fallback to current known solution if search fails
    def get_fallback_solution():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return {"U": U, "V": V, "W": W}

    # Tensor definition
    # T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # Actually, the tensor for matrix multiplication C = AB is:
    # T_{i,j,k} = 1 if the term A_{i,l} B_{l,j} contributes to C_{i,j}
    # The decomposition is: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # Here, U is (n*m) x R, V is (m*p) x R, W is (p*n) x R?
    # Wait, the interface says:
    # U: (n*m, rank) -> indices (i, l) where i is row of A, l is col of A / row of B?
    # Let's check the seed code indices.
    # Seed U indices: 0,1,2,3,4,5. n=2, m=3. n*m=6.
    # Seed V indices: 0..7. m*p=12.
    # Seed W indices: 0..7. p*n=8.
    
    # The tensor T has dimensions (n*m, m*p, p*n).
    # T[a, b, c] is non-zero if the multiplication is valid.
    # Let's define the tensor T explicitly.
    # A is n x m. B is m x p. C is n x p.
    # U index: (i, l) where i in 0..n-1, l in 0..m-1.
    # V index: (l, j) where l in 0..m-1, j in 0..p-1.
    # W index: (j, i) where j in 0..p-1, i in 0..n-1.
    # The product term is U[i,l,r] * V[l,j,r] * W[j,i,r].
    # This must equal C[i,j] which is sum_l A[i,l] B[l,j].
    # So for a fixed r, the sum over r of U[i,l,r] V[l,j,r] W[j,i,r] must equal A[i,l]B[l,j] for all i,l,j?
    # No, the sum is over r.
    # C[i,j] = sum_l A[i,l] B[l,j].
    # The decomposition represents the bilinear map.
    # T[i, l, j, k?] No, the tensor is 3-way for the bilinear form?
    # Standard tensor for matrix multiplication (n x m) * (m x p) is of order 3?
    # Usually defined on indices (i, j, k) where i is row of A, j is col of B, k is intermediate?
    # No, the rank of matrix multiplication is the tensor rank of the 3-way tensor T with indices (i, j, k) where:
    # T[i, j, k] = 1 if i, j, k are compatible?
    # Let's look at the shapes:
    # U: (n*m, R). Rows indexed by (i, l).
    # V: (m*p, R). Rows indexed by (l, j).
    # W: (p*n, R). Rows indexed by (j, i).
    # The condition is:
    # sum_r U[i, l, r] V[l, j, r] W[j, i, r] = 1 if the term A[i,l]B[l,j] exists?
    # Actually, the matrix multiplication tensor T is defined such that:
    # <T, A \otimes B \otimes C^T> = trace(C^T (AB)) = sum_{i,j} C[j,i] A[i,l] B[l,j] (with sum over l)
    # = sum_{i,l,j} T[i,l,j] A[i,l] B[l,j] C[j,i]?
    # If we set T[i,l,j] = 1 for all valid i,l,j, then:
    # sum_{i,l,j} 1 * A[i,l] B[l,j] C[j,i] = sum_{i,j} C[j,i] (sum_l A[i,l] B[l,j]) = trace(C^T AB).
    # So the tensor T has dimensions (n*m, m*p, p*n) and T[i,l, l', j, j', i'] = 1 if l == l' and i == i' and j == j'?
    # No, the indices of the tensor factors are:
    # U: (i, l)
    # V: (l, j)
    # W: (j, i)
    # The tensor T is non-zero only when the intermediate index l matches in U and V, and the output indices match in V and W, and input in U and W?
    # Actually, the tensor T is simply the identity on the "matching" of indices.
    # T[u, v, w] = 1 if the indices u=(i,l), v=(l,j), w=(j,i) form a valid chain.
    # Specifically, u=(i,l), v=(l',j), w=(j',i').
    # T[u,v,w] = 1 if l==l', j==j', i==i'.
    # Otherwise 0.
    
    # Let's construct the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for l in range(m):
            for j in range(p):
                u_idx = i * m + l
                v_idx = l * p + j
                w_idx = j * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Search for lower rank
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    for R in target_ranks:
        if time.time() > deadline:
            break
        
        # Try to find a solution for rank R
        found = False
        attempts = 0
        max_attempts = 2000 # Adjust based on time
        
        while not found and time.time() < deadline:
            attempts += 1
            # Random initialization
            # Use small integers to help with rounding
            U = np.random.randint(-2, 3, (n*m, R))
            V = np.random.randint(-2, 3, (m*p, R))
            W = np.random.randint(-2, 3, (p*n, R))
            
            # ALS Refinement
            # We want to minimize || T - U V W^T ||_F
            # This is not a simple ALS because it's a 3-way tensor decomposition (CP decomposition).
            # Standard ALS for CP:
            # Fix V, W, update U.
            # U_{i,r} = sum_{j,k} T_{i,j,k} V_{j,r} W_{k,r} / (sum_{j,k} (V_{j,r} W_{k,r})^2)
            # This is for symmetric or specific layouts. Here the indices are coupled.
            # The term is U_{u,r} V_{v,r} W_{w,r}.
            # For fixed r, we want U_{u,r} to be the projection of the "slice" T_{u, :, :} onto the vector V[:,r] * W[:,r]?
            # Let vec(V_r) and vec(W_r) be the vectors.
            # T_u is a matrix (m*p, p*n).
            # We want U_{u,r} such that U_{u,r} < V_r, W_r>_elementwise approx T_u.
            # Actually, the inner product is sum_{v,w} T[u,v,w] V[v,r] W[w,r].
            # So U[u,r] = (sum_{v,w} T[u,v,w] V[v,r] W[w,r]) / (sum_{v,w} V[v,r]^2 W[w,r]^2)
            
            for step in range(20):
                # Update U
                # Precompute for each r: vec_V_r = V[:, r], vec_W_r = W[:, r]
                # denom_U[r] = sum(V[:,r]^2 * W[:,r]^2)
                # num_U[u, r] = sum_{v,w} T[u,v,w] V[v,r] W[w,r]
                
                # Efficient computation:
                # T is sparse.
                # We can iterate over non-zeros of T.
                # But for speed, let's use dense ops if small enough.
                # n*m=6, m*p=12, p*n=8. Total 576 entries. Very small.
                
                # V_r is (m*p, 1). W_r is (p*n, 1).
                # Outer = V_r * W_r.T -> (m*p, p*n)
                # T_u dot Outer -> scalar.
                
                # Vectorized update for all u and r:
                # This is a bit complex to vectorize fully without loops over r.
                # Given small size, loop over r is fine.
                
                new_U = np.zeros_like(U)
                new_V = np.zeros_like(V)
                new_W = np.zeros_like(W)
                
                for r in range(R):
                    v_r = V[:, r]
                    w_r = W[:, r]
                    outer_vw = np.outer(v_r, w_r) # (m*p, p*n)
                    
                    # Update U[:, r]
                    # num_U[:, r] = T @ outer_vw (sum over v,w)
                    # T is (n*m, m*p, p*n).
                    # T @ outer_vw -> (n*m,)
                    num_U_r = np.einsum('uvw,vw->u', T, outer_vw)
                    denom_U_r = np.sum(v_r**2 * w_r**2)
                    if denom_U_r > 1e-10:
                        new_U[:, r] = num_U_r / denom_U_r
                    else:
                        new_U[:, r] = 0
                        
                    # Update V[:, r]
                    # Fix U, W.
                    # V[v, r] = sum_{u,w} T[u,v,w] U[u,r] W[w,r] / sum_{u,w} (U[u,r] W[w,r])^2
                    u_r = U[:, r]
                    outer_uw = np.outer(u_r, w_r) # (n*m, p*n)
                    # We need to sum over u,w for each v.
                    # T[u,v,w] * outer_uw[u,w]
                    # Reshape T to (n*m, m*p, p*n) -> (m*p, n*m, p*n)
                    # T_v = T[:, v, :] -> (n*m, p*n)
                    # num_V_v = sum_{u,w} T[u,v,w] U[u,r] W[w,r]
                    # This is T_v dot outer_uw
                    # We can do: np.einsum('uvw,uw->v', T, outer_uw)
                    num_V_r = np.einsum('uvw,uw->v', T, outer_uw)
                    denom_V_r = np.sum(u_r**2 * w_r**2)
                    if denom_V_r > 1e-10:
                        new_V[:, r] = num_V_r / denom_V_r
                    else:
                        new_V[:, r] = 0
                        
                    # Update W[:, r]
                    # Fix U, V.
                    # W[w, r] = sum_{u,v} T[u,v,w] U[u,r] V[v,r] / sum_{u,v} (U[u,r] V[v,r])^2
                    v_r_old = V[:, r]
                    outer_uv = np.outer(u_r, v_r_old) # (n*m, m*p)
                    # num_W_w = sum_{u,v} T[u,v,w] U[u,r] V[v,r]
                    # np.einsum('uvw,uv->w', T, outer_uv)
                    num_W_r = np.einsum('uvw,uv->w', T, outer_uv)
                    denom_W_r = np.sum(u_r**2 * v_r_old**2)
                    if denom_W_r > 1e-10:
                        new_W[:, r] = num_W_r / denom_W_r
                    else:
                        new_W[:, r] = 0

                U = new_U
                V = new_V
                W = new_W
                
            # Check if converged to exact solution
            # Reconstruct T_approx
            # T_approx[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
            # Use einsum
            T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
            
            # Check error
            err = np.linalg.norm(T - T_approx)
            
            if err < 1e-9:
                # Try to round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                T_approx_round = np.einsum('ur,vr,wr->uvw', U_round, V_round, W_round)
                err_round = np.linalg.norm(T - T_approx_round)
                
                if err_round < 1e-9:
                    # Verify exactness with fractions or just float check is enough for this size?
                    # Float check is usually sufficient if values are small.
                    # But to be safe, check if all entries are half-integers.
                    if np.all(np.abs(U_round * 2 - np.round(U_round * 2)) < 1e-9) and \
                       np.all(np.abs(V_round * 2 - np.round(V_round * 2)) < 1e-9) and \
                       np.all(np.abs(W_round * 2 - np.round(W_round * 2)) < 1e-9):
                        
                        # Convert to list of lists of floats (or ints if possible)
                        # The task asks for integer/half-integer entries.
                        # Returning floats that are exactly 0.5, 1.0, etc. is fine.
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
            
            # Optional: Random perturbation if stuck?
            # For now, just restart with new random init.

    # If no better solution found, return fallback
    return get_fallback_solution()
