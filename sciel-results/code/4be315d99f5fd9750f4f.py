import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W (lists) form a valid decomposition."""
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_approx = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_approx, T, atol=1e-9)
    
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    def greedy_sparse_recovery(T_transformed, target_rank, max_terms=50):
        """
        Greedy algorithm to find a sparse rank-1 decomposition of a 3D tensor.
        T_transformed has shape (a, b, c).
        Returns (U, V, W) arrays with shape (a, rank), (b, rank), (c, rank)
        such that sum_r U[:,r] outer V[:,r] outer W[:,r] approximates T_transformed.
        """
        a, b, c = T_transformed.shape
        U = np.zeros((a, target_rank))
        V = np.zeros((b, target_rank))
        W = np.zeros((c, target_rank))
        
        residual = T_transformed.copy()
        
        for r in range(target_rank):
            if time.time() > deadline:
                break
            
            # Find the best rank-1 approximation of the residual
            # We use a greedy approach: try random vectors and pick the best
            best_score = float('-inf')
            best_u, best_v, best_w = None, None, None
            best_coef = 0.0
            
            n_tries = 200
            for _ in range(n_tries):
                if time.time() > deadline:
                    break
                
                # Random unit vectors
                u_rand = np.random.randn(a)
                v_rand = np.random.randn(b)
                w_rand = np.random.randn(c)
                
                # Normalize
                u_norm = np.linalg.norm(u_rand)
                v_norm = np.linalg.norm(v_rand)
                w_norm = np.linalg.norm(w_rand)
                if u_norm < 1e-10 or v_norm < 1e-10 or w_norm < 1e-10:
                    continue
                u_rand = u_rand / u_norm
                v_rand = v_rand / v_norm
                w_rand = w_rand / w_norm
                
                # Compute coefficient: <residual, u outer v outer w> / ||u outer v outer w||^2
                # <residual, u⊗v⊗w> = einsum('abc,a,b,c', residual, u_rand, v_rand, w_rand)
                numerator = np.einsum('abc,a,b,c', residual, u_rand, v_rand, w_rand)
                # ||u⊗v⊗w||^2 = ||u||^2 * ||v||^2 * ||w||^2 = 1 (since normalized)
                coef = numerator
                
                # Score: reduction in residual norm
                # ||residual - coef * u⊗v⊗w||^2 = ||residual||^2 - 2*coef*<residual, u⊗v⊗w> + coef^2 * ||u⊗v⊗w||^2
                # = ||residual||^2 - coef^2 (since <residual, u⊗v⊗w> = coef and ||u⊗v⊗w||^2 = 1)
                # So score = coef^2
                score = coef * coef
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u_rand, v_rand, w_rand
                    best_coef = coef
            
            if best_u is None:
                break
            
            # Add this rank-1 term
            U[:, r] = best_coef * best_u
            V[:, r] = best_v
            W[:, r] = best_w
            
            # Update residual
            residual -= best_coef * np.einsum('a,b,c->abc', best_u, best_v, best_w)
            
            # Check if residual is small enough
            if np.linalg.norm(residual) < 1e-10:
                break
        
        return U, V, W
    
    def try_greedy_with_transform(target_rank, n_transforms=20):
        """
        Try greedy sparse recovery with random basis changes to A and B spaces.
        """
        if time.time() > deadline:
            return None
        
        best_result = None
        best_rank = target_rank + 1
        
        for t in range(n_transforms):
            if time.time() > deadline:
                break
            
            # Create random invertible transformations for A (n*m) and B (m*p) spaces
            A_dim = n * m
            B_dim = m * p
            C_dim = p * n
            
            # Random invertible matrices
            A_transform = np.random.randn(A_dim, A_dim)
            B_transform = np.random.randn(B_dim, B_dim)
            
            # Ensure they're invertible
            if np.linalg.det(A_transform) < 1e-10 or np.linalg.det(B_transform) < 1e-10:
                continue
            
            # Transform the tensor: T'[a',b',c] = sum_{a,b} T[a,b,c] * A_transform[a',a] * B_transform[b',b]
            # This is equivalent to: T' = A_transform^T @ T @ B_transform (with appropriate reshaping)
            # More precisely: T'[a',b',c] = sum_{a,b} T[a,b,c] * A_transform[a',a] * B_transform[b',b]
            
            # Reshape T to (A_dim, B_dim, C_dim)
            T_reshaped = T.reshape(A_dim, B_dim, C_dim)
            
            # Apply transformations
            # T'[a',b',c] = sum_{a,b} T[a,b,c] * A_transform[a',a] * B_transform[b',b]
            # = einsum('abc,a'a,b'b->a'b'c', T, A_transform, B_transform)
            T_transformed = np.einsum('abc,ap,bq->pqc', T_reshaped, A_transform, B_transform)
            
            # Run greedy sparse recovery
            U_t, V_t, W_t = greedy_sparse_recovery(T_transformed, target_rank)
            
            # Check if this gives a valid decomposition
            # The decomposition in transformed basis: T' = sum_r U_t[:,r] outer V_t[:,r] outer W_t[:,r]
            # In original basis: T = sum_r (A_transform^T @ U_t[:,r]) outer (B_transform^T @ V_t[:,r]) outer W_t[:,r]
            # Wait, let me think about this more carefully.
            
            # T'[a',b',c] = sum_{a,b} T[a,b,c] * A_transform[a',a] * B_transform[b',b]
            # If T' = sum_r u_r[a'] * v_r[b'] * w_r[c], then:
            # sum_r u_r[a'] * v_r[b'] * w_r[c] = sum_{a,b} T[a,b,c] * A_transform[a',a] * B_transform[b',b]
            # 
            # Let's define: U_orig[a,r] such that u_r[a'] = sum_a U_orig[a,r] * A_transform[a',a]
            # So U_orig = A_transform^T @ U_t (where U_t has shape (A_dim, rank))
            # Similarly V_orig = B_transform^T @ V_t
            # W_orig = W_t (unchanged)
            
            U_orig = A_transform.T @ U_t
            V_orig = B_transform.T @ V_t
            W_orig = W_t
            
            # Verify
            U_check = [[float(x) for x in row] for row in U_orig]
            V_check = [[float(x) for x in row] for row in V_orig]
            W_check = [[float(x) for x in row] for row in W_orig]
            
            if verify(U_check, V_check, W_check):
                actual_rank = len(U_check[0])
                if actual_rank < best_rank:
                    best_rank = actual_rank
                    best_result = (U_check, V_check, W_check)
                
                if actual_rank <= target_rank:
                    break
        
        return best_result
    
    # First, try the hardcoded rank-28 solution
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Verify the rank-28 solution
    if verify(U, V, W):
        # Try greedy sparse recovery with random transforms to find rank 27 or lower
        result = try_greedy_with_transform(27, n_transforms=15)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
        
        # Try rank 26
        result = try_greedy_with_transform(26, n_transforms=10)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
        
        # Return rank 28
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook rank 30
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
