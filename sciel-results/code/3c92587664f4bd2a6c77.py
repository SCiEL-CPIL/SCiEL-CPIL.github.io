"""Rewritten program for matrix multiplication 4x4 * 4x9 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Schoolbook rank
    rank_schoolbook = n * m * p

    # We'll try to find a lower rank decomposition using a greedy/iterative approach.
    # The tensor T[i,j,k] corresponds to the matrix multiplication tensor.
    # For C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # In the trace convention: W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Build the full tensor as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # Reshape to matrix form for CP decomposition
    # Mode-0 unfolding: shape (n, m*p)
    # T_mat0[i, j*p+k] = T[i,j,k]
    T_mat0 = T.reshape(n, m * p)
    T_mat1 = T.transpose(1, 0, 2).reshape(m, n * p)
    T_mat2 = T.transpose(2, 0, 1).reshape(p, n * m)

    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*p, m*p, n*m) for rank of multiplication tensor
    # Actually the lower bound is more nuanced. Let's try a range of ranks.
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None

    # Try CP decomposition with various ranks
    # We'll use an iterative approach: initialize U, V, W and refine
    
    # For this small problem, let's try ranks from a reasonable lower bound
    # The bilinear complexity of 4x4 * 4x9...
    # Known results: 4x4x4 has rank 23 (Strassen-like), but 4x4x9 is different.
    # Let's try a range of ranks.
    
    # First, let's set up the problem more carefully.
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for this multiplication)
    # Wait, that's not right. For matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k. Yes, it IS the all-ones tensor.
    
    # So we need to decompose the all-ones tensor of shape (n,m,p) into rank-r CP factors.
    # T[i,j,k] = sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is equivalent to finding matrices U (nm x r), V (mp x r), W (pn x r)
    # such that the tensor formed by U⊗V⊗W (with appropriate index mapping) equals T.
    
    # Let's use a different approach: since T is all ones, we can try to find
    # a structured decomposition.
    
    # One approach: use the fact that 1 = sum of products of carefully chosen vectors.
    # For the all-ones tensor, we can use:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's try a greedy approach with random initializations and ALS (Alternating Least Squares)
    
    def cp_decompose(T, n, m, p, rank, max_iter=500, tol=1e-10, seed=42):
        """CP decomposition of all-ones tensor using ALS."""
        rng = np.random.RandomState(seed)
        
        # Initialize factors
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        # Scale to have unit norm columns initially
        for r in range(rank):
            nU = np.linalg.norm(U[:, r])
            nV = np.linalg.norm(V[:, r])
            nW = np.linalg.norm(W[:, r])
            if nU > 0 and nV > 0 and nW > 0:
                scale = (nU * nV * nW) ** (1.0/3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: for each r, U[:,r] = (T_mat0 * (V[r,:] ⊗ W[r,:])) / (||V[r,:]||^2 * ||W[r,:]||^2)
            # T_mat0[i, j*p+k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is tricky because the indices are mixed. Let me think about this differently.
            
            # Actually, let me reformulate. The tensor T has entries T[i,j,k] = 1.
            # We want: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # For fixed r, define:
            # u_r[i,j] = U[i*m+j, r]  (shape n x m)
            # v_r[j,k] = V[j*p+k, r]  (shape m x p)
            # w_r[k,i] = W[k*n+i, r]  (shape p x n)
            
            # Then: T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
            
            # This is a 3-way tensor with a cyclic index structure.
            # To do ALS, we need to solve for each factor given the others.
            
            # For updating U (all u_r): 
            # For fixed r, we want to minimize sum_{i,j,k} (T[i,j,k] - sum_{r'} u_{r'}[i,j] * v_{r'}[j,k] * w_{r'}[k,i])^2
            # with respect to u_r.
            # The optimal u_r[i,j] = sum_k T[i,j,k] * v_r[j,k] * w_r[k,i] / sum_k (v_r[j,k] * w_r[k,i])^2
            # But since T[i,j,k] = 1:
            # u_r[i,j] = sum_k v_r[j,k] * w_r[k,i] / sum_k (v_r[j,k] * w_r[k,i])^2
            
            # Wait, that's not quite right for ALS. Let me be more careful.
            
            # The residual for entry (i,j,k) is: R[i,j,k] = T[i,j,k] - sum_{r'} u_{r'}[i,j] * v_{r'}[j,k] * w_{r'}[k,i]
            # For updating u_r[i,j], we minimize sum_k (R[i,j,k])^2 over u_r[i,j].
            # d/d u_r[i,j] sum_k (T[i,j,k] - u_r[i,j]*v_r[j,k]*w_r[k,i] - sum_{r'≠r} ...)^2 = 0
            # Let S[i,j,k] = sum_{r'≠r} u_{r'}[i,j] * v_{r'}[j,k] * w_{r'}[k,i]
            # Then: sum_k (T[i,j,k] - S[i,j,k] - u_r[i,j]*v_r[j,k]*w_r[k,i]) * (-v_r[j,k]*w_r[k,i]) = 0
            # sum_k (T[i,j,k] - S[i,j,k]) * v_r[j,k]*w_r[k,i] = u_r[i,j] * sum_k (v_r[j,k]*w_r[k,i])^2
            # u_r[i,j] = sum_k (T[i,j,k] - S[i,j,k]) * v_r[j,k]*w_r[k,i] / sum_k (v_r[j,k]*w_r[k,i])^2
            
            # This is getting complex. Let me just use a simpler approach for this small problem.
            
            # Since the tensor is all ones, let me try a direct construction approach.
            # 
            # Key insight: If we can find vectors a_i (length m), b_j (length p), c_k (length n) 
            # such that the cyclic products sum to 1...
            #
            # Actually, for the all-ones tensor with cyclic indices, one simple decomposition is:
            # Use rank = n*m*p (schoolbook) where each term picks out one (i,j,k).
            #
            # For lower rank, we need overlapping contributions.
            #
            # Let me try a different strategy: since this is a small problem, let me use 
            # numerical optimization (scipy) to find a low-rank decomposition.
            
            return None  # Placeholder - will use scipy instead
        
        return None
    
    # Use scipy.optimize to find a low-rank CP decomposition
    from scipy.optimize import minimize
    
    def try_rank(rank, max_time=30.0):
        """Try to find a rank-r decomposition using optimization."""
        if time.time() > deadline - 10:
            return None
        
        n, m, p = DIMENSIONS
        total_params = rank * (n*m + m*p + p*n)
        
        # Parameter vector: flatten U, V, W
        # U: n*m x rank, V: m*p x rank, W: p*n x rank
        x0 = np.random.randn(total_params) * 0.1
        
        def objective(x):
            U = x[:rank*n*m].reshape(n*m, rank)
            V = x[rank*n*m:rank*(n*m+m*p)].reshape(m*p, rank)
            W = x[rank*(n*m+m*p):].reshape(p*n, rank)
            
            # Compute the reconstructed tensor
            # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Compare to T (all ones)
            
            # Efficient computation:
            # For each r: outer product of u_r (n*m), v_r (m*p), w_r (p*n)
            # with appropriate reshaping
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        error += (val - 1.0) ** 2
            return error
        
        # This is too slow for optimization. Let me use a vectorized approach.
        
        def objective_vec(x):
            U = x[:rank*n*m].reshape(n*m, rank)
            V = x[rank*n*m:rank*(n*m+m*p)].reshape(m*p, rank)
            W = x[rank*(n*m+m*p):].reshape(p*n, rank)
            
            # Build the tensor efficiently
            # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            # T_recon[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
            # This is a contraction over r with cyclic indices.
            
            # Use einsum: T_recon[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            T_recon = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
            
            # Compare to all-ones
            diff = T_recon - 1.0
            return np.sum(diff ** 2)
        
        # Try a few random initializations
        best_result = None
        n_attempts = min(5, max(1, int((deadline - time.time() - 10) / 5.0)))
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break
            
            x0 = np.random.randn(total_params) * 0.1
            # Scale up a bit
            x0 *= (n*m*p / rank) ** (1.0/3.0)
            
            result = minimize(objective_vec, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < 1e-6:  # Good enough
                U = result.x[:rank*n*m].reshape(n*m, rank)
                V = result.x[rank*n*m:rank*(n*m+m*p)].reshape(m*p, rank)
                W = result.x[rank*(n*m+m*p):].reshape(p*n, rank)
                
                # Verify
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                T_recon = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
                if np.max(np.abs(T_recon - 1.0)) < 1e-6:
                    return (U, V, W)
            
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        
        return None
    
    # Try ranks from lower to upper
    # Lower bound: at least max(n, m, p) for any non-trivial decomposition
    # For all-ones tensor, the rank is at least max(n, m, p) but could be higher
    # Let's try from a reasonable range
    
    # Actually, for the all-ones tensor T[i,j,k]=1 with the cyclic constraint,
    # the minimum rank is known to be related to the bilinear complexity.
    # For 4x4 * 4x9, let's try ranks from 20 to schoolbook.
    
    for rank in range(15, rank_schoolbook + 1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(rank, max_time=20.0)
        if result is not None:
            U, V, W = result
            # Convert to list of lists with rounded values
            U_list = [[round(U[i, r], 10) for r in range(rank)] for i in range(n*m)]
            V_list = [[round(V[i, r], 10) for r in range(rank)] for i in range(m*p)]
            W_list = [[round(W[i, r], 10) for r in range(rank)] for i in range(p*n)]
            
            # Verify the decomposition with the rounded values
            U_3d = np.array(U_list).reshape(n, m, rank)
            V_3d = np.array(V_list).reshape(m, p, rank)
            W_3d = np.array(W_list).reshape(p, n, rank)
            T_recon = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d)
            if np.max(np.abs(T_recon - 1.0)) < 1e-6:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
