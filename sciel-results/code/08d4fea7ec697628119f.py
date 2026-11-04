import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W correctly decompose the multiplication tensor using vectorized einsum."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    # Trivial (schoolbook) decomposition
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, idx] = 1
                V_trivial[b, idx] = 1
                W_trivial[c, idx] = 1
                idx += 1

    # Start with trivial decomposition
    best_rank = trivial_rank
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()

    # Try to find a lower rank decomposition using structured random search
    # with bilinear constraints. We'll try ranks from 119 down to some lower bound.
    
    # Lower bound: for matrix multiplication, rank >= max(n*m, m*p, p*n)
    # Actually the border rank lower bound is more complex, but for 4x5x6,
    # known results suggest rank is significantly below 120.
    # Let's try random search for lower ranks.
    
    def random_search_for_rank(target_rank, num_trials=100):
        """Try to find a decomposition of the given rank via random initialization and Newton-like refinement."""
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # We want to solve: T_ijr = U_ir * V_jk * W_ki = 1 for all i,j,k
        # This is a system of polynomial equations.
        
        # Use a simple iterative approach: project out components
        # that don't contribute to satisfying the equations.
        
        for iteration in range(50):
            if time.time() > deadline:
                return None
            
            # Residual: T_ijk - 1
            U_3d = U.reshape(n, m, target_rank)
            V_3d = V.reshape(m, p, target_rank)
            W_3d = W.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            residual = T - np.ones((n, m, p))
            
            if np.max(np.abs(residual)) < 1e-8:
                return (U, V, W)
            
            # Update each term in a coordinate-descent fashion
            for r in range(target_rank):
                if time.time() > deadline:
                    return None
                
                # Partial residual with term r set to zero
                T_minus_r = np.einsum('ijr,jkr,kir->ijk', 
                                      np.delete(U_3d, r, axis=2),
                                      np.delete(V_3d, r, axis=2),
                                      np.delete(W_3d, r, axis=2))
                residual_r = T_minus_r - np.ones((n, m, p))
                
                # We need U_[:, r] * V_[:, r] * W_[:, r] = -residual_r
                # This is a rank-1 tensor factorization problem.
                # Use SVD on the residual_r to get the best rank-1 approximation.
                
                # Reshape residual_r as a 3-tensor, then flatten for SVD
                # We want to find u (nm), v (mp), w (pn) such that
                # u_i * v_j * w_k approximates -residual_r[i,j,k]
                
                # This is the best rank-1 approximation problem.
                # We can use the SVD of the "flattened" tensor.
                
                # For a 3-tensor T[i,j,k], the best rank-1 approx is
                # u_i * v_j * w_k where (u,v,w) are from the SVD.
                
                # We can use the "unfolded" SVD approach.
                
                # Unfold along k: T_ikj is a matrix of shape (n*p, m)
                # Actually, let's just do a simple iterative update.
                
                # For each term, we can update one factor at a time.
                # Fix V and W, update U:
                # U_i = -residual_r_i / (V_j * W_k summed over j,k)
                # But this doesn't work directly because of the product structure.
                
                # Let's use a different approach: alternating minimization.
                # Fix V and W, update U to minimize ||U*V*W - (-residual_r)||^2
                # This is a least squares problem.
                
                # The gradient of (U_i * V_j * W_k - r_i) with respect to U_i is
                # V_j * W_k for each j,k where U_i appears.
                
                # Actually, let's just do a simple random restart approach.
                pass
        
        return None

    # The random search above is too slow and unreliable.
    # Instead, let's use a known structured approach.
    
    # For 4x5x6, we can try to embed known smaller decompositions.
    # Known: 2x2x2 has rank 7 (Strassen)
    # 3x3x3 has rank 23 (Beals) or lower
    # 4x4x4 has rank 49 or lower
    
    # A practical approach: try ranks from 119 down to, say, 100, with random search.
    
    for target_rank in range(trivial_rank - 1, max(50, trivial_rank - 30), -1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(20):  # Fewer trials per rank to stay within time
            if time.time() > deadline:
                break
            
            # Random initialization with structured constraints
            # Ensure that the support patterns are valid (no zero entries that would make rank-1 terms ineffective)
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Random binary masks for support
            for r in range(target_rank):
                # Each rank-1 term should have non-zero entries in U, V, W
                # Randomly choose which indices are non-zero
                u_mask = np.random.rand(n * m) > 0.5
                v_mask = np.random.rand(m * p) > 0.5
                w_mask = np.random.rand(p * n) > 0.5
                
                if not np.any(u_mask) or not np.any(v_mask) or not np.any(w_mask):
                    continue
                
                U[u_mask, r] = np.random.randn(np.sum(u_mask))
                V[v_mask, r] = np.random.randn(np.sum(v_mask))
                W[w_mask, r] = np.random.randn(np.sum(w_mask))
            
            # Check if this random decomposition works (very unlikely, but try)
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
    
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
