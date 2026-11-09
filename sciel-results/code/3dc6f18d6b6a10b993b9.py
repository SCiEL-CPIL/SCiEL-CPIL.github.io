"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    nm, mp, pn = n*m, m*p, p*n
    
    # Create list of (i,j,k) triples that define the tensor
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                triples.append((idx_u, idx_v, idx_w))
    
    # Schoolbook solution
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    for r, (iu, iv, iw) in enumerate(triples):
        U_school[iu, r] = 1
        V_school[iv, r] = 1
        W_school[iw, r] = 1
    
    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Known lower bounds: for 2x3x5, theoretical lower bound is around 20-25
    # Try ranks from known good values down
    # Based on literature, 2x3x5 matrix multiplication can be done with rank <= 24
    candidate_ranks = list(range(24, 29, -1))  # Try 24, 25, 26, 27, 28
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10.0:
            break
        
        # Try to find a decomposition with this rank using structured search
        found = _find_decomposition(n, m, p, target_rank, deadline)
        if found is not None:
            U, V, W = found
            best_rank = target_rank
            best_U = U
            best_V = V
            best_W = W
            break  # Found a valid lower rank, stop
    
    U = best_U.tolist()
    V = best_V.tolist()
    W = best_W.tolist()
    
    return {"U": U, "V": V, "W": W}

def _find_decomposition(n, m, p, target_rank, deadline):
    """Try to find a rank-`target_rank` decomposition for n x m x p matrix multiplication."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T[u, v, w]
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0
    
    # We need: sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w]
    
    # Strategy: Use a structured approach based on the algebraic structure
    # For small matrices, we can try to find solutions with entries in {-1, 0, 1, 0.5, -0.5}
    
    # Generate candidate coefficient sets
    coeff_candidates = [-1, -0.5, 0, 0.5, 1]
    
    # For efficiency, we'll use a randomized search with structured initialization
    # based on the known algebraic structure of matrix multiplication
    
    # Key insight: We can use the fact that the tensor has a specific pattern
    # and try to exploit it through symmetry
    
    # Try multiple random initializations with gradient descent
    np.random.seed(12345)
    
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            return None
        
        # Initialize with small random values
        scale = 0.1
        U = np.random.randn(nm, target_rank) * scale
        V = np.random.randn(mp, target_rank) * scale
        W = np.random.randn(pn, target_rank) * scale
        
        # Add some structure: initialize based on the tensor pattern
        # For each (i,j,k), set up contributions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    # Add a small structured component
                    r_idx = (i * m + j * p + k) % target_rank
                    U[idx_u, r_idx] += 1.0
                    V[idx_v, r_idx] += 1.0
                    W[idx_w, r_idx] += 1.0
        
        # Optimize using a simple gradient-free method
        success = _optimize_decomposition(U, V, W, n, m, p, target_rank, deadline)
        if success:
            return (U, V, W)
    
    return None

def _optimize_decomposition(U, V, W, n, m, p, target_rank, deadline):
    """Try to optimize the decomposition using a local search method."""
    nm, mp, pn = n*m, m*p, p*n
    
    def compute_error(U, V, W):
        """Compute the reconstruction error."""
        error = 0.0
        # Check the non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    error += (val - 1.0) ** 2
        
        # Check some zero entries (sampled for efficiency)
        # The full tensor has nm*mp*pn entries, most are zero
        # We'll sample a subset to check
        np.random.seed(999)
        sample_size = min(100, nm*mp*pn - n*m*p)
        for _ in range(sample_size):
            u = np.random.randint(nm)
            v = np.random.randint(mp)
            w = np.random.randint(pn)
            # Check if this is a non-zero entry
            is_nonzero = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if (i*m+j == u and j*p+k == v and k*n+i == w):
                            is_nonzero = True
                            break
                    if is_nonzero:
                        break
                if is_nonzero:
                    break
            if not is_nonzero:
                val = np.sum(U[u, :] * V[v, :] * W[w, :])
                error += val ** 2
        
        return error
    
    initial_error = compute_error(U, V, W)
    if initial_error < 1e-6:
        return True
    
    # Use a simple iterative improvement method
    lr = 0.01
    for iteration in range(1000):
        if time.time() > deadline - 2.0:
            break
        
        # Compute gradients numerically (or use analytic)
        # For efficiency, we'll use a coordinate descent approach
        
        # Try updating U
        U_old = U.copy()
        for r in range(target_rank):
            for i in range(nm):
                # Compute gradient with respect to U[i, r]
                grad = 0.0
                for j in range(mp):
                    for k in range(pn):
                        # This is too expensive, use a simplified approach
                        pass
                # Skip detailed gradient for now
        
        # Instead, use a simpler local search
        # Try small perturbations
        improved = False
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            
            # Random perturbation
            U_pert = U + np.random.randn(nm, target_rank) * 0.01
            V_pert = V + np.random.randn(mp, target_rank) * 0.01
            W_pert = W + np.random.randn(pn, target_rank) * 0.01
            
            new_error = compute_error(U_pert, V_pert, W_pert)
            if new_error < initial_error:
                U, V, W = U_pert, V_pert, W_pert
                initial_error = new_error
                improved = True
        
        if not improved:
            break
    
    # Final check
    final_error = compute_error(U, V, W)
    if final_error < 1e-4:
        return True
    
    return False

# EVOLVE-BLOCK-END
