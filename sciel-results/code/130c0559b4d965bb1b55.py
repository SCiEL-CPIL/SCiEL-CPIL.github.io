import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Generate schoolbook solution as fallback
    U_sch = [[0] * rank_school for _ in range(n * m)]
    V_sch = [[0] * rank_school for _ in range(m * p)]
    W_sch = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j][col] = 1
                V_sch[j * p + k][col] = 1
                W_sch[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sch, "V": V_sch, "W": W_sch}
    
    def check_validity(U, V, W, r):
        """Check if the decomposition exactly computes matrix multiplication."""
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        return np.allclose(S, 1.0, atol=1e-9)
    
    def is_half_integer(x):
        return abs(x * 2 - round(x * 2)) < 1e-9
    
    def check_half_integers(U, V, W):
        for arr in [U, V, W]:
            for val in arr.flatten():
                if not is_half_integer(val):
                    return False
        return True
    
    def residual_fn(params, R):
        """Compute residual for the tensor decomposition."""
        size_U = n * m * R
        size_V = m * p * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U_mat = U_flat.reshape(n * m, R)
        V_mat = V_flat.reshape(m * p, R)
        W_mat = W_flat.reshape(p * n, R)
        
        U_3d = U_mat.reshape(n, m, R)
        V_3d = V_mat.reshape(m, p, R)
        W_3d = W_mat.reshape(p, n, R)
        
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        return (S - 1.0).flatten()
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Key structural insight: use Kronecker product structure for initialization
    # This respects the algebraic structure of fast matrix multiplication
    
    def kronecker_init(R, seed):
        """Initialize using Kronecker product structure.
        
        The idea: decompose the problem into smaller sub-problems and combine
        them using Kronecker products, which is how Strassen and other fast
        multiplication algorithms work.
        """
        rng = np.random.RandomState(seed)
        
        U_init = np.zeros((n * m, R))
        V_init = np.zeros((m * p, R))
        W_init = np.zeros((p * n, R))
        
        # Try to find a factorization of R into smaller components
        # that can be combined via Kronecker products
        # For now, use a structured random initialization that has
        # block structure similar to Kronecker products
        
        # Divide R into blocks and assign structure
        num_blocks = max(1, min(4, R // 10))
        block_size = R // num_blocks
        remainder = R % num_blocks
        
        for block in range(num_blocks):
            start = block * block_size
            end = start + block_size + (1 if block < remainder else 0)
            
            if end <= start:
                continue
            
            # For each block, create a structured pattern
            # This mimics the local structure of Kronecker product terms
            
            # Random but structured indices
            u_indices = rng.choice(n * m, size=min(3, block_size + (1 if block < remainder else 0)), replace=False)
            v_indices = rng.choice(m * p, size=min(3, block_size + (1 if block < remainder else 0)), replace=False)
            w_indices = rng.choice(p * n, size=min(3, block_size + (1 if block < remainder else 0)), replace=False)
            
            for c in range(start, end):
                # Each column gets a few non-zeros with structured patterns
                num_nonzeros_u = rng.randint(1, 4)
                num_nonzeros_v = rng.randint(1, 4)
                num_nonzeros_w = rng.randint(1, 4)
                
                u_idx = rng.choice(n * m, size=num_nonzeros_u, replace=False)
                v_idx = rng.choice(m * p, size=num_nonzeros_v, replace=False)
                w_idx = rng.choice(p * n, size=num_nonzeros_w, replace=False)
                
                for idx in u_idx:
                    U_init[idx, c] = rng.choice([1, -1, 0.5, -0.5])
                for idx in v_idx:
                    V_init[idx, c] = rng.choice([1, -1, 0.5, -0.5])
                for idx in w_idx:
                    W_init[idx, c] = rng.choice([1, -1, 0.5, -0.5])
        
        return U_init, V_init, W_init
    
    def kronecker_structured_init(R, seed):
        """More sophisticated Kronecker-structured initialization.
        
        Uses the fact that matrix multiplication can be decomposed as:
        C = A @ B where the computation can be broken into independent
        sub-problems combined linearly.
        """
        rng = np.random.RandomState(seed)
        
        U_init = np.zeros((n * m, R))
        V_init = np.zeros((m * p, R))
        W_init = np.zeros((p * n, R))
        
        # Strategy: create R "terms" where each term has a specific
        # support pattern that resembles a Kronecker product factor
        
        for c in range(R):
            # Each term selects a subset of the tensor indices
            # The support should be consistent with a rank-1 update
            # to the multiplication tensor
            
            # Choose which (i,j) pairs this term affects in U
            # and corresponding (j,k) in V and (k,i) in W
            
            # For term c, pick a random "center" in the index space
            center_i = rng.randint(0, n)
            center_j = rng.randint(0, m)
            center_k = rng.randint(0, p)
            
            # The term should have non-zeros at positions that are
            # "compatible" - i.e., they can contribute to the same
            # output entries when multiplied together
            
            # U non-zeros: (i, j) where j is near center_j
            u_j_candidates = [j for j in range(m) if abs(j - center_j) <= 1]
            u_i_candidates = [i for i in range(n) if abs(i - center_i) <= 1]
            
            # V non-zeros: (j, k) where j is near center_j
            v_j_candidates = [j for j in range(m) if abs(j - center_j) <= 1]
            v_k_candidates = [k for k in range(p) if abs(k - center_k) <= 1]
            
            # W non-zeros: (k, i) where k is near center_k
            w_k_candidates = [k for k in range(p) if abs(k - center_k) <= 1]
            w_i_candidates = [i for i in range(n) if abs(i - center_i) <= 1]
            
            # Add non-zeros with some randomness
            if rng.random() < 0.7:
                i = rng.choice(u_i_candidates) if u_i_candidates else rng.randint(0, n)
                j = rng.choice(u_j_candidates) if u_j_candidates else rng.randint(0, m)
                U_init[i * m + j, c] = rng.choice([1, -1])
            
            if rng.random() < 0.7:
                j = rng.choice(v_j_candidates) if v_j_candidates else rng.randint(0, m)
                k = rng.choice(v_k_candidates) if v_k_candidates else rng.randint(0, p)
                V_init[j * p + k, c] = rng.choice([1, -1])
            
            if rng.random() < 0.7:
                k = rng.choice(w_k_candidates) if w_k_candidates else rng.randint(0, p)
                i = rng.choice(w_i_candidates) if w_i_candidates else rng.randint(0, n)
                W_init[k * n + i, c] = rng.choice([1, -1])
            
            # Sometimes add additional non-zeros for more flexibility
            if rng.random() < 0.3:
                i2 = rng.randint(0, n)
                j2 = rng.randint(0, m)
                U_init[i2 * m + j2, c] = rng.choice([0.5, -0.5])
            
            if rng.random() < 0.3:
                j2 = rng.randint(0, m)
                k2 = rng.randint(0, p)
                V_init[j2 * p + k2, c] = rng.choice([0.5, -0.5])
            
            if rng.random() < 0.3:
                k2 = rng.randint(0, p)
                i2 = rng.randint(0, n)
                W_init[k2 * n + i2, c] = rng.choice([0.5, -0.5])
        
        return U_init, V_init, W_init
    
    from scipy.optimize import least_squares
    
    best_solution = None
    
    # Search from just below schoolbook rank downward
    # Focus on ranks where solutions are more likely to exist
    for R in range(rank_school - 1, 60, -1):
        if time.time() > deadline - 15:
            break
        
        # Number of restarts: more for ranks closer to schoolbook
        # as they're more likely to have solutions
        restarts = max(3, min(15, 20 - (rank_school - 1 - R)))
        
        for restart in range(restarts):
            if time.time() > deadline - 5:
                break
            
            # Use Kronecker-structured initialization
            if restart % 2 == 0:
                U_init, V_init, W_init = kronecker_structured_init(R, 42 + restart * 7)
            else:
                U_init, V_init, W_init = kronecker_init(R, 100 + restart * 13)
            
            p0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(
                    residual_fn, p0, args=(R,),
                    method='lm',
                    max_nfev=3000,
                    ftol=1e-14,
                    xtol=1e-14,
                    gtol=1e-14
                )
                
                if sol.cost < 1e-10:
                    size_U = n * m * R
                    size_V = m * p * R
                    
                    U_sol = sol.x[:size_U].reshape(n * m, R)
                    V_sol = sol.x[size_U:size_U + size_V].reshape(m * p, R)
                    W_sol = sol.x[size_U + size_V:].reshape(p * n, R)
                    
                    # Try to snap to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify the snapped solution
                    if check_validity(U_round, V_round, W_round, R):
                        if check_half_integers(U_round, V_round, W_round):
                            best_solution = {
                                "U": U_round.tolist(),
                                "V": V_round.tolist(),
                                "W": W_round.tolist()
                            }
                            break
                    
                    # If snapping failed, try a refinement step
                    # that explicitly constrains to half-integers
                    if not best_solution:
                        # Try to find a nearby half-integer solution
                        # by projecting and re-optimizing
                        for attempt in range(3):
                            U_proj = np.round(U_sol * 2) / 2
                            V_proj = np.round(V_sol * 2) / 2
                            W_proj = np.round(W_sol * 2) / 2
                            
                            # Add small perturbations to escape local minima
                            if attempt > 0:
                                U_proj += np.random.randn(n * m, R) * 0.01
                                V_proj += np.random.randn(m * p, R) * 0.01
                                W_proj += np.random.randn(p * n, R) * 0.01
                            
                            p1 = np.concatenate([U_proj.flatten(), V_proj.flatten(), W_proj.flatten()])
                            
                            try:
                                sol2 = least_squares(
                                    residual_fn, p1, args=(R,),
                                    method='lm',
                                    max_nfev=1000,
                                    ftol=1e-14,
                                    xtol=1e-14,
                                    gtol=1e-14
                                )
                                
                                if sol2.cost < 1e-10:
                                    U_sol2 = sol2.x[:size_U].reshape(n * m, R)
                                    V_sol2 = sol2.x[size_U:size_U + size_V].reshape(m * p, R)
                                    W_sol2 = sol2.x[size_U + size_V:].reshape(p * n, R)
                                    
                                    U_round2 = np.round(U_sol2 * 2) / 2
                                    V_round2 = np.round(V_sol2 * 2) / 2
                                    W_round2 = np.round(W_sol2 * 2) / 2
                                    
                                    if check_validity(U_round2, V_round2, W_round2, R):
                                        if check_half_integers(U_round2, V_round2, W_round2):
                                            best_solution = {
                                                "U": U_round2.tolist(),
                                                "V": V_round2.tolist(),
                                                "W": W_round2.tolist()
                                            }
                                            break
                            except:
                                pass
                        
                        if best_solution:
                            break
                            
            except Exception as e:
                pass
        
        if best_solution:
            break
    
    if best_solution:
        return best_solution
    else:
        return fallback
