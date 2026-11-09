"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x7 matrix.
    
    Strategy:
    1. Try to find a low-rank decomposition using a randomized search + local optimization.
    2. The tensor T has indices (i, j, k) where i in [0,2), j in [0,3), k in [0,7).
       U has shape (2*3, rank) = (6, rank)
       V has shape (3*7, rank) = (21, rank)
       W has shape (7*2, rank) = (14, rank)
       
    The decomposition condition is:
    For all (i, j, k): T[i,j,k] = sum_r U[i*3+j, r] * V[j*7+k, r] * W[k*2+i, r]
    
    Since the standard matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k),
    we need: sum_r U[i*3+j, r] * V[j*7+k, r] * W[k*2+i, r] = 1 for all i,j,k.
    
    This is a system of 2*3*7 = 42 equations.
    
    We'll use a greedy approach: try to find a low-rank decomposition by iteratively
    adding terms. Start with rank=1 and increase until we find a valid decomposition.
    
    For each rank r, we try to solve the system. This is a non-linear system, but we
    can use the fact that if we fix U and V, then W is determined linearly (and vice versa).
    
    Actually, let's think about this differently. The tensor is the standard matrix multiplication
    tensor M_{2,3,7}. We want to decompose it into a sum of rank-1 tensors.
    
    A rank-1 tensor is U ⊗ V ⊗ W where U is a vector of length 6, V of length 21, W of length 14.
    
    The condition is: for all (i,j,k), sum_r U_r[i*3+j] * V_r[j*7+k] * W_r[k*2+i] = 1.
    
    Let's try a structured approach. Note that the 2x3x7 matrix multiplication can be viewed
    as: C = A * B where A is 2x3 and B is 3x7.
    
    We can try to use the fact that 7 = 3 + 4, and decompose the multiplication into
    two parts: A * B_1 (2x3 times 3x3) and A * B_2 (2x3 times 3x4).
    
    For 2x3x3: the standard decomposition has rank 6. Can we do better?
    For 2x3x4: the standard decomposition has rank 8. Can we do better?
    
    Actually, let's just do a direct search. We'll try ranks from 1 upward and use
    a randomized local search to find a valid decomposition.
    
    Given the time constraint, let's try a more structured approach.
    
    Key insight: The matrix multiplication tensor M_{n,m,p} has a known lower bound.
    For M_{2,3,7}, the schoolbook rank is 2*3*7 = 42.
    
    Let's try to find a decomposition with rank significantly less than 42.
    
    Approach: Use the fact that we can combine terms. For example, if we have two rank-1
    terms that differ only in one factor, we might be able to combine them.
    
    Actually, let's try a direct numerical approach. We'll parameterize the decomposition
    and use optimization to find a valid one.
    
    For a given rank r, we have:
    - U: 6 * r variables
    - V: 21 * r variables  
    - W: 14 * r variables
    Total: 41 * r variables
    
    And we have 42 equations (one for each (i,j,k) pair).
    
    For r >= 2, we have more variables than equations, so a solution should exist (over reals).
    The question is whether we can find one with rational/half-integer entries.
    
    Let's try r=2 first, then r=3, etc.
    
    For each r, we'll use a randomized initialization and then use a local search to
    satisfy the 42 equations.
    
    The objective function is: sum_{i,j,k} (sum_{r'} U[r', i*3+j] * V[r', j*7+k] * W[r', k*2+i] - 1)^2
    
    We want to minimize this to 0.
    
    Let's implement this with a simple gradient-free approach (e.g., Nelder-Mead or a
    randomized local search).
    
    Given the time limit, let's try a few ranks and see if we can find a valid decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds margin
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    num_equations = n * m * p  # 42
    
    # Precompute index mappings
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    # Try to find a decomposition for increasing ranks
    best_rank = n * m * p  # Schoolbook rank
    best_decomp = None
    
    # First, let's construct the schoolbook decomposition as a fallback
    rank_schoolbook = n * m * p
    U_school = [[0.0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0.0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0.0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_decomp = (U_school, V_school, W_school)
    
    # Try lower ranks
    for rank in range(2, min(rank_schoolbook, 25)):  # Try ranks 2 to 24
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        max_attempts = 20
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + rank)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Local search to minimize the objective
            # Objective: sum_{i,j,k} (sum_r U[i*3+j, r] * V[j*7+k, r] * W[k*2+i, r] - 1)^2
            
            def objective(params):
                """Compute the objective function from flattened parameters."""
                # Unpack parameters
                u_size = n * m * rank
                v_size = m * p * rank
                w_size = p * n * rank
                
                U_flat = params[:u_size]
                V_flat = params[u_size:u_size + v_size]
                W_flat = params[u_size + v_size:u_size + v_size + w_size]
                
                U_mat = U_flat.reshape(n * m, rank)
                V_mat = V_flat.reshape(m * p, rank)
                W_mat = W_flat.reshape(p * n, rank)
                
                obj = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                            obj += (val - 1.0) ** 2
                return obj
            
            # Simple local search with random perturbations
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            current_obj = objective(params)
            
            # Gradient-free local search
            step_size = 0.1
            for step in range(200):
                if time.time() > deadline - 30:
                    break
                
                if current_obj < 1e-10:
                    found = True
                    break
                
                # Try a random perturbation
                new_params = params + np.random.randn(len(params)) * step_size
                new_obj = objective(new_params)
                
                if new_obj < current_obj:
                    params = new_params
                    current_obj = new_obj
                    step_size *= 0.95
                else:
                    step_size *= 0.9
                
                if step_size < 1e-8:
                    break
            
            if current_obj < 1e-10:
                # Extract the solution
                u_size = n * m * rank
                v_size = m * p * rank
                w_size = p * n * rank
                
                U_sol = params[:u_size].reshape(n * m, rank)
                V_sol = params[u_size:u_size + v_size].reshape(m * p, rank)
                W_sol = params[u_size + v_size:u_size + v_size + w_size].reshape(p * n, rank)
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_sol[i * m + j, r] * V_sol[j * p + k, r] * W_sol[k * n + i, r]
                            if abs(val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    found = True
                    # Convert to lists
                    U_list = U_sol.tolist()
                    V_list = V_sol.tolist()
                    W_list = W_sol.tolist()
                    
                    best_rank = rank
                    best_decomp = (U_list, V_list, W_list)
                    break
        
        if found:
            break
    
    # If we found a better decomposition, return it; otherwise return schoolbook
    if best_decomp is not None and best_rank < rank_schoolbook:
        U, V, W = best_decomp
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = best_decomp
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
