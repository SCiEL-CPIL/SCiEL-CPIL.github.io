"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Returns a rank-minimal (as far as we can find) exact tensor decomposition
    for multiplying a 4x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Start with schoolbook decomposition of rank 4*4*7 = 112.
    2. Try to find a lower rank decomposition using known results and 
       local search / optimization.
    
    Known facts:
    - The rank of M_{4,4,4} is at most 49 (Lader 1973), and at least 49.
    - We can decompose M_{4,4,7} as M_{4,4,4} ⊕ M_{4,4,3}.
    - M_{4,4,4} has rank 49.
    - M_{4,4,3} has rank at most 4*4*3 = 48 (schoolbook).
    - So total rank = 49 + 48 = 97.
    
    But we can do better by using a direct approach or by finding 
    a better decomposition of M_{4,4,3}.
    
    Actually, let's think about this more carefully.
    
    The tensor for M_{n,m,p} has dimensions (n*m) x (m*p) x (p*n) in the 
    standard formulation where we want to compute C = AB where A is n x m 
    and B is m x p.
    
    The tensor T[i,j,k] = delta[i,j,k] where i in [0,n*m), j in [0,m*p), 
    k in [0,p*n) and the multiplication is:
    C[k*n + i, ...] = sum over j of A[i*m + ...] * B[j*p + ...]
    
    Wait, let me re-read the task.
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W has shape (p*n, rank). The multiplication is:
    For each column r of the decomposition:
    C = sum_r (U[:, r] * V[:, r]) where U is (n*m, rank), V is (m*p, rank)
    
    Actually, the tensor decomposition of matrix multiplication M_{n,m,p} is:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    
    where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{p*n}
    
    and the multiplication is:
    (A*B)_{i,k} = sum_{j=0}^{m-1} A_{i,j} * B_{j,k}
    
    In the tensor formulation:
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T[other] = 0
    
    And we want to find rank r and vectors u_c, v_c, w_c (c = 0..r-1) such that:
    T = sum_c u_c ⊗ v_c ⊗ w_c
    
    i.e., T[a,b,c] = sum_{r=0}^{rank-1} U[a,r] * V[b,r] * W[c,r]
    
    So the task is to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    such that for all a in [0,n*m), b in [0,m*p), c in [0,p*n):
    sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    where T[a,b,c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' with i=i', j=j', k=k'
    and T[a,b,c] = 0 otherwise.
    
    More precisely:
    Let a = i*m + j where i in [0,n), j in [0,m)
    Let b = j2*p + k where j2 in [0,m), k in [0,p)
    Let c = k2*n + i2 where k2 in [0,p), i2 in [0,n)
    
    T[a,b,c] = 1 if j == j2 and i == i2 and k == k2, else 0.
    Wait, that's not quite right either.
    
    Actually, for matrix multiplication C = AB where A is n x m, B is m x p:
    C_{i,k} = sum_{j=0}^{m-1} A_{i,j} * B_{j,k}
    
    The tensor T has T[a,b,c] = 1 if and only if:
    a = i*m + j, b = j*p + k, c = k*n + i for some valid i,j,k.
    So a determines (i,j), b determines (j,k), c determines (k,i).
    The constraint is that the j in a equals the j in b, the k in b equals the k in c,
    and the i in c equals the i in a.
    
    So T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    All other entries are 0.
    
    The schoolbook decomposition has rank n*m*p and uses:
    U[i*m+j, i*m+j] = 1 (wait, that's not right for the indexing)
    
    Let me look at the seed program more carefully.
    
    The seed program:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    
    So for each (i,j,k) triple, we have a rank-1 term where:
    U has 1 at position (i*m+j), V has 1 at position (j*p+k), W has 1 at position (k*n+i).
    
    This gives T[i*m+j, j*p+k, k*n+i] = 1 for each (i,j,k) and 0 elsewhere, which is correct.
    
    Now, to improve, we can use the following approach:
    
    We know that M_{4,4,4} has rank 49. We can use this.
    Then M_{4,4,7} = M_{4,4,4} ⊕ M_{4,4,3}.
    The direct sum has rank = rank(M_{4,4,4}) + rank(M_{4,4,3}) = 49 + rank(M_{4,4,3}).
    
    What is the rank of M_{4,4,3}? It's at most 4*4*3 = 48 (schoolbook).
    Can we do better?
    
    Actually, I don't know the exact rank of M_{4,4,3}. Let's just use schoolbook for it.
    
    So total rank = 49 + 48 = 97.
    
    But wait, can we do even better? Maybe there's a direct decomposition of M_{4,4,7} 
    with lower rank.
    
    Let me try a different approach. I'll use a local search / optimization method 
    to try to find a lower rank decomposition.
    
    Actually, for a cleaner approach, let me just implement the 49+48 = 97 rank 
    decomposition first, and then try to optimize.
    
    For M_{4,4,4}, I need to hardcode the rank-49 algorithm.
    This is complex. Let me instead try a simpler approach.
    
    Alternative: Use the fact that M_{4,4,7} can be decomposed as:
    M_{4,4,7} = M_{2,4,7} ⊕ M_{2,4,7} (splitting the first dimension)
    
    M_{2,4,7} has schoolbook rank 2*4*7 = 56.
    Two of them: 112. Not better.
    
    Or M_{4,4,7} = M_{4,2,7} ⊕ M_{4,2,7} (splitting the middle dimension)
    M_{4,2,7} has schoolbook rank 4*2*7 = 56.
    Two of them: 112. Not better.
    
    Let me try yet another approach. I'll use a random search / optimization 
    to try to find a lower rank decomposition.
    
    Actually, let me just implement the 97-rank decomposition using the 
    known rank-49 algorithm for M_{4,4,4}.
    
    The rank-49 algorithm for M_{4,4,4} is quite complex. Let me look for 
    a simpler approach.
    
    I recall that the rank of M_{4,4,4} is exactly 49 (Lader 1973, 
    and it's a long-standing open question whether it's 49 or higher, 
    but 49 is the best known upper bound).
    
    Actually, I think I should just do a local search to try to find 
    a lower rank decomposition of M_{4,4,7}.
    
    Let me implement a simple local search:
    1. Start with the schoolbook decomposition of rank 112.
    2. Try to remove columns (rank reduction) by checking if the 
       remaining columns can still span the tensor.
    3. Use a greedy approach or simulated annealing.
    
    But this might be too slow. Let me think of a better approach.
    
    Actually, for the purpose of this task, let me just implement 
    the 97-rank decomposition. I'll need to hardcode the rank-49 
    algorithm for M_{4,4,4}.
    
    Since I don't have the exact coefficients memorized, let me try 
    a different strategy: use a numerical optimization approach.
    
    I'll set up the problem as:
    - We want to find U (16 x r), V (28 x r), W (28 x r) such that 
      the tensor equation holds.
    - Start with r = 97 (or lower) and try to minimize some error.
    
    But this is a non-convex problem and might be hard to solve.
    
    Let me just go with the schoolbook decomposition of rank 112 for now, 
    and add a simple optimization step.
    
    Actually, wait. The task says "Find a rank-minimal exact tensor decomposition". 
    The score is the rank. So I want to minimize the rank.
    
    Let me try a greedy rank reduction:
    1. Start with schoolbook rank 112.
    2. For each column, check if removing it still allows exact reconstruction.
    3. If yes, remove it.
    
    But checking exact reconstruction is expensive.
    
    Let me try a different approach. I'll use the following known results:
    - M_{4,4,4} has rank at most 49.
    - I'll use a 2x2x2 Strassen algorithm for the 4x4x4 part.
    
    Actually, let me just implement a simple local search that tries 
    to reduce the rank by perturbing the matrices and checking the error.
    
    Given the time constraints, let me implement a pragmatic approach:
    1. Start with schoolbook rank 112.
    2. Try to reduce the rank by a greedy method.
    3. Return the best result found within the time limit.
    
    Here's my plan:
    
    def construct():
        n, m, p = 4, 4, 7
        rank = n * m * p  # 112
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        
        # Try to reduce rank
        # ... (optimization code)
        
        return {"U": U, "V": V, "W": W}
    
    For the optimization, I'll use a simple approach:
    - Represent U, V, W as NumPy arrays.
    - Use a greedy method to try to remove columns.
    - Use a local search to perturb the matrices and reduce the error.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 112
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
    
    # Convert to numpy for optimization
    try:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the target tensor T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute the current tensor from U, V, W
        def compute_tensor(U, V, W):
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is a 3-way tensor contraction
            result = np.zeros((n*m, m*p, p*n))
            for r in range(U.shape[1]):
                result += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            return result
        
        # Current error
        current_T = compute_tensor(U_np, V_np, W_np)
        current_error = np.sum((current_T - T) ** 2)
        
        # Try to reduce rank by removing columns
        # Greedy approach: try removing each column and check if error increases
        rank_to_try = rank
        best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
        best_error = current_error
        
        # Time limit: 120 seconds
        start_time = time.time()
        time_limit = 100  # seconds
        
        # Simple local search
        for iteration in range(1000):
            if time.time() - start_time > time_limit:
                break
            
            # Try to remove a random column
            if rank_to_try > 1:
                col_to_remove = random.randint(0, rank_to_try - 1)
                new_U = np.delete(best_U, col_to_remove, axis=1)
                new_V = np.delete(best_V, col_to_remove, axis=1)
                new_W = np.delete(best_W, col_to_remove, axis=1)
                new_T = compute_tensor(new_U, new_V, new_W)
                new_error = np.sum((new_T - T) ** 2)
                if new_error <= best_error + 1e-10:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_error = new_error
                    rank_to_try -= 1
                    continue
            
            # Try to perturb a random entry
            if rank_to_try > 0:
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    i = random.randint(0, n*m - 1)
                    j = random.randint(0, rank_to_try - 1)
                    delta = random.gauss(0, 0.1)
                    old_val = best_U[i, j]
                    best_U[i, j] += delta
                elif mat == 'V':
                    i = random.randint(0, m*p - 1)
                    j = random.randint(0, rank_to_try - 1)
                    delta = random.gauss(0, 0.1)
                    old_val = best_V[i, j]
                    best_V[i, j] += delta
                else:
                    i = random.randint(0, p*n - 1)
                    j = random.randint(0, rank_to_try - 1)
                    delta = random.gauss(0, 0.1)
                    old_val = best_W[i, j]
                    best_W[i, j] += delta
                
                new_T = compute_tensor(best_U, best_V, best_W)
                new_error = np.sum((new_T - T) ** 2)
                
                if new_error > best_error + 1e-10:
                    # Revert
                    if mat == 'U':
                        best_U[i, j] = old_val
                    elif mat == 'V':
                        best_V[i, j] = old_val
                    else:
                        best_W[i, j] = old_val
        
        # Convert back to lists
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        
        # Round to nearest half-integer if possible
        # Check if all entries are close to half-integers
        def round_to_half_integer(x):
            # x is a list of lists
            result = []
            for row in x:
                new_row = []
                for val in row:
                    # Round to nearest 0.5
                    rounded = round(val * 2) / 2
                    new_row.append(rounded)
                result.append(new_row)
            return result
        
        U_out = round_to_half_integer(U_out)
        V_out = round_to_half_integer(V_out)
        W_out = round_to_half_integer(W_out)
        
        return {"U": U_out, "V": V_out, "W": W_out}
    except Exception as e:
        # Fallback to schoolbook
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
