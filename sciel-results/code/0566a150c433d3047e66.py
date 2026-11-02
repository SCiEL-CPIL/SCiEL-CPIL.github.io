import numpy as np
from itertools import product, combinations, permutations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if consistent
    # For 2x3 * 3x3:
    # i in [0,1], j in [0,1,2], k in [0,1,2]
    # U index: i*m + j = 3i + j (6 values)
    # V index: j*p + k = 3j + k (9 values)
    # W index: k*n + i = 2k + i (6 values)
    
    # A rank-1 tensor is defined by:
    # U vector u (length 6), V vector v (length 9), W vector w (length 6)
    # The tensor entry T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    # For the standard matrix multiplication tensor, T[i,j,k] = 1 for all valid (i,j,k)
    # where i in [0,1], j in [0,2], k in [0,2].
    
    # We need to find a set of rank-1 tensors whose sum equals T.
    # Each rank-1 tensor is defined by three vectors.
    
    # Key insight: For a rank-1 tensor to contribute to T[i,j,k],
    # we need u[i*m+j] * v[j*p+k] * w[k*n+i] to be non-zero.
    # The support of a rank-1 tensor is the set of (i,j,k) where all three are non-zero.
    
    # To minimize rank, we want each rank-1 tensor to cover multiple (i,j,k) entries.
    # The maximum number of entries a single rank-1 tensor can cover is limited by
    # the structure of the indices.
    
    # Let's think about what a rank-1 tensor looks like:
    # u is a vector of length 6 (indices 0-5 corresponding to (i,j) pairs)
    # v is a vector of length 9 (indices 0-8 corresponding to (j,k) pairs)
    # w is a vector of length 6 (indices 0-5 corresponding to (k,i) pairs)
    
    # For a rank-1 tensor to be "simple", let's consider vectors with small support.
    
    # Strategy: Search for a decomposition with rank <= 15.
    # We'll use a deterministic approach:
    # 1. Generate candidate rank-1 patterns (support structures)
    # 2. For each pattern, determine the possible coefficient values
    # 3. Search for a combination of patterns that sums to T
    
    # A rank-1 pattern is defined by:
    # - Support of u: subset of {0,...,5}
    # - Support of v: subset of {0,...,8}
    # - Support of w: subset of {0,...,5}
    
    # The covered (i,j,k) triples are those where:
    # i*m+j in supp_u, j*p+k in supp_v, k*n+i in supp_w
    
    # Let's enumerate all possible support patterns that cover at least 2 entries.
    # This is still a large space, so we need to be smarter.
    
    # Alternative approach: Use the fact that the tensor is dense (all 18 entries are 1).
    # We need sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all (i,j,k).
    
    # Let's try a structured search:
    # For each rank r from 15 down to 1, try to find a decomposition.
    
    # Given the complexity, let's first try to find a known decomposition.
    # The standard lower bound for 2x3 * 3x3 is rank >= 15 (from known results).
    # Let's try to construct a rank 15 decomposition.
    
    # One approach: Use the "border rank" techniques or known constructions.
    # For small matrices, we can use explicit constructions.
    
    # Let's try a different approach: formulate as an integer linear programming problem
    # and solve it using a simple search since the dimensions are small.
    
    # The number of possible rank-1 patterns is large, but we can restrict to
    # patterns where each vector has support size 1 or 2.
    
    # Let's define a pattern by its support sets.
    # For support size 1: u has one non-zero, v has one non-zero, w has one non-zero.
    # This covers exactly one (i,j,k) if consistent, or zero if not.
    
    # For support size 2: more complex.
    
    # Let's try a randomized search with a time limit.
    import time
    start_time = time.time()
    
    # Fallback: schoolbook rank 18
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to find rank 15 decomposition
    # We'll use a greedy approach with local search.
    
    # Define the target tensor
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = 1
    
    # A rank-1 tensor is defined by three vectors u, v, w.
    # The contribution to T[i,j,k] is u[3*i+j] * v[3*j+k] * w[2*k+i].
    
    # We want to find vectors u_r, v_r, w_r for r=1,...,R such that
    # sum_r u_r[3*i+j] * v_r[3*j+k] * w_r[2*k+i] = 1 for all (i,j,k).
    
    # Let's try R = 15.
    
    # We'll use a randomized local search:
    # 1. Initialize R rank-1 tensors randomly
    # 2. Iteratively adjust one tensor at a time to reduce the error
    
    import random
    
    def compute_error(U_list, V_list, W_list):
        """Compute the error between the sum of rank-1 tensors and T."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(len(U_list)):
                        val += U_list[r][3*i+j] * V_list[r][3*j+k] * W_list[r][2*k+i]
                    error += (val - 1.0) ** 2
        return error
    
    def random_init(R):
        """Initialize R rank-1 tensors with random small integer/half-integer values."""
        U_list = []
        V_list = []
        W_list = []
        for r in range(R):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            # Randomly set a few non-zero entries
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, n*m-1)
                u[idx] = random.choice([-1.0, 1.0, -0.5, 0.5])
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, m*p-1)
                v[idx] = random.choice([-1.0, 1.0, -0.5, 0.5])
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, p*n-1)
                w[idx] = random.choice([-1.0, 1.0, -0.5, 0.5])
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        return U_list, V_list, W_list
    
    best_error = float('inf')
    best_solution = None
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() - start_time > 60:
            break
        
        U_list, V_list, W_list = random_init(15)
        current_error = compute_error(U_list, V_list, W_list)
        
        # Local search: try to improve by adjusting one tensor
        for iteration in range(200):
            if time.time() - start_time > 60:
                break
            
            # Pick a random tensor to adjust
            r = random.randint(0, 14)
            
            # Try changing one entry of u, v, or w
            improved = False
            for vec_idx in range(3):
                if vec_idx == 0:
                    vec = U_list[r]
                    vec_len = n * m
                elif vec_idx == 1:
                    vec = V_list[r]
                    vec_len = m * p
                else:
                    vec = W_list[r]
                    vec_len = p * n
                
                for idx in range(vec_len):
                    old_val = vec[idx]
                    for new_val in [-1.0, 1.0, -0.5, 0.5, 0.0]:
                        if new_val == old_val:
                            continue
                        vec[idx] = new_val
                        new_error = compute_error(U_list, V_list, W_list)
                        if new_error < current_error:
                            current_error = new_error
                            improved = True
                            break
                        else:
                            vec[idx] = old_val
                
                if improved:
                    break
            
            if not improved:
                # Try a larger move: reset one tensor
                vec_idx = random.randint(0, 2)
                if vec_idx == 0:
                    U_list[r] = [0.0] * (n * m)
                    for _ in range(random.randint(1, 3)):
                        idx = random.randint(0, n*m-1)
                        U_list[r][idx] = random.choice([-1.0, 1.0, -0.5, 0.5])
                elif vec_idx == 1:
                    V_list[r] = [0.0] * (m * p)
                    for _ in range(random.randint(1, 3)):
                        idx = random.randint(0, m*p-1)
                        V_list[r][idx] = random.choice([-1.0, 1.0, -0.5, 0.5])
                else:
                    W_list[r] = [0.0] * (p * n)
                    for _ in range(random.randint(1, 3)):
                        idx = random.randint(0, p*n-1)
                        W_list[r][idx] = random.choice([-1.0, 1.0, -0.5, 0.5])
                current_error = compute_error(U_list, V_list, W_list)
        
        if current_error < best_error:
            best_error = current_error
            best_solution = (U_list, V_list, W_list)
        
        if best_error < 1e-10:
            break
    
    if best_solution is not None and best_error < 1e-10:
        U_list, V_list, W_list = best_solution
        U = [[U_list[r][i] for r in range(15)] for i in range(n * m)]
        V = [[V_list[r][i] for r in range(15)] for i in range(m * p)]
        W = [[W_list[r][i] for r in range(15)] for i in range(p * n)]
        return {"U": U, "V": V, "W": W}
    
    return fallback
# EVOLVE-BLOCK-END
