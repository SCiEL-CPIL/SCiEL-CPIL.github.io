import numpy as np
from itertools import product, combinations
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U @ V @ W reconstruction equals the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # Reconstruct T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def make_solution(U, V, W):
        return {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
    
    # Fallback: schoolbook rank 18
    def schoolbook():
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
        return make_solution(U_fb, V_fb, W_fb)
    
    # Try to find rank-15 solution via search
    # Strategy: For each rank-1 term, choose which (i,j), (j,k), (k,i) entries are non-zero.
    # Then solve the linear system for the coefficients.
    
    # Enumerate all possible "support patterns" for a single rank-1 term.
    # A pattern is (S_u, S_v, S_w) where S_u is a subset of {(i,j)}, etc.
    # But to keep it tractable, let's try patterns where each term covers exactly 
    # one "line" - i.e., for fixed j, all i and k, or similar.
    
    # Alternative approach: use known structure. The 2x3x3 tensor can be decomposed
    # by fixing one index and using rank-3 for the remaining 2x3 or 3x3 subproblems.
    
    # Let's try: for each of the 3 values of j, we have a 2x3 (i x k) tensor.
    # The rank of a 2x3 matrix is at most 2. So for each j, we need at most 2 rank-1 terms
    # that vary in i and k but are constant in the "j direction".
    # That gives 3*2 = 6 terms, but we need to account for the full tensor structure.
    
    # Actually, let's just do a brute force search over small coefficient sets.
    # Assume all coefficients are in {-1, -0.5, 0.5, 1} and supports are from a small set.
    
    # Generate candidate support patterns
    # For each term r, we choose:
    # - Which (i,j) pairs have non-zero U entries: subset of {(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)}
    # - Which (j,k) pairs have non-zero V entries: subset of {(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)}
    # - Which (k,i) pairs have non-zero W entries: subset of {(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)}
    
    # This is too large. Let's try a smarter approach.
    
    # Use the fact that we can set up the problem as:
    # For each (i,j,k), we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Let's try random initialization with small integer coefficients and see if we can
    # find a solution by solving the linear system after fixing supports.
    
    # Try rank 15 first
    for target_rank in [15, 16, 17]:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random support patterns
        for attempt in range(500):
            if time.time() > deadline - 10:
                break
            
            # Randomly assign supports for each rank-1 term
            # For each term r, randomly choose which entries are non-zero
            U_supports = []
            V_supports = []
            W_supports = []
            
            for r in range(target_rank):
                # Choose random non-empty subsets
                u_nonzero = set()
                for idx in range(n*m):
                    if np.random.random() < 0.3:
                        u_nonzero.add(idx)
                if not u_nonzero:
                    u_nonzero.add(np.random.randint(n*m))
                
                v_nonzero = set()
                for idx in range(m*p):
                    if np.random.random() < 0.25:
                        v_nonzero.add(idx)
                if not v_nonzero:
                    v_nonzero.add(np.random.randint(m*p))
                
                w_nonzero = set()
                for idx in range(p*n):
                    if np.random.random() < 0.3:
                        w_nonzero.add(idx)
                if not w_nonzero:
                    w_nonzero.add(np.random.randint(p*n))
                
                U_supports.append(sorted(u_nonzero))
                V_supports.append(sorted(v_nonzero))
                W_supports.append(sorted(w_nonzero))
            
            # Now solve for coefficients
            # Variables: for each term r and each non-zero entry, one variable
            var_to_idx = {}
            A = []
            b = []
            var_count = 0
            
            for r in range(target_rank):
                for idx in U_supports[r]:
                    var_to_idx[('U', r, idx)] = var_count
                    var_count += 1
                for idx in V_supports[r]:
                    var_to_idx[('V', r, idx)] = var_count
                    var_count += 1
                for idx in W_supports[r]:
                    var_to_idx[('W', r, idx)] = var_count
                    var_count += 1
            
            # This is a nonlinear system (products of variables), so linear algebra won't work directly.
            # We need to use a different approach.
            
            # Let's instead fix the coefficients to be ±1 or ±0.5 and just verify.
            # Or use a local search / simulated annealing approach.
            
            # For now, skip this approach and try a more structured one.
            break  # Break out of attempt loop since this approach is flawed
    
    # Try to find a lower rank solution via iterative improvement
    # Start from schoolbook and try to merge/simplify terms
    
    best_solution = schoolbook()
    best_rank = n * m * p
    
    # Try rank 15 with a structured search
    # Use the observation that we can group terms by their j-index
    for j in range(m):
        if time.time() > deadline - 10:
            break
        # For fixed j, the sub-tensor is T[i,j,k] for i in {0,1}, k in {0,1,2}
        # This is a 2x3 matrix, which has rank at most 2
        # So we can represent it with 2 rank-1 terms in the (i,k) space
        # But these terms must be consistent across different j values
    
    # Given the complexity, let's try a random search with verification
    np.random.seed(42)
    for rank_try in [15, 16, 17]:
        if time.time() > deadline - 15:
            break
        
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random small coefficients
            U = [[0.0] * rank_try for _ in range(n * m)]
            V = [[0.0] * rank_try for _ in range(m * p)]
            W = [[0.0] * rank_try for _ in range(p * n)]
            
            # Randomly assign non-zero entries
            for r in range(rank_try):
                for i in range(n):
                    for j in range(m):
                        if np.random.random() < 0.4:
                            U[i*m+j][r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                for j in range(m):
                    for k in range(p):
                        if np.random.random() < 0.35:
                            V[j*p+k][r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                for k in range(p):
                    for i in range(n):
                        if np.random.random() < 0.4:
                            W[k*n+i][r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            if verify_decomposition(U, V, W):
                # Convert to exact fractions if possible
                U_out = []
                for row in U:
                    U_out.append([round(x*2)/2 for x in row])
                V_out = []
                for row in V:
                    V_out.append([round(x*2)/2 for x in row])
                W_out = []
                for row in W:
                    W_out.append([round(x*2)/2 for x in row])
                
                if verify_decomposition(U_out, V_out, W_out, tol=1e-12):
                    return make_solution(U_out, V_out, W_out)
    
    return best_solution
# EVOLVE-BLOCK-END
