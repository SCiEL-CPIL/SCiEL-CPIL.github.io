"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    def _verify(U, V, W):
        """Verify decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Reconstruct tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k])
                    # Wait, that's not right. The trilinear form is:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = 0.0
                    for r in range(U_arr.shape[1]):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _verify_fast(U, V, W):
        """Faster verification using einsum"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # We need: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a trilinear form. Let's check all 24 entries.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _snap_to_half_int(arr):
        """Snap values to nearest half-integer"""
        return np.round(2 * arr) / 2.0

    def _try_random_decomposition(rank, n_tries=500):
        """Try to find a rank-r decomposition via random search with local refinement"""
        best_result = None
        for attempt in range(n_tries):
            if time.time() - start_time > time_limit:
                break

            # Initialize with random half-integer values
            # Use a structured initialization: start from schoolbook and try to merge
            # Or use random initialization
            
            # Strategy: Use a combination of structured and random
            # Initialize U, V, W with small random half-integers
            scale = 1.0
            U_rand = np.random.choice([-1, 0, 0.5, 1, 1.5, 2], size=(n*m, rank))
            V_rand = np.random.choice([-1, 0, 0.5, 1, 1.5, 2], size=(m*p, rank))
            W_rand = np.random.choice([-1, 0, 0.5, 1, 1.5, 2], size=(p*n, rank))

            # Local search: try to adjust entries to satisfy constraints
            # This is a hard problem, so use a heuristic approach
            
            # Alternative: Use the fact that we can fix some variables and solve for others
            # For now, try a simple approach: check if the random init works (unlikely)
            if _verify_fast(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                return U_rand.tolist(), V_rand.tolist(), W_rand.tolist()

        return None

    def _try_structured_decomposition(rank):
        """Try structured decompositions based on known patterns"""
        # For rank 12 (the lower bound), try to find a solution
        # One approach: use the fact that the tensor is all-ones
        
        # Try: set W to have specific structure
        # If W[k*n+i, r] = a_r for all k,i (constant across k and i for each r)
        # Then we need sum_r a_r * U[i*m+j, r] * V[j*p+k, r] = 1
        
        # This is still complex. Let's try a different structured approach.
        
        # Approach: Use permutation matrices
        # For each r, define a permutation that maps (i,j,k) to specific indices
        
        # Actually, let's try a more systematic approach:
        # The matrix multiplication tensor can be decomposed using
        # rank-one tensors of the form: u_i * v_j * w_k
        
        # For the 2x3x4 case, let's try to find rank-r decompositions
        # by solving a system of equations
        
        if rank < 12:
            return None  # Below lower bound
        
        # Try to construct a solution
        # One known approach: use the fact that for small matrices,
        # the rank can be found via algebraic methods
        
        # For now, try a greedy approach:
        # Start with rank 12 and see if we can find a solution
        
        # Initialize with zeros
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Try to fill in entries systematically
        # This is a research-level problem, so use a heuristic
        
        # Heuristic: try to make each "column" (rank dimension) contribute
        # to multiple (i,j,k) entries
        
        # For each rank dimension r, choose a pattern
        # that covers multiple (i,j,k) triples
        
        # A simple pattern: for each r, pick one (i,j,k) and set
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        # This gives rank 24 (schoolbook)
        
        # To reduce rank, we need each column to cover multiple (i,j,k)
        # This requires U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # for multiple (i,j,k) with the same r
        
        # This means we need U[i*m+j, r] = V[j*p+k, r] = W[k*n+i, r] = ±1
        # for all (i,j,k) in some set, which is only possible if
        # the set has a specific structure
        
        # For now, return None and let the random search try
        
        return None

    # Try to find lower-rank decompositions
    # Start from rank 12 (lower bound) and go up
    
    best_rank = rank_schoolbook
    best_result = (U_sb, V_sb, W_sb)
    
    # Try ranks from 12 to rank_schoolbook - 1
    for r in range(12, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        # Try structured decomposition
        result = _try_structured_decomposition(r)
        if result is not None:
            best_rank = r
            best_result = result
            break  # Found a solution at this rank, stop
        
        # Try random search
        result = _try_random_decomposition(r, n_tries=100)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If we found a better solution, return it
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        # Ensure they are half-integers
        U = [[float(x) for x in row] for row in U]
        V = [[float(x) for x in row] for row in V]
        W = [[float(x) for x in row] for row in W]
        return {"U": U, "V": V, "W": W}
    
    # Otherwise return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
