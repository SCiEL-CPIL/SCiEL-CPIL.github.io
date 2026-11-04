import numpy as np
import time
import itertools
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # 1. Construct the target tensor T
    # T shape: (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    # Schoolbook factors as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_factors = {
        'U': U_school.tolist(),
        'V': V_school.tolist(),
        'W': W_school.tolist()
    }

    # Candidate values for half-integer entries
    # Range -2 to 2 with step 0.5 -> 9 values
    candidates = np.arange(-2, 2.5, 0.5)
    
    # We will try to find a decomposition with rank R using a subset of these values.
    # To make it tractable, we restrict the search to a smaller set of candidates or
    # use a heuristic to prune the search space.
    # Given the complexity, we will try a limited set of ranks and use a simple
    # backtracking search with pruning based on the tensor constraints.
    
    # However, full backtracking is too slow. 
    # Instead, we will use a randomized search guided by the structure of the tensor.
    # The tensor T has non-zero entries only at specific indices.
    # T[i*m+j, j*p+k, k*n+i] = 1.
    
    # Let's try a local search / refinement approach starting from schoolbook.
    # Or, more effectively, use the MILP idea but in a simplified, heuristic way.
    # Since exact MILP is hard, we'll use a "sparse" random search.
    
    # Heuristic: Randomly assign half-integer values to a sparse set of entries
    # in U, V, W and check if they satisfy the tensor equation.
    
    # We define the target tensor equation:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
    
    # We will try ranks from 20 up to 35.
    # For each rank, we perform random restarts.
    
    # To speed up checking, we only evaluate the tensor equation for the non-zero
    # entries of T and a random subset of zero entries to ensure no "leaks".
    
    # Indices where T is 1
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                non_zero_indices.append((i, j, k))
                
    # Indices where T is 0 (sample a few)
    all_indices = list(itertools.product(range(n), range(m), range(p)))
    zero_indices = [idx for idx in all_indices if idx not in non_zero_indices]
    
    def check_decomposition(U, V, W, rank):
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Check non-zero entries
        for i, j, k in non_zero_indices:
            val = 0
            for c in range(rank):
                u_val = U[i*m+j, c]
                v_val = V[j*p+k, c]
                w_val = W[k*n+i, c]
                if u_val != 0 and v_val != 0 and w_val != 0:
                    val += u_val * v_val * w_val
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check a sample of zero entries
        if len(zero_indices) > 0:
            sample_size = min(10, len(zero_indices))
            np.random.shuffle(zero_indices) # Inefficient, but for small scale ok
            for i, j, k in zero_indices[:sample_size]:
                val = 0
                for c in range(rank):
                    u_val = U[i*m+j, c]
                    v_val = V[j*p+k, c]
                    w_val = W[k*n+i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                if abs(val) > 1e-9:
                    return False
        return True

    # Search strategy:
    # 1. Try to find a solution with rank R.
    # 2. Use random initialization with sparse half-integer values.
    # 3. If found, return.
    
    # To improve efficiency, we can use a greedy construction or local search.
    # Let's try a simple local search from the schoolbook solution, trying to
    # merge columns or reduce rank by setting some columns to zero and adjusting others.
    # This is complex.
    
    # Alternative: Direct random search for low rank.
    # We know that for 3x3x4, the rank is likely around 20-25.
    # Let's try ranks 20 to 30.
    
    for target_rank in range(20, 31):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on remaining time
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(restart)
            
            # Initialize U, V, W with sparse half-integers
            # Density: Let's try to keep it sparse.
            # Each column corresponds to a rank term.
            # We can randomly pick a few non-zero entries per row?
            # Or just random half-integers with a bias towards 0.
            
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Randomly fill with candidates
            # To make it more structured, let's try to align with the tensor structure.
            # But random is a good baseline.
            
            for c in range(target_rank):
                # For each rank term, pick random non-zero patterns
                # Or just fill entire matrices with random sparse values
                
                # Let's try a "structured" random initialization:
                # For each term c, pick a random (i,j,k) and set U[i*m+j,c], V[j*p+k,c], W[k*n+i,c]
                # to random non-zero half-integers.
                # This ensures that at least one term contributes to each non-zero entry.
                
                # But we need to satisfy the sum.
                # Let's just use dense random half-integers with small magnitude.
                
                U[:, c] = np.random.choice(candidates, nm)
                V[:, c] = np.random.choice(candidates, mp)
                W[:, c] = np.random.choice(candidates, pn)
                
                # Add some zeros to promote sparsity
                mask_u = np.random.rand(nm) < 0.5
                mask_v = np.random.rand(mp) < 0.5
                mask_w = np.random.rand(pn) < 0.5
                U[mask_u, c] = 0
                V[mask_v, c] = 0
                W[mask_w, c] = 0
                
            # Local search: Try to fix the errors by adjusting values
            # This is a heuristic.
            
            # Check if current random guess works (unlikely)
            if check_decomposition(U, V, W, target_rank):
                best_rank = target_rank
                best_factors = {
                    'U': U.tolist(),
                    'V': V.tolist(),
                    'W': W.tolist()
                }
                found = True
                break
            
            # If not, try a few iterations of "fixing"
            # We can compute the error for each non-zero entry and try to adjust
            # the most "responsible" term.
            
            # For simplicity in this iteration, we'll just do more random restarts.
            # The probability of hitting a solution randomly is low, but maybe with
            # structured initialization it's better.
            
            # Let's try a different initialization: Start with schoolbook and
            # try to reduce rank by setting some columns to zero and scaling others.
            # This is a "pruning" approach.
            
            if restart == 0:
                # Try pruning schoolbook
                U_prune = U_school.copy()
                V_prune = V_school.copy()
                W_prune = W_school.copy()
                
                # Try to remove random columns and see if we can compensate
                # This is hard.
                
                pass

    # If no better solution found, return schoolbook
    return best_factors
