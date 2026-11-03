import time
import numpy as np
import random
import sys
from itertools import product

# Try to import scipy for optimization
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute valid indices (a, b, c) corresponding to (i*m+j, j*p+k, k*n+i)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.append((i*m+j, j*p+k, k*n+i))
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for a, b, c in valid_indices:
        T[a, b, c] = 1.0

    def verify(U, V, W):
        """
        Verify if U, V, W exactly decompose the tensor.
        """
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # Check valid entries
        for a, b, c in valid_indices:
            if abs(T_rec[a, b, c] - 1.0) > 1e-9:
                return False
        # Check invalid entries
        mask = np.zeros((nm, mp, pn), dtype=bool)
        for a, b, c in valid_indices:
            mask[a, b, c] = True
        invalid_vals = T_rec[~mask]
        if np.any(np.abs(invalid_vals) > 1e-9):
            return False
        return True

    def to_half_integers(U, V, W):
        """
        Snap to nearest half-integers and verify.
        """
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if verify(U2, V2, W2):
            return U2, V2, W2
        return None

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((nm, rank_fb))
    V_fb = np.zeros((mp, rank_fb))
    W_fb = np.zeros((pn, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
                
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb
    
    deadline = time.time() + 110  # Safety margin

    # Structural Change: Systematic Grid Search on Half-Integers
    # For small ranks, the space of half-integers in [-1, 1] or [-2, 2] is manageable.
    # We prioritize smaller ranks and smaller value ranges.
    
    # Define value sets to search
    # Start with small values, expand if needed
    val_sets = [
        [0.0, 0.5, 1.0, -0.5, -1.0],
        [0.0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5],
        [0.0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5, 2.0, -2.0]
    ]
    
    # Theoretical lower bound for rank is max(nm, mp, pn) = 12
    # However, Strassen-like improvements might exist.
    # For 2x3x4, the known optimal rank is likely around 12-14.
    # Let's search ranks from 12 up to best_rank - 1.
    
    min_rank_to_check = max(nm, mp, pn)
    
    # We will iterate ranks from min_rank_to_check to best_rank - 1
    # For each rank, we try different value sets.
    # To avoid combinatorial explosion, we limit the search depth.
    # Total variables = (nm + mp + pn) * rank = (6 + 12 + 8) * rank = 26 * rank.
    # For rank 12, that's 312 variables. 
    # Grid search is impossible for 312 variables even with 5 values.
    # We need a smarter search or a different structural approach.
    
    # Alternative Structural Approach: 
    # Use the "Strassen" style construction or known decompositions.
    # Or, use a local search / gradient descent that is more robust than the previous one.
    
    # Let's try a different optimization strategy: 
    # 1. Start with the schoolbook solution (rank 24).
    # 2. Try to reduce rank by 1 at a time by "merging" or "eliminating" terms?
    #    This is hard.
    
    # Better: Use the fact that the tensor is small. 
    # Let's use a stochastic local search / simulated annealing on the discrete half-integer space.
    # Or, use a continuous optimization that is more robust (e.g., Powell or Nelder-Mead on subsets).
    
    # Given the previous failure of L-BFGS-B to find low rank solutions (stuck in local minima or slow),
    # let's try a "divide and conquer" or "coordinate descent" approach on the discrete grid.
    
    # However, 312 variables is too many for pure grid search.
    
    # Let's try to exploit the structure. 
    # The matrix multiplication tensor has a specific sparsity.
    
    # Let's try a different initialization for the continuous optimization:
    # Instead of random uniform, try to initialize with a known low-rank approximation 
    # or use a "greedy" construction.
    
    # Actually, the most promising structural change for this specific small tensor 
    # is to use a **Genetic Algorithm** or **Evolutionary Strategy** on the discrete half-integer values.
    # This explores the discrete space directly, avoiding the "snap" failure mode.
    
    def discrete_evo_search(rank, val_set, generations=100, pop_size=50):
        """
        Simple evolutionary search for half-integer entries.
        """
        dim_U = nm * rank
        dim_V = mp * rank
        dim_W = pn * rank
        total_dim = dim_U + dim_V + dim_W
        
        # Encode: indices into val_set
        def decode(genome):
            U = np.array([val_set[g] for g in genome[:dim_U]]).reshape(nm, rank)
            V = np.array([val_set[g] for g in genome[dim_U:dim_U+dim_V]]).reshape(mp, rank)
            W = np.array([val_set[g] for g in genome[dim_U+dim_V:]]).reshape(pn, rank)
            return U, V, W
        
        def fitness(genome):
            U, V, W = decode(genome)
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = 0.0
            for a, b, c in valid_indices:
                error += (T_rec[a, b, c] - 1.0)**2
            mask = np.zeros((nm, mp, pn), dtype=bool)
            for a, b, c in valid_indices:
                mask[a, b, c] = True
            invalid_vals = T_rec[~mask]
            error += np.sum(invalid_vals**2)
            return error
        
        # Initialize population
        val_len = len(val_set)
        population = [np.random.randint(0, val_len, size=total_dim) for _ in range(pop_size)]
        
        best_genome = None
        best_fit = float('inf')
        
        for gen in range(generations):
            if time.time() > deadline:
                break
            
            # Evaluate
            fits = [fitness(g) for g in population]
            
            for i in range(pop_size):
                if fits[i] < best_fit:
                    best_fit = fits[i]
                    best_genome = population[i].copy()
                    
                    if best_fit < 1e-9:
                        U, V, W = decode(best_genome)
                        if verify(U, V, W):
                            return U, V, W
            
            # Selection: keep top half
            indices = np.argsort(fits)
            parents = [population[i].copy() for i in indices[:pop_size//2]]
            
            # Crossover and Mutation
            new_population = parents.copy()
            while len(new_population) < pop_size:
                p1, p2 = random.choice(parents), random.choice(parents)
                # Uniform crossover
                mask = np.random.randint(0, 2, size=total_dim)
                child = np.where(mask == 1, p1, p2)
                
                # Mutation: flip a few genes
                num_mut = random.randint(1, 5)
                mut_indices = np.random.choice(total_dim, size=num_mut, replace=False)
                child[mut_indices] = np.random.randint(0, val_len, size=num_mut)
                
                new_population.append(child)
            
            population = new_population
        
        return None

    # Try evolutionary search for low ranks
    # Start with rank 12, then 13, etc.
    # Use small value sets first
    
    for rank in range(min_rank_to_check, best_rank):
        if time.time() > deadline:
            break
        
        # Try different value sets for this rank
        # Smaller value sets are easier to search
        for val_set in val_sets:
            if time.time() > deadline:
                break
            
            # Limit generations/pop size based on remaining time
            # Rough estimate: fitness eval is ~O(nm*mp*pn) = 288 ops.
            # Pop 50, Gen 100 => 5000 evals => ~1.4M ops. Fast.
            
            result = discrete_evo_search(rank, val_set, generations=200, pop_size=100)
            if result is not None:
                U_s, V_s, W_s = result
                best_result = (U_s, V_s, W_s)
                best_rank = rank
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        # If evo search fails, try continuous optimization with better init?
        # Maybe skip for now to keep it simple and fast.

    U_out, V_out, W_out = best_result
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
