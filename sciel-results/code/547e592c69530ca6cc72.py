"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook (rank 36) as fallback.
    2. Use structured random search with better initialization.
    3. Exploit the fact that we only need to verify 36 active triples + ensure
       all other combinations sum to 0.
    4. Use local search / hill climbing to refine candidates.
    5. Leverage the specific structure of the 3x3x4 case by searching for a 
       decomposition of rank 18 (the theoretical lower bound for some subspaces)
       or lower by representing the multiplication as a bilinear map over a finite field.
    6. NEW: Model as sparse linear regression with algebraic units, using greedy 
       term addition to minimize residual Frobenius norm.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s budget, leaving 10s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Active triples: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    # Precompute all triples that need to be checked
    all_triples = list(itertools.product(range(n * m), range(m * p), range(p * n)))
    active_set = set(active_triples)
    
    def verify(U, V, W, r):
        """Check if U, V, W with rank r satisfy all constraints."""
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            expected = 1 if (a, b, c) in active_set else 0
            if s != expected:
                return False
        return True
    
    def error_count(U, V, W, r):
        """Count how many triples are violated (for local search)."""
        errors = 0
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            expected = 1 if (a, b, c) in active_set else 0
            if s != expected:
                errors += 1
        return errors
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # NEW: Sparse linear regression approach with algebraic units
    # The idea: represent each rank-1 term as a tensor u ⊗ v ⊗ w where 
    # u, v, w have entries from a small set of algebraic units.
    # Greedily add terms that best reduce the residual.
    
    def greedy_sparse_regression(max_rank, time_budget):
        """
        Greedily construct a decomposition by adding rank-1 terms that 
        maximally reduce the residual Frobenius norm.
        """
        start_time = time.time()
        deadline = start_time + time_budget
        
        # Define the target tensor T
        T = np.zeros((n * m, m * p, p * n))
        for (a, b, c) in active_triples:
            T[a, b, c] = 1.0
        
        # Define algebraic units: small integers and half-integers
        # For efficiency, use a restricted set
        units = [-2, -1, -0.5, 0, 0.5, 1, 2]
        
        # Current residual
        residual = T.copy()
        
        # Store the decomposition
        U_list = []
        V_list = []
        W_list = []
        
        # Greedily add terms
        for rank_step in range(max_rank):
            if time.time() > deadline:
                break
            
            # For each possible rank-1 term, compute how much it reduces the residual
            # This is expensive, so we use a heuristic: try random candidates and keep the best
            
            best_reduction = 0
            best_u = None
            best_v = None
            best_w = None
            
            # Try a number of random candidates
            num_candidates = 500
            
            for _ in range(num_candidates):
                if time.time() > deadline:
                    break
                
                # Generate random sparse vectors with entries from units
                # Use a sparsity pattern that matches the structure
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                
                # Assign non-zero entries
                num_u = random.randint(1, min(4, n * m))
                idx_u = random.sample(range(n * m), num_u)
                for idx in idx_u:
                    u[idx] = random.choice([x for x in units if x != 0])
                
                num_v = random.randint(1, min(4, m * p))
                idx_v = random.sample(range(m * p), num_v)
                for idx in idx_v:
                    v[idx] = random.choice([x for x in units if x != 0])
                
                num_w = random.randint(1, min(4, p * n))
                idx_w = random.sample(range(p * n), num_w)
                for idx in idx_w:
                    w[idx] = random.choice([x for x in units if x != 0])
                
                # Compute the rank-1 tensor
                term = np.einsum('i,j,k->ijk', u, v, w)
                
                # Compute the reduction in residual norm
                new_residual = residual - term
                reduction = np.sum(residual**2) - np.sum(new_residual**2)
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_u = u.copy()
                    best_v = v.copy()
                    best_w = w.copy()
            
            if best_u is None:
                break
            
            # Check if adding this term makes the residual exactly zero (or close enough)
            residual -= np.einsum('i,j,k->ijk', best_u, best_v, best_w)
            
            # Check if we've found an exact solution
            if np.allclose(residual, 0, atol=1e-10):
                # Convert to integer/half-integer representation
                U = []
                V = []
                W = []
                for i in range(n * m):
                    U.append([float(best_u_list[i][ci]) for ci in range(len(best_u_list))])
                # Actually, we need to store all terms
                # Let's reconstruct properly
                
                # Convert numpy arrays to lists
                U = np.array(best_u_list).T.tolist()
                V = np.array(best_v_list).T.tolist()
                W = np.array(best_w_list).T.tolist()
                
                # Verify
                if verify(U, V, W, len(best_u_list)):
                    return {"U": U, "V": V, "W": W}
            
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
        
        # If we didn't find an exact solution, return None
        return None
    
    # Try the greedy sparse regression approach
    # This is computationally expensive, so we use a time budget
    time_for_greedy = 60  # 60 seconds for greedy approach
    
    result = greedy_sparse_regression(best_rank - 1, time_for_greedy)
    if result is not None:
        best_result = result
        best_rank = len(result["U"][0])
    
    # Try structured search for lower ranks
    # Known: for 3x3*3x4, we expect rank around 22-24
    # Try from rank 18 upward (optimistic) to best_rank-1
    
    def try_rank_structured(r, max_attempts=200):
        """Try to find a rank-r decomposition using structured random search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with structured patterns
            # Use a mix of strategies:
            # 1. Sparse random
            # 2. Based on Strassen-like patterns
            
            strategy = attempt % 3
            
            if strategy == 0:
                # Sparse random: mostly zeros with some ±1
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nonzero = random.randint(1, min(3, r))
                    cols = random.sample(range(r), num_nonzero)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Dense random with small values
                scale = 2
                U = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(n * m)]
                V = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(m * p)]
                W = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(p * n)]
                
            else:
                # Strassen-inspired: use block structure
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                # For each column, assign a pattern
                for ci in range(r):
                    # Random pattern with some structure
                    u_vals = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                    v_vals = [random.choice([-1, 0, 1]) for _ in range(m * p)]
                    w_vals = [random.choice([-1, 0, 1]) for _ in range(p * n)]
                    for a in range(n * m):
                        U[a][ci] = u_vals[a]
                    for b in range(m * p):
                        V[b][ci] = v_vals[b]
                    for c in range(p * n):
                        W[c][ci] = w_vals[c]
            
            # Quick check: verify active triples first
            active_ok = True
            for (a, b, c) in active_triples:
                s = sum(U[a][ci] * V[b][ci] * W[c][ci] for ci in range(r))
                if s != 1:
                    active_ok = False
                    break
            
            if not active_ok:
                continue
            
            # Full verification
            if verify(U, V, W, r):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find lower rank decompositions
    # Start from optimistic rank and go up
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        # Adaptive number of attempts based on remaining time
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # Spend more time on promising ranks
        if r <= 22:
            max_attempts = min(500, int(time_remaining * 10))
        elif r <= 26:
            max_attempts = min(300, int(time_remaining * 5))
        else:
            max_attempts = min(100, int(time_remaining * 2))
        
        result = try_rank_structured(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            # Continue searching for even lower ranks? No, we want minimum, 
            # but we found one at rank r. Let's try lower ranks too.
            # Actually, we're going up from 18, so if we find rank r, 
            # we should check if there's something lower. But we already tried lower.
            # So we can stop here.
            break
    
    return best_result
# EVOLVE-BLOCK-END
