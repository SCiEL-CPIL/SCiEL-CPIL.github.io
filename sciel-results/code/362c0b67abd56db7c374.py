"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute lower bounds
    T0 = tensor.reshape(n*m, -1)
    rank0 = np.linalg.matrix_rank(T0)
    T1 = tensor.transpose(1,0,2).reshape(m*p, -1)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = tensor.transpose(2,0,1).reshape(p*n, -1)
    rank2 = np.linalg.matrix_rank(T2)
    lower_bound = max(rank0, rank1, rank2)
    
    # Trivial solution with rank = n*m*p
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    best_rank = rank_trivial
    
    if lower_bound >= rank_trivial:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    found_better = False
    
    # Precompute useful matrices
    # For each a (row index of U), T_a is the slice tensor[:, a, :] of shape (m*p, p*n)
    # T_a[b, c] = 1 if b = j*p+k and c = k*n+i where a = i*m+j
    
    def try_als(U_fixed, trial_rank, max_iter=200, tol=1e-10):
        """Run ALS to find V and W given U_fixed. Returns (V, W, error) or None."""
        np.random.seed(42)
        V_curr = np.random.randn(m*p, trial_rank)
        W_curr = np.random.randn(p*n, trial_rank)
        
        UtU = U_fixed.T @ U_fixed
        
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                return None
            
            # Update W
            VtV = V_curr.T @ V_curr
            G_W = UtU * VtV
            
            # rhs_W[c, r] = sum_{a,b} T[a,b,c] U[a,r] V[b,r]
            # = einsum('abc,ar,br->cr', tensor, U_fixed, V_curr)
            rhs_W = np.einsum('abc,ar,br->cr', tensor, U_fixed, V_curr)
            
            reg = 1e-10 * np.eye(trial_rank)
            try:
                W_curr = np.linalg.solve(G_W + reg, rhs_W.T).T
            except np.linalg.LinAlgError:
                W_curr = np.linalg.lstsq(G_W + reg, rhs_W.T, rcond=None)[0].T
            
            # Update V
            WtW = W_curr.T @ W_curr
            G_V = UtU * WtW
            
            rhs_V = np.einsum('abc,ar,cr->br', tensor, U_fixed, W_curr)
            
            try:
                V_curr = np.linalg.solve(G_V + reg, rhs_V.T).T
            except np.linalg.LinAlgError:
                V_curr = np.linalg.lstsq(G_V + reg, rhs_V.T, rcond=None)[0].T
            
            # Check convergence
            recon = np.einsum('ar,br,cr->abc', U_fixed, V_curr, W_curr)
            error = np.linalg.norm(recon - tensor)
            if error < tol:
                return V_curr, W_curr, error
        
        return None
    
    def try_round_and_verify(U_fixed, V_curr, W_curr, trial_rank):
        """Try to round V and W to rational numbers and verify exact decomposition."""
        for factor in [1, 2, 3, 4, 6, 8, 12]:
            V_r = np.round(V_curr * factor) / factor
            W_r = np.round(W_curr * factor) / factor
            
            recon_r = np.einsum('ar,br,cr->abc', U_fixed, V_r, W_r)
            error_r = np.linalg.norm(recon_r - tensor)
            
            if error_r < 1e-8:
                # Verify with exact arithmetic check
                recon_exact = np.einsum('ar,br,cr->abc', U_fixed, V_r, W_r)
                if np.allclose(recon_exact, tensor, atol=1e-10):
                    return U_fixed.copy(), V_r.copy(), W_r.copy()
        
        return None
    
    # Strategy 1: Structured U patterns based on the tensor structure
    # For matrix multiplication, we can try U patterns where each row (i,j) 
    # has non-zeros in columns corresponding to specific k values
    
    # Try ranks from lower_bound to rank_trivial - 1
    for trial_rank in range(lower_bound, rank_trivial):
        if time.time() > deadline - 40:
            break
        
        # Generate structured U patterns
        # Pattern 1: U[i*m+j, r] = 1 if r corresponds to (i,j,k) for some k
        # This is basically the trivial pattern but with fewer columns
        
        # Pattern 2: Use a subset of the trivial pattern
        # Select trial_rank out of n*m*p possible (i,j,k) triples
        
        np.random.seed(1000 + trial_rank)
        
        # Try multiple random subsets of the trivial pattern
        num_attempts = min(30, max(5, int((deadline - 40 - time.time()) / 2)))
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 40:
                break
            
            # Create U by selecting trial_rank random (i,j,k) triples
            all_triples = list(product(range(n), range(m), range(p)))
            selected = np.random.choice(len(all_triples), size=trial_rank, replace=False)
            
            U_fixed = np.zeros((n*m, trial_rank))
            for r, idx in enumerate(selected):
                i, j, k = all_triples[idx]
                U_fixed[i*m+j, r] = 1
            
            # Check if U has full row rank (each row should have at least one non-zero)
            if np.any(np.all(U_fixed == 0, axis=1)):
                continue
            
            result = try_als(U_fixed, trial_rank)
            if result is not None:
                V_curr, W_curr, error = result
                rounded = try_round_and_verify(U_fixed, V_curr, W_curr, trial_rank)
                if rounded is not None:
                    best_U, best_V, best_W = rounded
                    best_rank = trial_rank
                    found_better = True
                    break
        
        if found_better:
            break
    
    # Strategy 2: If no better solution found, try random sparse U with ALS
    if not found_better:
        for trial_rank in range(max(lower_bound, best_rank - 2), best_rank):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(2000 + trial_rank)
            num_attempts = min(20, max(3, int((deadline - 30 - time.time()) / 2)))
            
            for attempt in range(num_attempts):
                if time.time() > deadline - 30:
                    break
                
                U_fixed = np.zeros((n*m, trial_rank))
                for a in range(n*m):
                    num_nz = np.random.randint(1, 3)
                    cols = np.random.choice(trial_rank, size=num_nz, replace=False)
                    for c in cols:
                        U_fixed[a, c] = np.random.choice([-1, 1])
                
                result = try_als(U_fixed, trial_rank)
                if result is not None:
                    V_curr, W_curr, error = result
                    rounded = try_round_and_verify(U_fixed, V_curr, W_curr, trial_rank)
                    if rounded is not None:
                        best_U, best_V, best_W = rounded
                        best_rank = trial_rank
                        found_better = True
                        break
            
            if found_better:
                break
    
    # Strategy 3: Algebraic approach using the structure of matrix multiplication
    # For 2x3 * 3x7, we can exploit the fact that n=2 is small
    # Try to find a decomposition with rank less than trivial by using the algebraic structure
    
    if not found_better and time.time() < deadline - 25:
        # For matrix multiplication, the tensor has a specific structure
        # We can try to use the fact that the middle dimension is shared
        
        # Try ranks that are divisors or related to the dimensions
        candidate_ranks = []
        for r in range(lower_bound, best_rank):
            if r % 1 == 0:  # All integers
                candidate_ranks.append(r)
        
        # Sort by likelihood (closer to lower_bound first)
        candidate_ranks.sort()
        
        for trial_rank in candidate_ranks:
            if time.time() > deadline - 25:
                break
            
            # Try structured U based on the algebraic variety
            # For 2x3 * 3x7, we can try U patterns where:
            # - Row (0,j) connects to certain k values
            # - Row (1,j) connects to certain k values
            
            np.random.seed(3000 + trial_rank)
            num_attempts = min(15, max(3, int((deadline - 25 - time.time()) / 3)))
            
            for attempt in range(num_attempts):
                if time.time() > deadline - 25:
                    break
                
                # Create U with structure: each row has non-zeros in a pattern
                U_fixed = np.zeros((n*m, trial_rank))
                
                # For each row (i,j), choose a subset of columns
                for i in range(n):
                    for j in range(m):
                        a = i*m + j
                        # Choose 1-2 columns for this row
                        num_nz = np.random.randint(1, 3)
                        cols = np.random.choice(trial_rank, size=num_nz, replace=False)
                        for c in cols:
                            U_fixed[a, c] = np.random.choice([-1, 0, 1])
                
                # Check if U has full row rank
                if np.any(np.all(U_fixed == 0, axis=1)):
                    continue
                
                result = try_als(U_fixed, trial_rank)
                if result is not None:
                    V_curr, W_curr, error = result
                    rounded = try_round_and_verify(U_fixed, V_curr, W_curr, trial_rank)
                    if rounded is not None:
                        best_U, best_V, best_W = rounded
                        best_rank = trial_rank
                        found_better = True
                        break
            
            if found_better:
                break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
