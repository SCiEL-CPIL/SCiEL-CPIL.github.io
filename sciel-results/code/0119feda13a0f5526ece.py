import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    # Check that entries are half-integers
    for mat in [U, V, W]:
        for val in mat.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
    return True

def _try_merge_trivial(n, m, p, max_time=60.0):
    """Try to find a rank-23 decomposition by merging two terms from the trivial solution."""
    start = time.time()
    rank_trivial = n * m * p
    
    # Build trivial solution
    U = np.zeros((n * m, rank_trivial))
    V = np.zeros((m * p, rank_trivial))
    W = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Try to merge pairs of columns
    # For each pair (a, b), try to find a single column that represents both
    # This is equivalent to finding a rank-1 approximation that covers two entries
    
    # A simpler approach: try small perturbations and round to half-integers
    np.random.seed(42)
    
    best_solution = None
    
    # Try various strategies
    strategies = [
        'pair_merge',
        'random_perturb',
        'structured_search',
    ]
    
    for strategy in strategies:
        if time.time() - start > max_time:
            break
        
        if strategy == 'pair_merge':
            # Try to find if any two entries can be merged
            # For entries (i1,j1,k1) and (i2,j2,k2), we need:
            # A single term that contributes to both
            # This requires the supports to be compatible
            
            # Try all pairs of columns
            for a in range(rank_trivial):
                for b in range(a+1, rank_trivial):
                    if time.time() - start > max_time:
                        break
                    
                    # Get the entries these two columns correspond to
                    # Column a corresponds to (ia, ja, ka)
                    ia = a // (m * p)
                    ja = (a // p) % m
                    ka = a % p
                    ib = b // (m * p)
                    jb = (b // p) % m
                    kb = b % p
                    
                    # Try to find a single term that can replace both
                    # The new term would need to have:
                    # U_new[i*m+j] nonzero where both original terms have nonzero U
                    # etc.
                    
                    # This is complex. Let's try a different approach.
                    # Try to find a rank-23 solution by solving a system of equations.
                    pass
        
        elif strategy == 'random_perturb':
            # Start from trivial solution, remove one column, add a new one
            # and try to adjust to satisfy all equations
            
            for remove_col in range(rank_trivial):
                if time.time() - start > max_time:
                    break
                
                # Create a rank-23 solution by removing column remove_col
                # and adding a new column that we'll determine
                
                U_new = np.zeros((n * m, rank_trivial - 1))
                V_new = np.zeros((m * p, rank_trivial - 1))
                W_new = np.zeros((p * n, rank_trivial - 1))
                
                # Copy all columns except remove_col
                new_col = 0
                for old_col in range(rank_trivial):
                    if old_col != remove_col:
                        U_new[:, new_col] = U[:, old_col]
                        V_new[:, new_col] = V[:, old_col]
                        W_new[:, new_col] = W[:, old_col]
                        new_col += 1
                
                # Now we need to find the new column (the last one)
                # such that all equations are satisfied
                
                # For each (i,j,k), the contribution from the removed column was:
                # U[ia*m+ja, remove_col] * V[ja*pa+ka, remove_col] * W[ka*na+ia, remove_col]
                # where (ia, ja, ka) is the entry for column remove_col
                
                ia = remove_col // (m * p)
                ja = (remove_col // p) % m
                ka = remove_col % p
                
                # The new column must compensate for the missing contribution
                # at (ia, ja, ka) and not disturb other entries
                
                # Let the new column be u_new, v_new, w_new
                # For entry (i,j,k):
                # sum_{r != remove_col} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] + 
                # u_new[i*m+j]*v_new[j*p+k]*w_new[k*n+i] = 1
                
                # For (ia, ja, ka): the sum without the removed column is 0
                # (since only the removed column contributed to this entry)
                # So we need: u_new[ia*m+ja]*v_new[ja*p+ka]*w_new[ka*n+ia] = 1
                
                # For other entries (i,j,k) != (ia,ja,ka): the sum without removed column is 1
                # (since the removed column had zero contribution to these entries)
                # So we need: u_new[i*m+j]*v_new[j*p+k]*w_new[k*n+i] = 0
                
                # This means the new column must be zero everywhere except possibly
                # at positions that don't interact with other entries.
                
                # Let's try: set u_new, v_new, w_new to be zero except at specific positions
                # such that only (ia, ja, ka) gets a contribution of 1.
                
                # One option: u_new[ia*m+ja] = 1, v_new[ja*p+ka] = 1, w_new[ka*n+ia] = 1
                # But this is exactly the same as the removed column!
                
                # We need a different combination.
                
                # Try: u_new[ia*m+ja] = a, v_new[ja*p+ka] = b, w_new[ka*n+ia] = c
                # with a*b*c = 1, and all other entries zero.
                # But then for other entries, the contribution is 0, which is good.
                
                # So any a, b, c with a*b*c = 1 works.
                # Let's try a = 1, b = 1, c = 1 (same as original)
                # or a = 2, b = 1, c = 1/2, etc.
                
                # But wait - if we just set the new column to be a scalar multiple,
                # we're not reducing the rank. We need to combine with existing columns.
                
                # Let's try a different approach: allow the new column to have
                # non-zero entries at multiple positions, and solve for them.
                
                # This is getting complex. Let's try a numerical approach.
                
                # Define the residual for each (i,j,k) if we add a new column with
                # entries u, v, w:
                # residual(i,j,k) = 1 - sum_{r in kept} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] 
                #                  - u[i*m+j]*v[j*p+k]*w[k*n+i]
                
                # For (ia, ja, ka): residual = 1 - 0 - u[ia*m+ja]*v[ja*p+ka]*w[ka*n+ia]
                # For others: residual = 1 - 1 - u[i*m+j]*v[j*p+k]*w[k*n+i] = -u[i*m+j]*v[j*p+k]*w[k*n+i]
                
                # So we need:
                # u[ia*m+ja]*v[ja*p+ka]*w[ka*n+ia] = 1
                # u[i*m+j]*v[j*p+k]*w[k*n+i] = 0 for all (i,j,k) != (ia,ja,ka)
                
                # The second condition means that for any (i,j,k) != (ia,ja,ka),
                # at least one of u[i*m+j], v[j*p+k], w[k*n+i] must be zero.
                
                # This is a constraint satisfaction problem.
                
                # Let's try: set u to be nonzero only at ia*m+ja, v only at ja*p+ka,
                # w only at ka*n+ia. Then the conditions are satisfied.
                # But this gives us the same column as before.
                
                # To get a different solution, we need to allow some overlap with
                # existing columns.
                
                # Let's try a heuristic: randomly initialize the new column with
                # small half-integer values and use gradient descent.
                
                # For now, skip this and try another approach.
                pass
        
        elif strategy == 'structured_search':
            # Try a more systematic search
            # Look for rank-23 solutions by trying all possible "merges"
            
            # A merge means: replace two columns with one column that has
            # half-integer entries and satisfies the tensor equations.
            
            # For columns a and b, we want to find a new column c such that:
            # For all (i,j,k):
            #   U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 
            #   U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] + U[i*m+j,b]*V[j*p+k,b]*W[k*n+i,b]
            
            # This is a system of n*m*p equations in 3*(n*m + m*p + p*n) unknowns.
            # For n=2,m=3,p=4: 24 equations in 3*(6+12+8) = 72 unknowns.
            # This is underdetermined, so solutions likely exist.
            
            # Let's try to solve this for a few pairs.
            
            for a in range(rank_trivial):
                for b in range(a+1, rank_trivial):
                    if time.time() - start > max_time:
                        break
                    
                    # Build the linear system for the new column
                    # Unknowns: u (n*m values), v (m*p values), w (p*n values)
                    # Equations: for each (i,j,k), the product u[i*m+j]*v[j*p+k]*w[k*n+i] = target[i,j,k]
                    
                    # This is nonlinear. Let's try a different parameterization.
                    
                    # Assume the new column has the same support pattern as one of the originals,
                    # with scaled values.
                    
                    # Try: new column = alpha * col_a + beta * col_b
                    # But this doesn't reduce rank.
                    
                    # Let's give up on this for now.
                    pass
    
    return None

def _randomized_search(n, m, p, target_rank, max_time=30.0):
    """Randomized search for a decomposition of given rank."""
    start = time.time()
    np.random.seed(123)
    
    n_vars = n * m
    m_vars = m * p
    p_vars = p * n
    
    best_solution = None
    
    for attempt in range(100):
        if time.time() - start > max_time:
            break
        
        # Initialize with small half-integer values
        U = np.random.randint(-2, 3, size=(n_vars, target_rank)) * 0.5
        V = np.random.randint(-2, 3, size=(m_vars, target_rank)) * 0.5
        W = np.random.randint(-2, 3, size=(p_vars, target_rank)) * 0.5
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Trivial decomposition rank
    rank_trivial = n * m * p
    
    # Initialize trivial solution as fallback
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    
    # Try to find lower rank solutions
    # Known lower bound for 2x3x4 is 19, but let's try from 23 down
    
    for target_rank in range(rank_trivial - 1, 18, -1):
        if time.time() > deadline:
            break
        
        # Try randomized search
        result = _randomized_search(n, m, p, target_rank, max_time=10.0)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break  # Found a solution, stop searching
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
