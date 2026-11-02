"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s / 150s limit
    n, m, p = DIMENSIONS

    def build_strassen_35():
        rank = 35
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for block in range(5):
            k1 = 2 * block
            k2 = 2 * block + 1
            r = 7 * block + 0
            U[0][r] += 1; U[3][r] += 1
            V[k1][r] += 1; V[10 + k2][r] += 1
            W[2 * k1][r] += 1; W[2 * k2 + 1][r] += 1
            r = 7 * block + 1
            U[2][r] += 1; U[3][r] += 1
            V[k1][r] += 1
            W[2 * k1 + 1][r] += 1; W[2 * k2 + 1][r] -= 1
            r = 7 * block + 2
            U[0][r] += 1
            V[k2][r] += 1; V[10 + k2][r] -= 1
            W[2 * k2][r] += 1; W[2 * k2 + 1][r] += 1
            r = 7 * block + 3
            U[3][r] += 1
            V[10 + k1][r] += 1; V[k1][r] -= 1
            W[2 * k1][r] += 1; W[2 * k1 + 1][r] += 1
            r = 7 * block + 4
            U[0][r] += 1; U[1][r] += 1
            V[10 + k2][r] += 1
            W[2 * k1][r] -= 1; W[2 * k2][r] += 1
            r = 7 * block + 5
            U[2][r] += 1; U[0][r] -= 1
            V[k1][r] += 1; V[k2][r] += 1
            W[2 * k2 + 1][r] += 1
            r = 7 * block + 6
            U[1][r] += 1; U[3][r] -= 1
            V[10 + k1][r] += 1; V[10 + k2][r] += 1
            W[2 * k1][r] += 1
        return U, V, W

    def verify(U, V, W):
        """Check that sum_r U[i,r]*V[j,r]*W[k,r] = delta(i,k) for all i in 0..3, j in 0..19, k in 0..19 with correct trace convention."""
        # U has shape (4, rank) indexed by (a,b) -> a*2+b
        # V has shape (20, rank) indexed by (b,d) -> b*10+d  
        # W has shape (20, rank) indexed by (c,d) -> c*10+d  (trace convention: W[k*n+i, r] where k is col of B, n is rows of A)
        # Actually per the task: W[k*n+i, r] where k is the column index of B (0..9), i is row of output (0..1)
        # So W index = k*2 + i where k is B-column, i is output row
        # The tensor entry T[a,b,c,d] = delta(a,c) should equal sum_r U[a*2+b, r] * V[b*10+d, r] * W[d*2+c, r]
        rank = len(U[0])
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(10):
                        expected = 1.0 if a == c else 0.0
                        actual = 0.0
                        ui = a * 2 + b
                        vj = b * 10 + d
                        wk = d * 2 + c
                        for r in range(rank):
                            actual += U[ui][r] * V[vj][r] * W[wk][r]
                        if abs(actual - expected) > 1e-9:
                            return False
        return True

    # Start with the known rank-35 solution
    U, V, W = build_strassen_35()
    
    # Verify our base solution is correct
    if not verify(U, V, W):
        # Fallback: return as-is
        return {"U": U, "V": V, "W": W}
    
    best_rank = 35
    best_U, best_V, best_W = U, V, W

    # Try to reduce rank by zeroing out columns and checking if the remaining
    # decomposition still works. This is a greedy approach: try removing each
    # rank-1 term and see if we can compensate.
    
    # Alternative: try a structured search. The key observation is that in the
    # Strassen decomposition, the A-side vectors repeat across blocks. Perhaps
    # we can merge some terms.
    
    # Let's try: can we achieve rank 34 by finding a linear dependency among
    # the 35 rank-1 terms?
    import numpy as np
    
    # Build the tensor from the decomposition and check if any term is redundant
    # by checking if it can be expressed as a combination of others.
    
    # For each term r, check if removing it leaves a solvable system for the
    # remaining terms. This is expensive, so let's try a simpler heuristic:
    # check if the 35 vectors in the U, V, W spaces have any linear dependencies
    # that allow merging.
    
    # Simpler approach: try all pairs of terms to see if one can be absorbed
    # into another by adjusting coefficients.
    
    # Actually, let's try a more direct approach: use the fact that the tensor
    # T[a,b,c,d] = delta(a,c) has a special structure. For each (b,d), the
    # contribution to the (a,c) plane is delta(a,c). So for each (b,d), we need
    # a rank-2 decomposition of the 2x2 identity, which requires at least 2
    # terms unless we share across (b,d) pairs.
    
    # The lower bound: for each of the 20 (b,d) pairs, we need the (a,c) part
    # to give identity. If we use s terms total, each term contributes to at
    # most one (b,d) pair (since V and W are fixed per term). Wait, no - a
    # single term can contribute to multiple (b,d) pairs if V and W have
    # support on multiple (b,d) entries.
    
    # Let's just try to see if rank 34 is possible by a search.
    # Strategy: take the rank-35 solution and try to set one column to zero,
    # then solve for adjustments in the remaining 34 columns.
    
    # This is a linear system problem. For each output entry (a,b,c,d), we need:
    # sum_{r=0}^{33} U[a*2+b][r] * V[b*10+d][r] * W[d*2+c][r] = delta(a,c)
    # where we've removed column 34.
    
    # This is nonlinear in the entries, so let's try a different approach:
    # fix U and V, solve for W, or fix U and W, solve for V, etc.
    
    # Let's try: fix U and V from the rank-35 solution (with column 34 removed),
    # and see if we can find W that works.
    
    # Actually, the simplest improvement to try: can we reduce from 35 to 34
    # by finding that one of the 35 terms is actually a linear combination of
    # the others in the tensor sense?
    
    # Let's build the full tensor from the decomposition and check its rank
    # properties.
    
    # T[a,b,c,d] = sum_r U[a*2+b][r] * V[b*10+d][r] * W[d*2+c][r]
    
    # Reshape to a matrix: for each (b,d), we have a 2x2 matrix M_{bd}[a,c] = T[a,b,c,d]
    # The tensor rank is at least the max over all (b,d) of the rank of M_{bd},
    # which is 2 (since M_{bd} = I_2 for all b,d).
    # Also, the tensor rank is at least the sum over b of the rank of the
    # 2x(2*10) matrix T[:,b,:,:] reshaped... this is getting complex.
    
    # Let me just try a practical search: attempt to find a rank-34 solution
    # by perturbing the rank-35 solution and solving.
    
    # Try to find a rank-34 solution by exploiting the block structure.
    # The 5 blocks have identical U vectors but different V and W patterns.
    # We can try to merge one pair of terms across blocks.
    
    # Strategy: For each pair of blocks (i, j), try to see if we can
    # replace 2 terms (one from each block) with 1 term that works for both.
    
    # Actually, let's try a more direct approach: use optimization to find
    # a rank-34 solution by starting from the rank-35 solution and trying
    # to set one column to zero while adjusting the others.
    
    import numpy as np
    
    def try_rank_34():
        """Try to find a rank-34 solution by removing one term and solving."""
        # For each term r to remove, try to solve for the remaining 34 terms
        # This is a nonlinear problem, so we'll use a heuristic approach.
        
        # The tensor equation is:
        # T[a,b,c,d] = sum_{r=0}^{33} U[a*2+b][r] * V[b*10+d][r] * W[d*2+c][r] = delta(a,c)
        
        # Let's fix U and V from the rank-35 solution (with column r removed)
        # and try to solve for W. This gives a linear system.
        
        # For each (b,d), we have:
        # sum_{r=0}^{33} U[a*2+b][r] * V[b*10+d][r] * W[d*2+c][r] = delta(a,c)
        
        # This is linear in W if U and V are fixed.
        
        for r_remove in range(35):
            # Build the reduced U, V (remove column r_remove)
            U_red = []
            V_red = []
            for i in range(4):
                U_red.append([U[i][r] for r in range(35) if r != r_remove])
            for j in range(20):
                V_red.append([V[j][r] for r in range(35) if r != r_remove])
            
            # Now solve for W. For each (b,d), we need to find W[d*2+c][r] for c=0,1
            # such that the equation holds.
            
            # Let's set up the linear system. For fixed (b,d), the equation is:
            # sum_{r=0}^{33} U[a*2+b][r] * V[b*10+d][r] * W[d*2+c][r] = delta(a,c)
            
            # This is a system of 4 equations (for a=0,1 and c=0,1) in 68 unknowns
            # (34 terms * 2 values of c). But the structure is such that for each
            # (b,d), we only need W[d*2+c][r] for c=0,1, so 68 unknowns total
            # across all 20 (b,d) pairs.
            
            # Actually, let's think differently. For each (b,d), we have a 2x2 matrix
            # M_{bd}[a,c] = sum_r U[a*2+b][r] * V[b*10+d][r] * W[d*2+c][r]
            # and we need M_{bd} = I_2.
            
            # This is equivalent to: for each (b,d), the 2x2 matrix formed by
            # outer products of U[:,b] and V[:,d] weighted by W entries equals I.
            
            # Let's try to solve this as a least squares problem.
            
            # Build the linear system: for each (a,b,c,d), we have one equation.
            # Total: 2*2*2*10 = 80 equations.
            # Unknowns: W has 20*34 = 680 entries, but many are zero in the original.
            
            # Let's use a simpler approach: for each (b,d), solve for the 68 W values
            # that appear in that (b,d) slice. But W[d*2+c][r] appears in multiple
            # (b,d) slices for different b.
            
            # This is getting complex. Let's try a different strategy: use the fact
            # that the problem has a lot of structure and try random perturbations.
            
            pass
        
        return None
    
    # Try a different approach: use the known rank-34 decomposition for 2x2x8
    # and extend it to 2x2x10.
    
    # Actually, let's try to use the fact that for 2x2 matrix multiplication,
    # we can use the rank-7 Strassen algorithm, and for the 2x10 part, we can
    # use a different decomposition.
    
    # The key insight: the tensor T[a,b,c,d] = delta(a,c) can be decomposed as:
    # sum_{r=0}^{R-1} u_r[a,b] * v_r[b,d] * w_r[d,c]
    
    # For each (b,d), we need sum_r u_r[a,b] * v_r[b,d] * w_r[d,c] = delta(a,c)
    
    # This means that for each (b,d), the 2x2 matrix sum_r (u_r[:,b] * v_r[b,d]) * w_r[d,:]^T = I
    
    # Let's try to find a solution with R=34 by using optimization.
    
    # Use scipy.optimize to minimize the error.
    from scipy.optimize import minimize
    
    def tensor_error(params):
        """Compute the error of the tensor decomposition."""
        # params is a flat array of length 34 * (4 + 20 + 20) = 34 * 44 = 1496
        # First 34*4 = 136 entries for U (4 rows * 34 cols)
        # Next 34*20 = 680 entries for V (20 rows * 34 cols)
        # Last 34*20 = 680 entries for W (20 rows * 34 cols)
        
        U_mat = params[0:136].reshape(4, 34)
        V_mat = params[136:816].reshape(20, 34)
        W_mat = params[816:1496].reshape(20, 34)
        
        error = 0.0
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(10):
                        expected = 1.0 if a == c else 0.0
                        actual = 0.0
                        ui = a * 2 + b
                        vj = b * 10 + d
                        wk = d * 2 + c
                        for r in range(34):
                            actual += U_mat[ui, r] * V_mat[vj, r] * W_mat[wk, r]
                        error += (actual - expected) ** 2
        return error
    
    # Start from the rank-35 solution with one column removed (try each column)
    best_error = float('inf')
    best_params = None
    
    for r_remove in range(35):
        # Build initial params from the rank-35 solution with column r_remove removed
        U_init = []
        V_init = []
        W_init = []
        for i in range(4):
            U_init.extend([U[i][r] for r in range(35) if r != r_remove])
        for j in range(20):
            V_init.extend([V[j][r] for r in range(35) if r != r_remove])
        for k in range(20):
            W_init.extend([W[k][r] for r in range(35) if r != r_remove])
        
        params_init = np.array(U_init + V_init + W_init, dtype=float)
        
        # Check if the initial solution already works (unlikely but possible)
        err_init = tensor_error(params_init)
        if err_init < 1e-9:
            return {"U": [U_init[i*34:(i+1)*34] for i in range(4)],
                    "V": [V_init[j*34:(j+1)*34] for j in range(20)],
                    "W": [W_init[k*34:(k+1)*34] for k in range(20)]}
        
        # Try to optimize
        try:
            result = minimize(tensor_error, params_init, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x
                if best_error < 1e-9:
                    break
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    if best_params is not None and best_error < 1e-9:
        U_mat = best_params[0:136].reshape(4, 34)
        V_mat = best_params[136:816].reshape(20, 34)
        W_mat = best_params[816:1496].reshape(20, 34)
        
        # Convert to lists of lists
        U_out = [list(U_mat[i, :]) for i in range(4)]
        V_out = [list(V_mat[j, :]) for j in range(20)]
        W_out = [list(W_mat[k, :]) for k in range(20)]
        
        # Verify
        if verify(U_out, V_out, W_out):
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # If we couldn't find a rank-34 solution, return the rank-35 solution
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
