import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[a, b, c, d]
    T = np.zeros((n, m, m, p))
    for b in range(m):
        for c in range(m):
            if b == c:
                T[:, b, c, :] = 1.0

    # We search for rank R
    # Known lower bound is around 25. Trivial is 36.
    # We try ranks from 25 to 30.
    
    for R in range(25, 31):
        # Genetic Algorithm setup
        pop_size = 50
        generations = 200
        
        # Chromosome: (mask_U, mask_V, mask_W)
        # Shapes: (n*m, R), (m*p, R), (p*n, R)
        
        def random_chromosome():
            # Initial density: aim for ~10-20% non-zeros
            # Total non-zeros should be roughly 3 * R * density
            # Let's just use random density 0.2
            mask_U = np.random.rand(n * m, R) < 0.2
            mask_V = np.random.rand(m * p, R) < 0.2
            mask_W = np.random.rand(p * n, R) < 0.2
            return mask_U, mask_V, mask_W

        population = [random_chromosome() for _ in range(pop_size)]
        
        def evaluate(chromosome):
            mask_U, mask_V, mask_W = chromosome
            
            # Try a few random instantiations of U, V
            # Use +/- 1 for non-zeros
            for _ in range(5):
                U_val = np.zeros((n * m, R))
                V_val = np.zeros((m * p, R))
                
                # Assign random signs to non-zeros
                U_val[mask_U] = np.random.choice([-1, 1], size=np.sum(mask_U))
                V_val[mask_V] = np.random.choice([-1, 1], size=np.sum(mask_V))
                
                # Solve for W
                # Equation: sum_r U[ab, r] V[cd, r] W[da, r] = T[a, b, c, d]
                # Indices:
                # a: 0..n-1
                # b: 0..m-1
                # c: 0..m-1
                # d: 0..p-1
                # W index: d*n + a
                
                # We want to solve A x = b
                # x is the vector of W entries where mask_W is True.
                # Let's construct the linear system.
                
                # Number of non-zeros in W
                nz_W = np.sum(mask_W)
                if nz_W == 0:
                    return 1.0 # Bad fitness
                
                # Map W indices to variable indices
                # We need to build A of shape (n*m*m*p, nz_W)
                # And b of shape (n*m*m*p,)
                
                # This might be memory intensive if done naively for large R, 
                # but n*m*m*p = 3*3*3*4 = 108.
                # nz_W is at most p*n*R = 12*R. For R=25, nz_W ~ 300.
                # 108 x 300 matrix is small.
                
                A = np.zeros((n * m * m * p, nz_W))
                b = np.zeros(n * m * m * p)
                
                # Flatten T
                T_flat = T.flatten()
                b[:] = T_flat
                
                # Iterate over all tensor indices
                # a, b_idx, c, d
                idx = 0
                for a in range(n):
                    for b_idx in range(m):
                        for c in range(m):
                            for d in range(p):
                                # Coefficient for W[d*n + a, r] is U[a*m + b_idx, r] * V[c*p + d, r]
                                # We only include W if mask_W[d*n + a, r] is True.
                                
                                # Get the U and V values for this (a,b) and (c,d)
                                U_row = U_val[a * m + b_idx, :]
                                V_row = V_val[c * p + d, :]
                                
                                # Product vector P[r] = U_row[r] * V_row[r]
                                P = U_row * V_row
                                
                                # For each r, if mask_W[d*n + a, r] is True, 
                                # add P[r] to A[idx, var_idx]
                                
                                # We need to map (d*n + a, r) to var_idx
                                # Precompute mapping for W?
                                # Since mask_W changes, we do it on the fly or precompute for this chromosome.
                                
                                # Optimization: Precompute var indices for W
                                # This is done outside the loop in evaluate? 
                                # For now, simple approach:
                                # We'll just check mask_W inside the loop.
                                # To be faster, let's get the list of (row, col) where mask_W is True.
                                
                                pass # Placeholder for logic below
                                
                                idx += 1
                                
                # The loop above is too slow if we don't vectorize the A construction.
                # Let's vectorize the A construction.
                
                # Get indices of non-zeros in W
                w_rows, w_cols = np.where(mask_W)
                var_indices = np.arange(len(w_rows))
                
                # We need to fill A.
                # A[i, j] corresponds to equation i and variable j (which is W[w_rows[j], w_cols[j]])
                # Equation i is (a, b, c, d).
                # Variable j is (d', a', r).
                # Non-zero if (d, a) == (d', a') AND r == r'.
                # Wait, the equation sums over r.
                # Term in equation (a,b,c,d) for variable (d', a', r) is:
                # If d==d' and a==a': U[a*m+b, r] * V[c*p+d, r]
                # Else: 0
                
                # So A[i, j] = U[a_i*m+b_i, r_j] * V[c_i*p+d_i, r_j] if (a_i, d_i) == (a_j, d_j) else 0.
                
                # Let's build A using broadcasting or loops over the small dimension R.
                
                A = np.zeros((n * m * m * p, nz_W))
                
                # Map equation index to (a, b, c, d)
                # We can precompute a,b,c,d for each equation index
                a_eq = np.repeat(np.repeat(np.repeat(np.arange(n), m), m), p)
                b_eq = np.repeat(np.repeat(np.arange(m), m), p)
                c_eq = np.repeat(np.arange(m), m * p)
                d_eq = np.arange(p)
                
                # Map variable index to (d_w, a_w, r)
                # w_rows is d*n + a. So d_w = w_rows // n, a_w = w_rows % n.
                d_w = w_rows // n
                a_w = w_rows % n
                r_w = w_cols
                
                # We need to compute A[i, j]
                # Condition: a_eq[i] == a_w[j] AND d_eq[i] == d_w[j]
                # Value: U[a_eq[i]*m + b_eq[i], r_w[j]] * V[c_eq[i]*p + d_eq[i], r_w[j]]
                
                # This is a big matrix multiplication / indexing problem.
                # Let's do it in chunks or use sparse?
                # n*m*m*p = 108. nz_W is small.
                # We can loop over j (variables) and fill columns of A.
                
                for j in range(nz_W):
                    aw = a_w[j]
                    dw = d_w[j]
                    rw = r_w[j]
                    
                    # Find equations i where a_eq[i] == aw and d_eq[i] == dw
                    # a_eq and d_eq are vectors of length 108.
                    # We can use boolean indexing.
                    mask_eq = (a_eq == aw) & (d_eq == dw)
                    eq_indices = np.where(mask_eq)[0]
                    
                    if len(eq_indices) == 0:
                        continue
                    
                    # For these equations, the coefficient is U[aw*m + b_eq[i], rw] * V[c_eq[i]*p + dw, rw]
                    # b_eq and c_eq are vectors. We need values at eq_indices.
                    b_vals = b_eq[eq_indices]
                    c_vals = c_eq[eq_indices]
                    
                    # U indices: aw*m + b_vals
                    # V indices: c_vals*p + dw
                    u_vals = U_val[aw * m + b_vals, rw]
                    v_vals = V_val[c_vals * p + dw, rw]
                    
                    A[eq_indices, j] = u_vals * v_vals
                
                # Solve A x = b
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    return 1.0
                
                # Check residual
                if residuals is None or residuals[0] > 1e-6:
                    return 1.0 + residuals[0] if residuals is not None else 1.0
                
                # Reconstruct W
                W_val = np.zeros((p * n, R))
                W_val[w_rows, w_cols] = x
                
                # Check if W_val is "integer-like" or "half-integer-like"
                # The task allows integers/half-integers.
                # We can check if rounding works.
                U_round = np.round(U_val) # U was +/- 1, so it's already integer
                V_round = np.round(V_val)
                W_round = np.round(W_val * 2) / 2 # Try half-integers
                
                # Verify the decomposition with rounded values
                U_r = U_round.reshape(n, m, R)
                V_r = V_round.reshape(m, p, R)
                W_r = W_round.reshape(p, n, R)
                
                S = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                err = np.sum((S - T) ** 2)
                
                if err < 1e-6:
                    return 0.0 # Perfect fitness
            
            # If no random instantiation worked
            return 1.0

        # GA Loop
        best_fitness = float('inf')
        best_chrom = population[0]
        
        for gen in range(generations):
            fitnesses = [evaluate(ch) for ch in population]
            
            # Update best
            min_idx = np.argmin(fitnesses)
            if fitnesses[min_idx] < best_fitness:
                best_fitness = fitnesses[min_idx]
                best_chrom = population[min_idx]
                
            if best_fitness < 1e-6:
                # Found a solution
                mask_U, mask_V, mask_W = best_chrom
                # Re-run evaluate to get the values
                # We need to return the actual values.
                # Let's re-instantiate and solve again to get the specific values that worked.
                # We modify evaluate to return the values if it finds a solution.
                # For now, we just break and re-derive.
                pass

            # Selection, Crossover, Mutation
            new_population = []
            # Elitism
            new_population.append(best_chrom)
            
            while len(new_population) < pop_size:
                # Tournament selection
                i1, i2 = np.random.randint(0, pop_size, 2)
                if fitnesses[i1] < fitnesses[i2]:
                    p1 = population[i1]
                else:
                    p1 = population[i2]
                    
                i1, i2 = np.random.randint(0, pop_size, 2)
                if fitnesses[i1] < fitnesses[i2]:
                    p2 = population[i1]
                else:
                    p2 = population[i2]
                
                # Crossover: Uniform crossover on masks
                c1_U = np.where(np.random.rand(*p1[0].shape) < 0.5, p1[0], p2[0])
                c1_V = np.where(np.random.rand(*p1[1].shape) < 0.5, p1[1], p2[1])
                c1_W = np.where(np.random.rand(*p1[2].shape) < 0.5, p1[2], p2[2])
                
                # Mutation: Flip bits with prob 0.01
                c1_U = np.where(np.random.rand(*c1_U.shape) < 0.01, ~c1_U, c1_U)
                c1_V = np.where(np.random.rand(*c1_V.shape) < 0.01, ~c1_V, c1_V)
                c1_W = np.where(np.random.rand(*c1_W.shape) < 0.01, ~c1_W, c1_W)
                
                new_population.append((c1_U, c1_V, c1_W))
                
            population = new_population
            
            # Early stopping if we found a perfect solution
            if best_fitness < 1e-6:
                break
                
        if best_fitness < 1e-6:
            # We need to extract the solution values from best_chrom
            mask_U, mask_V, mask_W = best_chrom
            # Re-run the logic to get values
            for _ in range(10):
                U_val = np.zeros((n * m, R))
                V_val = np.zeros((m * p, R))
                U_val[mask_U] = np.random.choice([-1, 1], size=np.sum(mask_U))
                V_val[mask_V] = np.random.choice([-1, 1], size=np.sum(mask_V))
                
                w_rows, w_cols = np.where(mask_W)
                nz_W = len(w_rows)
                if nz_W == 0: continue
                
                A = np.zeros((n * m * m * p, nz_W))
                b = T.flatten()
                
                a_eq = np.repeat(np.repeat(np.repeat(np.arange(n), m), m), p)
                b_eq = np.repeat(np.repeat(np.arange(m), m), p)
                c_eq = np.repeat(np.arange(m), m * p)
                d_eq = np.arange(p)
                
                d_w = w_rows // n
                a_w = w_rows % n
                r_w = w_cols
                
                for j in range(nz_W):
                    aw = a_w[j]
                    dw = d_w[j]
                    rw = r_w[j]
                    mask_eq = (a_eq == aw) & (d_eq == dw)
                    eq_indices = np.where(mask_eq)[0]
                    if len(eq_indices) == 0: continue
                    b_vals = b_eq[eq_indices]
                    c_vals = c_eq[eq_indices]
                    u_vals = U_val[aw * m + b_vals, rw]
                    v_vals = V_val[c_vals * p + dw, rw]
                    A[eq_indices, j] = u_vals * v_vals
                    
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                    
                if residuals is None or residuals[0] > 1e-6:
                    continue
                    
                W_val = np.zeros((p * n, R))
                W_val[w_rows, w_cols] = x
                
                U_round = np.round(U_val)
                V_round = np.round(V_val)
                W_round = np.round(W_val * 2) / 2
                
                U_r = U_round.reshape(n, m, R)
                V_r = V_round.reshape(m, p, R)
                W_r = W_round.reshape(p, n, R)
                
                S = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                err = np.sum((S - T) ** 2)
                
                if err < 1e-6:
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }

    # Fallback: Schoolbook
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
