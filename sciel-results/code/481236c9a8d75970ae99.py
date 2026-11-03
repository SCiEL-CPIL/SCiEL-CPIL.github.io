import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 145  # Safety margin before 150s kill
    
    # Target tensor T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i else 0
    # Dimensions: U: n*m=16, V: m*p=28, W: p*n=28
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # Generate target tensor as a sparse set of indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
    
    target_set = set(target_indices)
    
    # Fast verification function using vectorized operations
    # We check if U @ diag(V) @ W^T matches the target tensor
    # T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # We can reshape this as: for each r, the outer product of U[:,r], V[:,r], W[:,r]
    # Sum of these outer products must equal T.
    # To verify quickly, we can sample or use a hash-like check, but exactness is required.
    # However, for a GA, we need a fitness function.
    # Fitness: number of correct entries / total entries.
    # Total entries: 16 * 28 * 28 = 12544.
    # We can compute the full tensor product sum and compare to target.
    # T_calc = np.einsum('ur,vr,wr->uvw', U, V, W)
    # This is 16*28*28*rank operations. If rank=100, that's ~1.2M ops. Very fast.
    
    def fitness(U, V, W):
        # Compute tensor
        T_calc = np.einsum('ur,vr,wr->uvw', U, V, W)
        # Target is 1 at target_indices, 0 elsewhere.
        # Instead of building full target array, we can just check the target spots and a few random non-target spots?
        # No, we need exactness. But for GA fitness, we can approximate error.
        # Let's compute error on target spots and a sample of non-target spots.
        # Actually, building the full target array is cheap (12544 floats).
        T_target = np.zeros((dim_u, dim_v, dim_w), dtype=float)
        for (u, v, w) in target_indices:
            T_target[u, v, w] = 1.0
        
        # Compute L2 error
        err = np.sum((T_calc - T_target) ** 2)
        return err
    
    # Initial best: trivial Kronecker product rank 112
    rank_kron = n * m * p
    U_best = np.zeros((dim_u, rank_kron))
    V_best = np.zeros((dim_v, rank_kron))
    W_best = np.zeros((dim_w, rank_kron))
    for idx, (u, v, w) in enumerate(target_indices):
        U_best[u, idx] = 1
        V_best[v, idx] = 1
        W_best[w, idx] = 1
    
    best_err = 0.0
    best_rank = rank_kron
    
    # Genetic Algorithm Search
    # We want to find a decomposition with rank < best_rank that has error 0.
    # Start with a lower rank, e.g., 90, 80, etc.
    # If we find a solution with error 0, we can try to reduce rank further.
    
    def try_rank(rank, max_iter=500, pop_size=50):
        nonlocal best_err, best_rank, U_best, V_best, W_best
        if rank >= best_rank:
            return False
        
        # Initialize population
        # Use random initialization or perturbation of best known?
        # Random is better for global search.
        # Entries can be integers or half-integers. Let's use floats for optimization, then round.
        # Or search directly in integer/half-integer space?
        # The problem states "integer/half-integer entries".
        # Let's search in continuous space and then project to nearest 0.5 grid?
        # Or just search in continuous space and check if it's close to 0.5 grid?
        # Actually, the evaluator checks exact decomposition.
        # If we find a continuous solution, we can try to round it.
        # But rounding might break exactness.
        
        # Let's stick to integer/half-integer search directly?
        # That's a discrete search.
        # Maybe hybrid: optimize continuously, then refine on grid.
        
        # Let's try a simple GA on continuous variables first.
        U_pop = [np.random.randn(dim_u, rank) * 0.5 for _ in range(pop_size)]
        V_pop = [np.random.randn(dim_v, rank) * 0.5 for _ in range(pop_size)]
        W_pop = [np.random.randn(dim_w, rank) * 0.5 for _ in range(pop_size)]
        
        # Evaluate initial fitness
        fitnesses = []
        for i in range(pop_size):
            f = fitness(U_pop[i], V_pop[i], W_pop[i])
            fitnesses.append(f)
            if f < best_err:
                best_err = f
                U_best = U_pop[i].copy()
                V_best = V_pop[i].copy()
                W_best = W_pop[i].copy()
                best_rank = rank
                if best_err < 1e-9:
                    # Check if it's exactly on the grid?
                    # The problem requires integer/half-integer.
                    # If we find a continuous solution, we must convert it.
                    # This is tricky.
                    pass
        
        for gen in range(max_iter):
            if time.time() > deadline - 5:
                return False
            
            # Selection: keep best half
            indices = np.argsort(fitnesses)
            survivors = indices[:pop_size//2]
            
            U_new = [U_pop[i].copy() for i in survivors]
            V_new = [V_pop[i].copy() for i in survivors]
            W_new = [W_pop[i].copy() for i in survivors]
            
            # Crossover and Mutation to fill rest
            while len(U_new) < pop_size:
                i1, i2 = np.random.randint(0, len(U_new), 2)
                # Uniform crossover
                mask = np.random.rand(dim_u, rank) > 0.5
                U_child = np.where(mask, U_new[i1], U_new[i2])
                
                mask_v = np.random.rand(dim_v, rank) > 0.5
                V_child = np.where(mask_v, V_new[i1], V_new[i2])
                
                mask_w = np.random.rand(dim_w, rank) > 0.5
                W_child = np.where(mask_w, W_new[i1], W_new[i2])
                
                # Mutation
                if np.random.rand() < 0.1:
                    U_child += np.random.randn(dim_u, rank) * 0.1
                if np.random.rand() < 0.1:
                    V_child += np.random.randn(dim_v, rank) * 0.1
                if np.random.rand() < 0.1:
                    W_child += np.random.randn(dim_w, rank) * 0.1
                
                U_new.append(U_child)
                V_new.append(V_child)
                W_new.append(W_child)
            
            # Evaluate new population
            new_fitnesses = []
            for i in range(pop_size):
                f = fitness(U_new[i], V_new[i], W_new[i])
                new_fitnesses.append(f)
                if f < best_err:
                    best_err = f
                    U_best = U_new[i].copy()
                    V_best = V_new[i].copy()
                    W_best = W_new[i].copy()
                    best_rank = rank
                    if best_err < 1e-9:
                        # Found a solution!
                        # Now we need to ensure it's integer/half-integer.
                        # If not, we might fail the evaluator.
                        # Let's try to round it.
                        U_rounded = np.round(U_best * 2) / 2
                        V_rounded = np.round(V_best * 2) / 2
                        W_rounded = np.round(W_best * 2) / 2
                        err_r = fitness(U_rounded, V_rounded, W_rounded)
                        if err_r < 1e-9:
                            U_best = U_rounded
                            V_best = V_rounded
                            W_best = W_rounded
                            return True
                        else:
                            # Keep continuous best for now, but we need a valid one.
                            # If we can't round, we can't use this rank.
                            # We should probably stop this rank search if we can't find a grid solution.
                            # For now, let's continue searching.
                            pass
            
            U_pop, V_pop, W_pop = U_new, V_new, W_new
            fitnesses = new_fitnesses
            
        return best_err < 1e-9

    # Try decreasing ranks
    # Start from best_rank - 1 down to some minimum (e.g., 60)
    # We know 112 works. Let's try 100, 90, 80...
    # But GA is slow. Let's try a few specific ranks.
    
    # Actually, the "laser method" inspiration suggests structural decomposition.
    # But let's stick to the GA idea as the structural change from "direct construction".
    # The direct construction is O(1) and gives rank 112.
    # The GA is a search method.
    
    # To make it more likely to succeed, we can try to exploit the structure.
    # The tensor is T[i,j,k] = delta(i,j,k) in the product sense.
    # This is the standard matrix multiplication tensor.
    # Known lower bounds for 4x4x7 are not trivial.
    # Let's just run the GA for a few ranks.
    
    # Optimization: use integer/half-integer directly?
    # If we search in {0, 0.5, 1, 1.5...}, the space is huge.
    # Continuous search + rounding is a common heuristic.
    
    # Let's try rank 100 first.
    # If time permits, try 90, 80.
    
    ranks_to_try = [100, 95, 90, 85, 80]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        found = try_rank(r, max_iter=200, pop_size=30)
        if found:
            # If we found a valid grid solution, we can try to go lower
            # But let's just return the best valid one found so far.
            # The function returns the best found.
            pass
            
    # If we didn't find a better rank, return the trivial 112 rank solution
    # But we must ensure it's valid. The trivial one is always valid.
    
    # Convert best found to lists for the interface
    # If best_err is not 0, we should return the trivial solution to be safe.
    if best_err > 1e-9:
        # Return trivial solution
        U_out = [[0] * rank_kron for _ in range(dim_u)]
        V_out = [[0] * rank_kron for _ in range(dim_v)]
        W_out = [[0] * rank_kron for _ in range(dim_w)]
        for idx, (u, v, w) in enumerate(target_indices):
            U_out[u][idx] = 1
            V_out[v][idx] = 1
            W_out[w][idx] = 1
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Return the optimized solution
        # Ensure it's on the grid
        U_out = np.round(U_best * 2) / 2
        V_out = np.round(V_best * 2) / 2
        W_out = np.round(W_best * 2) / 2
        
        # Final verification
        err = fitness(U_out, V_out, W_out)
        if err > 1e-9:
            # Fallback to trivial
            U_out = [[0] * rank_kron for _ in range(dim_u)]
            V_out = [[0] * rank_kron for _ in range(dim_v)]
            W_out = [[0] * rank_kron for _ in range(dim_w)]
            for idx, (u, v, w) in enumerate(target_indices):
                U_out[u][idx] = 1
                V_out[v][idx] = 1
                W_out[w][idx] = 1
        else:
            U_out = U_out.tolist()
            V_out = V_out.tolist()
            W_out = W_out.tolist()
            
        return {"U": U_out, "V": V_out, "W": W_out}
