import numpy as np
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T[a, b, c, d] = 1 if b == c else 0
    # We check if sum_r U[a*m+b, r] * V[c*p+d, r] * W[d*n+a, r] == T[a,b,c,d]
    
    # Precompute indices for the tensor equation
    # U index: a*m + b
    # V index: c*p + d
    # W index: d*n + a
    
    # We will search for binary (0/1) solutions.
    # If a binary solution exists for rank R, it is a valid integer solution.
    
    # Start from a low rank estimate.
    # Theoretical lower bound is max(n*m, m*p, p*n) = 12.
    # Schoolbook is 36.
    # Try ranks from 25 to 35.
    
    for R in range(25, 36):
        # Genetic Algorithm Parameters
        pop_size = 50
        generations = 100
        mutation_rate = 0.05
        crossover_rate = 0.8
        
        # Initialize population with random sparse binary matrices
        population = []
        for _ in range(pop_size):
            # Density heuristic: try to keep it sparse
            density = 0.3 
            U = (np.random.rand(n * m, R) < density).astype(int)
            V = (np.random.rand(m * p, R) < density).astype(int)
            W = (np.random.rand(p * n, R) < density).astype(int)
            population.append((U, V, W))
        
        best_fitness = float('inf')
        best_solution = None
        
        for gen in range(generations):
            # Evaluate fitness
            new_population = []
            for ind in population:
                U, V, W = ind
                # Compute error: number of mismatches in the tensor
                # We need sum_r U[a,b,r] V[c,d,r] W[d,a,r] == T[a,b,c,d]
                # T is 1 if b==c, 0 otherwise.
                
                # Vectorized check:
                # S[a,b,c,d] = sum_r U[a,b,r] V[c,d,r] W[d,a,r]
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                
                S = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                
                # Target T
                # T[a,b,c,d] = 1 if b==c else 0
                # We can construct T or just check against it.
                # To avoid constructing T every time, we can compute the error directly.
                # Error = sum_{a,b,c,d} (S[a,b,c,d] - T[a,b,c,d])^2
                # Since S and T are integers (S is sum of products of 0/1, so integer),
                # (S-T)^2 is 0 if S==T, 1 if S!=T (since S,T are small).
                # Actually S can be larger than 1.
                
                # Construct T efficiently
                # T is 1 where b==c.
                # Indices: a in 0..n-1, b in 0..m-1, c in 0..m-1, d in 0..p-1
                # T[a,b,c,d] = 1 if b==c else 0
                
                # We can compute the error without explicitly building T of size 3*3*3*4=108.
                # It's small, so building it is fine.
                T = np.zeros((n, m, m, p), dtype=int)
                for b in range(m):
                    T[:, b, b, :] = 1
                
                err = np.sum((S - T) ** 2)
                
                if err == 0:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                
                new_population.append((ind, err))
            
            # Sort by fitness
            new_population.sort(key=lambda x: x[1])
            
            # Keep best
            if new_population[0][1] < best_fitness:
                best_fitness = new_population[0][1]
                best_solution = new_population[0][0]
                
            if best_fitness == 0:
                return {
                    "U": best_solution[0].tolist(),
                    "V": best_solution[1].tolist(),
                    "W": best_solution[2].tolist()
                }
            
            # Selection: Keep top half
            survivors = [ind for ind, fit in new_population[:pop_size//2]]
            
            # Crossover and Mutation to fill rest
            offspring = []
            while len(offspring) < pop_size - len(survivors):
                if random.random() < crossover_rate:
                    # Crossover
                    p1, p2 = random.sample(survivors, 2)
                    # Uniform crossover
                    U1, V1, W1 = p1
                    U2, V2, W2 = p2
                    
                    mask_U = np.random.rand(n*m, R) < 0.5
                    mask_V = np.random.rand(m*p, R) < 0.5
                    mask_W = np.random.rand(p*n, R) < 0.5
                    
                    U_new = np.where(mask_U, U1, U2)
                    V_new = np.where(mask_V, V1, V2)
                    W_new = np.where(mask_W, W1, W2)
                else:
                    p1 = random.choice(survivors)
                    U_new = p1[0].copy()
                    V_new = p1[1].copy()
                    W_new = p1[2].copy()
                
                # Mutation
                # Flip bits with probability mutation_rate
                if random.random() < mutation_rate:
                    U_new[random.randint(0, n*m-1), random.randint(0, R-1)] = 1 - U_new[random.randint(0, n*m-1), random.randint(0, R-1)]
                if random.random() < mutation_rate:
                    V_new[random.randint(0, m*p-1), random.randint(0, R-1)] = 1 - V_new[random.randint(0, m*p-1), random.randint(0, R-1)]
                if random.random() < mutation_rate:
                    W_new[random.randint(0, p*n-1), random.randint(0, R-1)] = 1 - W_new[random.randint(0, p*n-1), random.randint(0, R-1)]
                    
                offspring.append((U_new, V_new, W_new))
            
            population = survivors + offspring
            
    # Fallback to schoolbook rank 36
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
