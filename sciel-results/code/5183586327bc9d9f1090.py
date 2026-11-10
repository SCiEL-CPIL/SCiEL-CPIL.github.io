"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Island Model Differential Evolution for Multi-Objective Optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 325.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Budget allocation
        # We need to stay strictly under 200 evaluations.
        # Let's reserve 10 evaluations for a final "polish" or just keep it simple.
        # Island model: 3 islands (one per objective emphasis) or more?
        # WFG4 is 3-objective. Let's use 3 islands.
        # Population size per island: 20. Total pop = 60.
        # Initial evals: 60.
        # Remaining budget: 140.
        # Generations: 140 / (3 * 20) = ~2.3 generations? That's too few for DE.
        
        # Let's adjust.
        # Smaller islands? 3 islands, size 10. Total 30.
        # Initial evals: 30.
        # Remaining: 170.
        # Generations: 170 / 30 = ~5.6.
        
        # Alternatively, use a smaller population but more generations?
        # DE works best with pop size >= 5 * dim? 5*12=60.
        # If total pop is 60, we have 60 initial evals.
        # 140 remaining. 140/60 = 2.3 generations.
        
        # Maybe 2 islands?
        # Or just use the full budget for a larger population?
        # Let's try: 3 islands, size 15. Total 45.
        # Initial: 45.
        # Remaining: 155.
        # Generations: 155/45 = 3.4.
        
        # Let's stick to a robust DE setup.
        # N_islands = 3
        # N_ind_per_island = 20
        # Total Pop = 60
        # Initial Evals = 60
        # Budget left = 140
        # We can do 2 full generations (60 evals) -> 20 left.
        # Then 1 partial generation or just stop?
        # Actually, DE produces 1 offspring per individual per generation.
        # So 1 generation = 60 evals.
        # 2 generations = 120 evals.
        # Total used = 60 + 120 = 180.
        # 20 evals left.
        
        # Let's implement this.
        
        N_islands = 3
        N_ind = 20
        Total_Pop = N_islands * N_ind
        
        # 1. Initialization
        # Use Sobol for initial spread to ensure coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Generate a power of 2 >= Total_Pop
        next_pow2 = 1
        while next_pow2 < Total_Pop:
            next_pow2 *= 2
            
        points_unit = sampler.random(next_pow2)
        X_init = lo + range_ * points_unit[:Total_Pop]
        
        F_init = np.array([problem(x) for x in X_init])
        
        # Evaluate count
        evals_used = Total_Pop
        
        # Assign to islands
        # Island i focuses on objective i (0, 1, 2)
        # We will use a weighted sum to guide mutation or selection?
        # Standard DE: Random selection of parents.
        # To ensure coverage, we can initialize each island with points 
        # that are good in their respective objective?
        # Or just random assignment? 
        # Better: Sort initial points by each objective and assign top N_ind to each island?
        # No, that might cluster them.
        # Let's just assign round-robin or random, but use "Guided DE" where 
        # the mutation operator is biased towards the island's target objective.
        
        # Simpler: Just use standard DE but with 3 separate populations.
        # To help coverage, we can initialize each island with points from 
        # different regions of the Sobol sequence? 
        # Actually, Sobol is already spread out.
        
        # Let's structure the data
        # islands_X: list of arrays (N_ind, dim)
        # islands_F: list of arrays (N_ind, n_obj)
        
        islands_X = []
        islands_F = []
        
        # Simple assignment: split the initial Sobol points into 3 chunks
        # Chunk 0: first 20, Chunk 1: next 20, Chunk 2: last 20
        # This ensures spatial separation in the initial population.
        for i in range(N_islands):
            islands_X.append(X_init[i*N_ind:(i+1)*N_ind].copy())
            islands_F.append(F_init[i*N_ind:(i+1)*N_ind].copy())
            
        # Store all evaluated points for final hypervolume
        all_X = [X_init.copy()]
        all_F = [F_init.copy()]
        
        # 2. Evolution Loop
        # Max generations based on budget
        max_gens = (self.budget - evals_used) // Total_Pop
        
        # DE Parameters
        CR = 0.9
        F = 0.8
        
        for gen in range(max_gens):
            if time.time() > deadline - 5.0:
                break
                
            for i in range(N_islands):
                if time.time() > deadline - 5.0:
                    break
                    
                pop_X = islands_X[i]
                pop_F = islands_F[i]
                
                # For each individual in the island
                for j in range(N_ind):
                    if evals_used >= self.budget:
                        break
                        
                    # Select parents
                    # DE/rand/1
                    # Pick 3 distinct indices different from j
                    idxs = np.random.choice(N_ind, size=4, replace=False)
                    # Ensure j is not in the first 3? 
                    # Standard DE: a, b, c are distinct and != j
                    while j in idxs[:3]:
                        idxs = np.random.choice(N_ind, size=4, replace=False)
                    
                    a, b, c = idxs[0], idxs[1], idxs[2]
                    
                    # Mutation
                    v = pop_X[a] + F * (pop_X[b] - pop_X[c])
                    
                    # Constraint handling: Clip to bounds
                    # Since WFG4 bounds are [0, 1] usually, or given.
                    # Clipping is a valid repair strategy.
                    v = np.clip(v, lo, hi)
                    
                    # Evaluation
                    f_v = problem(v)
                    evals_used += 1
                    
                    # Record
                    all_X.append(v.reshape(1, -1))
                    all_F.append(np.array([f_v]).reshape(1, -1))
                    
                    # Selection: 
                    # In multi-objective DE, we can use a simple dominance check 
                    # or just replace if it dominates any existing member?
                    # Or use a local hypervolume improvement?
                    # Simplest: Replace if it dominates the current individual j?
                    # Or if it dominates any member of the island?
                    
                    # Let's try: If v dominates pop_X[j], replace.
                    # Or if v dominates any other member, replace the worst?
                    # To maintain diversity, let's just replace j if v is better in 
                    # at least one objective and not worse in others? (Weak dominance)
                    # Or strictly dominate?
                    
                    # Let's use a simple rule: 
                    # If v dominates pop_X[j], replace.
                    # Else, with 50% probability, replace anyway to encourage exploration?
                    # No, that kills quality.
                    
                    # Better: Check if v dominates ANY member in the island.
                    # If yes, replace the dominated member.
                    # If no, replace j if v is closer to the origin (heuristic)?
                    
                    # Let's stick to: Replace j if v dominates j.
                    # This is conservative.
                    
                    # Check dominance: v < pop_X[j] in all objectives?
                    if np.all(np.array(f_v) < pop_F[j]):
                        pop_X[j] = v
                        pop_F[j] = np.array(f_v)
                    else:
                        # If it doesn't dominate j, maybe it dominates someone else?
                        # For efficiency, let's just skip or do a simple check.
                        # Given small pop, checking all is cheap.
                        dominated_idx = -1
                        for k in range(N_ind):
                            if k == j: continue
                            if np.all(np.array(f_v) < pop_F[k]):
                                dominated_idx = k
                                break
                        if dominated_idx != -1:
                            pop_X[dominated_idx] = v
                            pop_F[dominated_idx] = np.array(f_v)
                        else:
                            # No one dominated. 
                            # To prevent stagnation, if v is better than j in 
                            # at least 1 obj and worse in <= 1 obj?
                            # Let's just keep j if no dominance.
                            pass

        # Combine all results
        final_X = np.vstack(all_X)
        final_F = np.vstack(all_F)
        
        return final_X

# EVOLVE-BLOCK-END
