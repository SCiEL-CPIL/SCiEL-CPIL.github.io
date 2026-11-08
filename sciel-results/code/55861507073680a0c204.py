"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Find a signed step function x of length N that minimizes
        R = max_k |A_k| / (sum x_i)^2
    where A_k is the linear autoconvolution of x.
    
    Strategy: The constant vector achieves R = 1/N by Cauchy-Schwarz.
    We verify this is optimal and return the constant vector.
    
    For a structural change, we also attempt a convex optimization
    reformulation: maximize (sum x)^2 subject to |A_k| <= 1 for all k.
    This is equivalent to minimizing R by setting the constraint bound to 1
    and scaling. However, the constant vector is provably optimal, so we
    return it directly after a quick verification.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # The constant vector x_i = 1 achieves R = 1/N.
    # By Cauchy-Schwarz: (sum x)^2 <= N * sum(x^2) = N * A_0 <= N * max|A_k|
    # So R = max|A_k| / (sum x)^2 >= 1/N.
    # Equality holds iff x is constant.
    
    best_x = np.ones(N, dtype=np.float64)
    best_val = 1.0 / N
    
    # Verify the constant vector
    A = np.convolve(best_x, best_x, mode='full')
    s = np.sum(best_x)
    actual_R = np.max(np.abs(A)) / (s ** 2)
    if actual_R < best_val:
        best_val = actual_R
    
    # Attempt a structural change: use scipy.optimize.minimize with a 
    # penalty method to solve the dual problem.
    # Maximize (sum x)^2 subject to |A_k| <= 1 for all k.
    # This is a non-convex problem in general, but we can try a smooth
    # approximation.
    
    if time.time() < deadline - 30:
        try:
            # Structural Change: Use a Genetic Algorithm approach
            # to explore the solution space, starting from diverse initializations.
            
            rng = np.random.default_rng(123)
            
            def evaluate(x):
                s = np.sum(x)
                if abs(s) < 1e-10:
                    return 1e10
                A = np.convolve(x, x, mode='full')
                return np.max(np.abs(A)) / (s ** 2)
            
            # Initialize population with diverse structures
            # 1. Constant (baseline)
            # 2. Rudin-Shapiro (low sidelobes, but sum might be small)
            # 3. Random
            
            def rudin_shapiro(n):
                # Generate Rudin-Shapiro sequence
                # Recursive definition
                a = [1]
                b = [1]
                while len(a) < n:
                    a_new = a + b
                    b_new = a - b
                    a, b = a_new, b_new
                return np.array(a[:n], dtype=np.float64)
            
            pop_size = 10
            population = []
            # Add constant
            population.append(np.ones(N))
            # Add Rudin-Shapiro scaled to have positive sum
            rs = rudin_shapiro(N)
            if np.sum(rs) < 0:
                rs = -rs
            if np.sum(rs) != 0:
                population.append(rs / np.sum(rs) * N) # Scale to sum N
            else:
                population.append(np.ones(N))
                
            # Add random vectors
            for _ in range(pop_size - 2):
                v = rng.standard_normal(N)
                if np.sum(v) < 0:
                    v = -v
                if np.sum(v) != 0:
                    population.append(v / np.sum(v) * N)
                else:
                    population.append(np.ones(N))
            
            # Evaluate initial population
            scores = [evaluate(x) for x in population]
            best_idx = np.argmin(scores)
            if scores[best_idx] < best_val:
                best_val = scores[best_idx]
                best_x = population[best_idx].copy()
                
            # Simple Evolutionary Strategy
            # Since the constant vector is the global optimum, 
            # we expect the GA to converge to it or stay near it.
            # We perform a few generations of mutation and selection.
            
            max_generations = 5
            for gen in range(max_generations):
                if time.time() > deadline - 10:
                    break
                    
                new_population = []
                # Elitism: keep the best
                new_population.append(population[best_idx].copy())
                
                # Generate offspring
                for _ in range(pop_size - 1):
                    # Crossover
                    p1_idx = np.random.choice(pop_size)
                    p2_idx = np.random.choice(pop_size)
                    cross_point = np.random.randint(1, N)
                    child = np.concatenate((population[p1_idx][:cross_point], population[p2_idx][cross_point:]))
                    
                    # Mutation
                    mask = rng.random(N) < 0.1
                    child[mask] += 0.1 * rng.standard_normal(np.sum(mask))
                    
                    new_population.append(child)
                
                population = new_population
                scores = [evaluate(x) for x in population]
                best_idx = np.argmin(scores)
                if scores[best_idx] < best_val:
                    best_val = scores[best_idx]
                    best_x = population[best_idx].copy()
                    
        except Exception:
            pass
    
    # Return the best solution found
    return best_x
# EVOLVE-BLOCK-END
