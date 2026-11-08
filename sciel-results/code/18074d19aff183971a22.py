"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the L-infinity norm of the autoconvolution.
    
    The autoconvolution of a step function h with 400 equal intervals is:
    (h * h)[k] = sum_{i=0}^{k} h[i] * h[k-i] for k = 0, ..., 2N-2
    
    We want to choose h[i] in {-1, 1} to minimize max_k |(h*h)[k]|.
    
    Strategy:
    1. Use continuous relaxation: x in [-1, 1]^N
    2. Minimize L-infinity norm of autoconvolution via projected gradient descent
    3. Discretize to {-1, 1} via hard thresholding
    4. Local search polish with simulated annealing
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Initialize with alternating pattern (known to give good autoconvolution bounds)
    x = np.ones(N)
    x[1::2] = -1.0
    
    def autoconvolution(h):
        """Compute autoconvolution of h with itself."""
        return np.convolve(h, h, mode='full')
    
    def compute_score(h):
        """Compute L-infinity norm of autoconvolution."""
        conv = autoconvolution(h)
        return np.max(np.abs(conv))
    
    # Projected gradient descent on continuous relaxation
    # Objective: min_t t s.t. |conv_k| <= t for all k
    # Equivalent to: min max_k |conv_k|
    
    # Use smooth approximation: soft-max via log-sum-exp
    def soft_max_smooth(conv, beta=10.0):
        """Smooth approximation of max(|conv_k|) using log-sum-exp."""
        abs_conv = np.abs(conv)
        # Scale to avoid overflow
        max_val = np.max(abs_conv)
        if max_val < 1e-10:
            return 0.0
        scaled = abs_conv / max_val
        # Log-sum-exp trick
        m = np.max(scaled)
        return max_val * (m + np.log(np.sum(np.exp(scaled - m)) / len(scaled)))
    
    def gradient_of_soft_max(conv, beta=10.0):
        """Gradient of soft max with respect to h."""
        abs_conv = np.abs(conv)
        max_val = np.max(abs_conv)
        if max_val < 1e-10:
            return np.zeros_like(conv)
        scaled = abs_conv / max_val
        m = np.max(scaled)
        weights = np.exp(scaled - m)
        weights /= np.sum(weights)
        
        # Gradient of |conv_k| w.r.t. h
        # conv_k = sum_i h[i] h[k-i]
        # d|conv_k|/dh[i] = sign(conv_k) * (h[k-i] + h[i] if i != k-i else 2*h[i])
        grad = np.zeros(N)
        for k in range(2*N - 1):
            if abs_conv[k] < 1e-10:
                continue
            sign_k = np.sign(conv[k])
            weight = weights[k] * sign_k
            for i in range(min(k+1, N)):
                j = k - i
                if j >= N:
                    continue
                if i == j:
                    grad[i] += 2.0 * weight * h[i]
                else:
                    grad[i] += weight * h[j]
                    grad[j] += weight * h[i]
        return grad
    
    # Simpler approach: use subgradient method directly
    def subgradient_method(h, max_iters=500, step_size=0.01):
        """Subgradient method for minimizing max |conv_k|."""
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            conv = autoconvolution(h)
            abs_conv = np.abs(conv)
            max_val = np.max(abs_conv)
            max_indices = np.where(abs_conv == max_val)[0]
            
            # Subgradient: average of gradients at active constraints
            grad = np.zeros(N)
            for k in max_indices:
                sign_k = np.sign(conv[k])
                for i in range(min(k+1, N)):
                    j = k - i
                    if j >= N:
                        continue
                    if i == j:
                        grad[i] += sign_k * 2.0 * h[i]
                    else:
                        grad[i] += sign_k * h[j]
                    grad[j] += sign_k * h[i]
            grad /= len(max_indices)
            
            # Projected gradient step
            lr = step_size / (1.0 + it * 0.01)
            h_new = h - lr * grad
            h_new = np.clip(h_new, -1.0, 1.0)
            
            # Accept if better
            if compute_score(h_new) < compute_score(h):
                h = h_new
                step_size *= 1.01
            else:
                step_size *= 0.95
        
        return h
    
    # Run subgradient method
    x_continuous = subgradient_method(x.copy(), max_iters=200, step_size=0.05)
    
    # Discretize to {-1, 1}
    x_binary = np.where(x_continuous >= 0, 1.0, -1.0)
    
    # Local search polish with simulated annealing
    def enforce_non_zero_sum(h):
        """Ensure the sum of heights is not zero by flipping one element if necessary."""
        s = np.sum(h)
        if s == 0:
            # Flip the first element. This changes the sum by +2 or -2, making it non-zero.
            h = h.copy()
            h[0] = -h[0]
        return h

    def genetic_search(h_init, max_gens=20, pop_size=50):
        """
        Genetic Algorithm with tournament selection and crossover.
        Uses the time deadline to stop early.
        """
        n = len(h_init)
        # Initialize population
        population = [h_init.copy() for _ in range(pop_size)]
        # Add some diversity
        for i in range(1, pop_size):
            if np.random.rand() < 0.5:
                population[i] = -h_init.copy()
            else:
                # Random noise
                mask = np.random.rand(n) < 0.1
                population[i] = h_init.copy()
                population[i][mask] = -population[i][mask]
        
        # Evaluate initial population
        scores = [compute_score(p) for p in population]
        best_idx = np.argmin(scores)
        best_h = population[best_idx].copy()
        best_score = scores[best_idx]
        
        for gen in range(max_gens):
            if time.time() > deadline - 10:
                break
            
            # Tournament selection
            def tournament_select():
                # Pick 3 random individuals
                candidates = np.random.randint(0, pop_size, 3)
                candidate_scores = [scores[i] for i in candidates]
                winner_idx = candidates[np.argmin(candidate_scores)]
                return population[winner_idx].copy()
            
            new_population = []
            for _ in range(pop_size):
                # Crossover
                if np.random.rand() < 0.8:
                    p1 = tournament_select()
                    p2 = tournament_select()
                    # Single point crossover
                    cross_point = np.random.randint(1, n)
                    child = np.concatenate([p1[:cross_point], p2[cross_point:]])
                else:
                    child = tournament_select()
                
                # Mutation
                # Flip a few bits
                num_mutations = np.random.poisson(1.0)
                if num_mutations > 0:
                    mut_indices = np.random.choice(n, num_mutations, replace=False)
                    child[mut_indices] = -child[mut_indices]
                
                new_population.append(child)
            
            # Evaluate new population
            new_scores = [compute_score(p) for p in new_population]
            
            # Elitism: keep the best from previous population
            # Find best in new population
            new_best_idx = np.argmin(new_scores)
            if new_scores[new_best_idx] < best_score:
                best_score = new_scores[new_best_idx]
                best_h = new_population[new_best_idx].copy()
            
            # Combine old and new to form next population (keep top half)
            # Sort by score
            combined = [(s, i, p) for s, i, p in zip(scores, range(pop_size), population)]
            combined += [(s, i + pop_size, p) for s, i, p in zip(new_scores, range(pop_size), new_population)]
            combined.sort(key=lambda x: x[0])
            
            # Take top pop_size
            population = [p for _, _, p in combined[:pop_size]]
            scores = [s for s, _, _ in combined[:pop_size]]
            
            # Update best if found in combined (though we already checked new, check combined for safety)
            if combined[0][0] < best_score:
                best_score = combined[0][0]
                best_h = combined[0][2].copy()
        
        return best_h, best_score
    
    # Run genetic search
    x_binary, best_score = genetic_search(x_binary, max_gens=30, pop_size=30)
    
    # Try multiple restarts if time permits
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Different initial patterns
        if restart == 0:
            init = np.ones(N)
        elif restart == 1:
            init = np.ones(N)
            init[::2] = -1.0
        elif restart == 2:
            init = np.random.choice([-1.0, 1.0], size=N)
        elif restart == 3:
            # Block pattern
            init = np.ones(N)
            init[100:200] = -1.0
            init[300:400] = -1.0
        else:
            # Sinusoidal pattern
            t = np.linspace(0, 4*np.pi, N)
            init = np.sign(np.sin(t))
            init[init == 0] = 1.0
        
        # Quick subgradient
        x_c = subgradient_method(init.copy(), max_iters=50, step_size=0.03)
        x_b = np.where(x_c >= 0, 1.0, -1.0)
        
        # Genetic search
        x_b, score = genetic_search(x_b, max_gens=20, pop_size=20)
        
        if score < best_score:
            best_score = score
            x_binary = x_b
    
    # Enforce non-zero sum constraint
    x_binary = enforce_non_zero_sum(x_binary)
    
    return x_binary
# EVOLVE-BLOCK-END
