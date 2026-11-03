"""Rewritten program for the C3-prime signed autocorrelation task."""
import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 140s deadline, 20s margin before 160s hard kill (150s wall-clock limit)
    
    # Initial random signed step function
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Compute initial autoconvolution
    ac = fftconvolve(x, x, mode='full')
    # For autoconvolution of a sequence with itself, the relevant part is the central portion
    # Actually, for "autoconvolution" in the context of step functions and C3',
    # we need to be careful about what's being measured.
    # The task says "maximum absolute autoconvolution divided by squared integral"
    # For a discrete signal x, autoconvolution is x * x (convolution with itself)
    # The "squared integral" is (sum x)^2
    # C3' = max|ac| / (sum x)^2
    
    # Let's compute the score function
    def compute_score(x):
        ac = fftconvolve(x, x, mode='full')
        # The maximum absolute value of the autoconvolution
        max_abs_ac = np.max(np.abs(ac))
        # Squared integral (squared sum for discrete)
        sq_integral = np.sum(x) ** 2
        if abs(sq_integral) < 1e-10:
            return np.inf
        return max_abs_ac / abs(sq_integral)
    
    best_x = x.copy()
    best_score = compute_score(best_x)
    
    # Local search with sign flips using O(N) incremental update
    # When we flip x[i] -> -x[i], the autoconvolution changes can be computed efficiently
    # ac[k] = sum_j x[j] * x[j+k] for autoconvolution
    # Flipping x[i] changes ac by: for each k, ac[k] changes by -2*x[i]*x[i+k] + 2*x[i]*x[i-k]... 
    # Actually, let's use a simpler approach: recompute when needed, but use the O(N) trick
    # for evaluating the effect of a flip on the max absolute value.
    
    # Alternative: use the fact that for a sign flip at position i,
    # the new ac differs from old ac in a predictable way.
    # But for simplicity and speed, let's do a more efficient local search.
    
    # Let's implement a proper incremental update.
    # ac[k] = sum_{j} x[j] x[j+k]
    # If we flip x[i] to -x[i], then:
    # new_ac[k] = ac[k] - 2*x[i]*x[i+k] + 2*x[i]*x[i-k]... wait, let me think.
    # Actually, x[i] appears in ac[k] when j=i (term x[i]*x[i+k]) or j+k=i (term x[i-k]*x[i]).
    # So the change in ac[k] is: -2*x[i]*x[i+k] for the j=i term, and -2*x[i-k]*x[i] for the j+k=i term.
    # But x[i-k]*x[i] = x[i]*x[i-k], so total change = -2*x[i]*(x[i+k] + x[i-k])... but this isn't quite right
    # because of boundary effects.
    
    # Let's just do a simpler local search: randomly try flips and compute the score.
    # With N=400, fftconvolve is fast enough for many iterations.
    
    import random
    random.seed(42)
    
    iterations = 0
    max_iterations = 500000
    
    # Simulated annealing or simple local search with random restarts
    temp = 1.0
    cooling = 0.9995
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random position to flip
        i = random.randint(0, N-1)
        x[i] = -x[i]
        
        new_score = compute_score(x)
        
        # Accept if better or with probability exp(-delta/temp)
        if new_score < best_score:
            best_x = x.copy()
            best_score = new_score
        else:
            delta = new_score - best_score
            if delta > 0:
                prob = np.exp(-delta / max(temp, 1e-10))
                if random.random() > prob:
                    x[i] = -x[i]  # reject
            else:
                # new_score >= best_score but we might still accept for exploration
                prob = np.exp(-delta / max(temp, 1e-10))
                if random.random() > prob:
                    x[i] = -x[i]
        
        iterations += 1
        temp *= cooling
        
        # If temperature is very low, do a few more deterministic local search steps
        if temp < 0.01:
            temp = 0.01
    
    # Final polish: try all single flips from best_x
    x = best_x.copy()
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            new_score = compute_score(x)
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
                improved = True
            else:
                x[i] = -x[i]
    
    return best_x
# EVOLVE-BLOCK-END
