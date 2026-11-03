"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    We use the Logistic Map at r=4 (a chaotic system) to generate the sign sequence.
    Different irrational initial conditions produce different ergodic trajectories.
    We evaluate the C3 bound (max absolute value of the triple autocorrelation)
    for multiple trajectories and select the one with the minimum bound.
    A fast local search is then applied to the best candidate to further minimize the bound.
    """
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3, defined as the maximum absolute value
        of the third-order autocorrelation coefficients.
        """
        # Pad to power of 2 for FFT efficiency
        # Convolution of 3 sequences of length N has length 3N-2.
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    def greedy_minimize(signs, max_passes=5):
        """
        Perform greedy local search: repeatedly flip signs if it reduces the bound.
        Returns (signs, bound).
        """
        current_bound = compute_c3_bound(signs)
        for _ in range(max_passes):
            improved = False
            for i in range(N):
                signs[i] *= -1.0
                new_bound = compute_c3_bound(signs)
                if new_bound < current_bound:
                    current_bound = new_bound
                    improved = True
                else:
                    signs[i] *= -1.0
            if not improved:
                break
        return signs, current_bound

    # Generate candidates using Logistic Map x_{n+1} = 4 x_n (1 - x_n)
    # We try several irrational seeds to explore different chaotic trajectories.
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0,
        0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9
    ]
    
    best_signs = None
    best_bound = float('inf')
    
    start_time = time.time()
    time_limit = 100.0
    
    # Phase 1: Initialize and Greedy Optimize from multiple seeds
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
            
        signs, bound = greedy_minimize(signs.copy(), max_passes=3)
        
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()

    # Phase 2: Iterated Local Search (ILS) on the best candidate
    current_signs = best_signs.copy()
    current_bound = best_bound
    
    perturb_strength = 3
    max_iterations = 2000
    
    np.random.seed(42)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        perturbed = current_signs.copy()
        for _ in range(perturb_strength):
            i = np.random.randint(0, N)
            perturbed[i] *= -1.0
            
        optimized_signs, optimized_bound = greedy_minimize(perturbed, max_passes=2)
        
        if optimized_bound < current_bound:
            current_bound = optimized_bound
            current_signs = optimized_signs.copy()
            
    # Final Polish
    final_signs, final_bound = greedy_minimize(current_signs, max_passes=10)
    
    return final_signs
# EVOLVE-BLOCK-END
