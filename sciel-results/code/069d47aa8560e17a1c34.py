import numpy as np
import time
import random

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def compute_value(h):
        """
        Computes the score for a given height vector h.
        Returns inf if invalid.
        """
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        
        # We can normalize h to have sum 1 to simplify the metric
        # value = abs(2 * N * max(conv(h,h)) / s^2)
        # Let h_norm = h / s. Then sum(h_norm) = 1.
        # value = abs(2 * N * max(conv(h_norm, h_norm)))
        # This is numerically more stable and avoids large/small s issues.
        h_norm = h / s
        z = np.convolve(h_norm, h_norm)
        max_z = np.max(z)
        return abs(2 * N * max_z)

    def normalize_sum(h, target_sum=1.0):
        """
        Adjusts h to have the target sum by subtracting the mean offset.
        This preserves the "shape" relative to the mean.
        """
        current_sum = np.sum(h)
        if current_sum == 0:
            return h
        # We want sum(h_new) = target_sum
        # h_new = h + delta
        # sum(h) + N * delta = target_sum
        # delta = (target_sum - sum(h)) / N
        delta = (target_sum - current_sum) / N
        return h + delta

    # Initial setup
    # Start with a random signal, normalized to sum 1
    # To help convergence, we can start with a structure similar to known good ones
    # or just random. Let's try a mix.
    
    # Strategy: Multiple restarts with annealing.
    # Each restart gets a portion of the time.
    
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    best_h = np.ones(N)
    best_score = compute_value(best_h)
    
    for r in range(num_restarts):
        # Check time
        if time.time() > deadline - 2:
            break
            
        # Initialize h for this restart
        # Option 1: Random Gaussian
        # Option 2: Smoothed random
        # Option 3: Based on previous best with large perturbation
        
        if r == 0:
            # Start with a structured guess: central peak, negative sides
            # This mimics the current program's insight but allows it to evolve freely
            x = np.linspace(-1, 1, N)
            h = np.exp(-10 * x**2) - 0.5 * (np.exp(-10 * (x - 0.5)**2) + np.exp(-10 * (x + 0.5)**2))
        else:
            # Random restart with some smoothing
            h = np.random.randn(N)
            # Apply a light Gaussian filter to make it smoother, which often helps
            kernel_size = 5
            kernel = np.exp(-0.5 * (np.arange(-kernel_size, kernel_size + 1) / (kernel_size/2))**2)
            h = np.convolve(h, kernel, mode='same')
            
        # Normalize to sum 1
        h = normalize_sum(h, 1.0)
        
        current_score = compute_value(h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # Annealing parameters
        T = 0.1  # Initial temperature
        T_min = 0.001
        cooling_rate = 0.99
        
        # Local search loop
        # We will perform as many steps as possible before time_per_restart runs out
        steps_this_restart = 0
        restart_deadline = time.time() + time_per_restart
        
        while time.time() < restart_deadline:
            # Pick a random subset of indices to perturb
            # Perturbing a small number of indices allows local moves
            num_perturb = random.randint(1, 10)
            indices = np.random.choice(N, num_perturb, replace=False)
            
            # Generate perturbations
            # Step size decreases as T decreases
            step_size = T * 0.1
            perturbations = np.random.randn(num_perturb) * step_size
            
            h_new = h.copy()
            h_new[indices] += perturbations
            
            # Re-normalize to sum 1
            h_new = normalize_sum(h_new, 1.0)
            
            new_score = compute_value(h_new)
            
            # Acceptance criterion
            if new_score < current_score:
                # Improvement, always accept
                h = h_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Worse, accept with probability exp(-delta/T)
                delta = new_score - current_score
                if delta < 0:
                    # Should not happen if we checked new_score < current_score
                    h = h_new
                    current_score = new_score
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        h = h_new
                        current_score = new_score
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            steps_this_restart += 1
            
            # If we get stuck for a long time, maybe reheat?
            # For simplicity, let it cool. The restart will handle exploration.
            
    return best_h
