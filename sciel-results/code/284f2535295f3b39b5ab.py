import numpy as np
import time
import math

N = 100

def _energy(X):
    """
    Compute the logarithmic energy.
    X: (N, 3) array of points on the unit sphere.
    Returns: float energy
    """
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # Distances
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid division by zero/log(0)
    # We only sum over i < j
    # To be efficient, we can use the upper triangle
    # But for simplicity and vectorization, let's compute full and mask
    
    # We need sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    
    # Clip to avoid log(0) if points coincide (should not happen with valid init, but safe)
    d = np.maximum(d, 1e-12)
    
    return -np.sum(np.log(d))

def _energy_subset(X, i, j):
    """
    Compute the change in energy if we move points i and j.
    This is an approximation or exact delta calculation.
    For simplicity in SA, we often just recompute total energy, but that's O(N^2).
    With N=100, O(N^2) is 10,000 ops. 
    150s / (10,000 ops) is plenty of time.
    Actually, let's just recompute total energy for robustness, as N is small.
    """
    return _energy(X)

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_point():
    # Uniform random point on sphere
    v = np.random.randn(3)
    return v / np.linalg.norm(v)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial configuration: Fibonacci sphere is a good start
    X = _fibonacci_sphere(N)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    current_energy = _energy(X)
    best_X = X.copy()
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 1e-6
    # Number of iterations depends on time
    # Let's estimate iterations per second
    # We will run a loop until time limit
    
    # Cooling schedule
    # We don't know total iterations in advance, so we use time-based cooling or fixed steps with time check.
    # Better: Define a large number of steps, but break if time runs out.
    # Or, use a cooling factor per step.
    
    # Let's use a cooling factor. If we assume ~100,000 steps in 140s (1.4 ms/step is optimistic for Python+NumPy overhead, maybe 10k-50k steps)
    # Let's target 50,000 steps.
    # Cooling factor lambda = (T_end / T_start)^(1 / steps)
    
    # Actually, let's just run a fixed number of steps that fits in time, or check time inside loop.
    # To ensure we use time well, we'll check time every 100 steps.
    
    T = T_start
    # We'll iterate until time limit.
    # Cooling: T = T_start * (T_end/T_start)^(t/T_total) is hard without T_total.
    # Let's use a fixed cooling rate per step.
    # If we run 20,000 steps:
    # lambda = (1e-6 / 10)^(1/20000) = exp(log(1e-7)/20000) = exp(-16.1/20000) = exp(-0.0008) = 0.9992
    cool_rate = 0.9992
    
    step_count = 0
    max_steps = 100000 # Cap to prevent infinite loop if time check fails
    
    while time.time() < deadline and step_count < max_steps:
        step_count += 1
        
        # Choose move type
        # 50% swap, 50% rotation
        if np.random.rand() < 0.5:
            # Swap two points
            i, j = np.random.choice(N, 2, replace=False)
            X_new = X.copy()
            X_new[i], X_new[j] = X[j], X[i]
        else:
            # Rotate one point slightly
            i = np.random.randint(N)
            # Generate random rotation axis
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            # Random angle, small
            angle = np.random.randn() * 0.1 # Small perturbation
            
            # Rodrigues' rotation formula
            # v' = v cos(a) + (k x v) sin(a) + k (k . v) (1 - cos(a))
            v = X[i]
            k = axis
            c = np.cos(angle)
            s = np.sin(angle)
            
            v_rot = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)
            
            X_new = X.copy()
            X_new[i] = v_rot / np.linalg.norm(v_rot) # Ensure unit length
            
        # Calculate new energy
        E_new = _energy(X_new)
        dE = E_new - current_energy
        
        # Acceptance criterion
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            X = X_new
            current_energy = E_new
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_X = X.copy()
        
        # Cool down
        T *= cool_rate
        
        # Time check every 100 steps to avoid overhead
        if step_count % 100 == 0:
            if time.time() > deadline - 2.0:
                break

    # Final polish with a very low temperature (greedy local search)
    # Use remaining time for local optimization (hill climbing)
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        T = 1e-8
        local_steps = int(remaining_time * 5000) # Estimate 5000 steps per second? Maybe less.
        # Let's just run a fixed number of steps based on time
        local_steps = 50000
        local_deadline = time.time() + remaining_time - 1.0
        
        for _ in range(local_steps):
            if time.time() > local_deadline:
                break
                
            if np.random.rand() < 0.5:
                i, j = np.random.choice(N, 2, replace=False)
                X_new = X.copy()
                X_new[i], X_new[j] = X[j], X[i]
            else:
                i = np.random.randint(N)
                axis = np.random.randn(3)
                axis /= np.linalg.norm(axis)
                angle = np.random.randn() * 0.01 # Very small
                v = X[i]
                k = axis
                c = np.cos(angle)
                s = np.sin(angle)
                v_rot = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)
                X_new = X.copy()
                X_new[i] = v_rot / np.linalg.norm(v_rot)
            
            E_new = _energy(X_new)
            dE = E_new - current_energy
            
            if dE < 0:
                X = X_new
                current_energy = E_new
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_X = X.copy()

    # Final normalization
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
