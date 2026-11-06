"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all arithmetic progressions with length >= 2
    aps = []
    for d in range(1, N // 2 + 1):
        for start in range(N - d):
            if start + d < N:
                indices = list(range(start, N, d))
                if len(indices) >= 2:
                    aps.append(indices)
    
    num_aps = len(aps)
    
    # Build the AP matrix as a numpy array of shape (num_aps, N) with int8
    M = np.zeros((num_aps, N), dtype=np.int8)
    for i, ap in enumerate(aps):
        for idx in ap:
            M[i, idx] = 1
    
    # Convert to float32 for fast matvec
    Mf = M.astype(np.float32)
    
    def compute_max_abs_sum(signs):
        """Compute maximum absolute sum over all APs using matrix-vector product."""
        sums = Mf.dot(signs.astype(np.float32))
        return int(np.max(np.abs(sums)))
    
    def compute_surrogate_loss(x):
        """Compute smooth surrogate: sum of squares of AP sums."""
        sums = Mf.dot(x)
        return float(np.sum(sums * sums))
    
    def compute_surrogate_grad(x):
        """Compute gradient of surrogate loss: 2 * Mf^T @ (Mf @ x)."""
        sums = Mf.dot(x)
        return 2.0 * Mf.T.dot(sums)
    
    rng = random.Random(42)
    
    # Phase 1: Gradient-based continuous optimization
    # Initialize with random continuous values in [-1, 1]
    best_signs = None
    best_cost = float('inf')
    
    # Try multiple initializations
    for init_seed in [42, 0, 1, 2, 3]:
        if time.time() >= deadline - 5.0:
            break
        
        rng_init = random.Random(init_seed)
        x = np.array([rng_init.uniform(-1, 1) for _ in range(N)], dtype=np.float64)
        
        # SGD with momentum
        lr = 0.01
        momentum = 0.9
        v = np.zeros(N, dtype=np.float64)
        
        # Run SGD for a number of steps
        num_steps = 500
        for step in range(num_steps):
            if time.time() >= deadline - 10.0:
                break
            
            grad = compute_surrogate_grad(x)
            v = momentum * v - lr * grad
            x = x + v
            
            # Project back to [-1, 1] to keep values bounded
            x = np.clip(x, -1.0, 1.0)
        
        # Discretize to ±1
        signs = np.sign(x).astype(np.int8)
        # Handle any zeros (shouldn't happen with uniform init, but be safe)
        signs[signs == 0] = 1
        
        cost = compute_max_abs_sum(signs)
        if cost < best_cost:
            best_cost = cost
            best_signs = signs.copy()
    
    # Phase 2: Simulated Annealing refinement
    if best_signs is not None:
        signs = best_signs.copy()
        current_cost = best_cost
    else:
        signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        current_cost = compute_max_abs_sum(signs)
        best_signs = signs.copy()
        best_cost = current_cost
    
    T = 5.0  # Initial temperature
    cooling_rate = 0.9995
    
    while time.time() < deadline - 5.0:
        if T < 0.01:
            break
            
        # Pick a random index to flip
        idx = rng.randrange(N)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        new_cost = compute_max_abs_sum(signs)
        
        if new_cost <= current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-(new_cost - current_cost) / T)
            delta = new_cost - current_cost
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_cost = new_cost
                else:
                    signs[idx] = old_sign
        
        T *= cooling_rate
    
    # Final local search: try flipping each position and keep if improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            old_sign = signs[idx]
            signs[idx] = -old_sign
            new_cost = compute_max_abs_sum(signs)
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
                current_cost = new_cost
                improved = True
            else:
                signs[idx] = old_sign
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
