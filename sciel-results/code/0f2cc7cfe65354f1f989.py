import numpy as np
import time

N = 600

def _score(h):
    """Compute the normalized maximum autoconvolution value."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Normalize to sum 1 for stability in score calculation, 
    # though the ratio is scale invariant.
    h_norm = h / total
    conv = np.convolve(h_norm, h_norm, mode='full')
    return conv.max()

def construct():
    start_time = time.time()
    time_limit = 100.0  # Leave some margin for 120s limit
    
    best_h = None
    best_score = float('inf')
    
    # Initializations
    inits = []
    # 1. Uniform
    inits.append(np.ones(N) / N)
    # 2. Random
    np.random.seed(42)
    for _ in range(3):
        h_rand = np.random.rand(N)
        inits.append(h_rand / h_rand.sum())
    # 3. Gaussian-like
    x = np.arange(N) - N//2
    h_gauss = np.exp(-x**2 / (2 * (N/4)**2))
    inits.append(h_gauss / h_gauss.sum())

    for h0 in inits:
        h = h0.copy()
        current_score = _score(h)
        best_local_h = h.copy()
        best_local_score = current_score
        
        # Simulated Annealing
        # Parameters
        T_start = 0.01  # Initial temperature (scale of score differences)
        T_end = 1e-6
        alpha = 0.999  # Cooling rate
        
        # Estimate number of iterations possible
        # Each iteration: 2 random indices, 1 conv (O(N^2) ~ 360k ops), 1 max.
        # In Python, 10k iterations might take ~10-20s? 
        # Let's test with a fixed number first, say 5000, and see time.
        # Actually, let's run until time limit.
        
        T = T_start
        it = 0
        while time.time() - start_time < time_limit:
            # Generate move
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            
            # Transfer delta
            # Scale of delta should be small
            delta = 0.01 / N * np.random.uniform(0.1, 1.0) * (1 if np.random.rand() < 0.5 else -1)
            
            # Ensure non-negativity
            if h[j] + delta < 0:
                delta = -h[j] + 1e-9 # Push to 0? Or just skip?
                # Better: clamp delta so h[j] >= 0
                if delta < 0:
                    delta = -h[j]
                else:
                    continue # If we want to add to j, but j is 0? No, we add to i, subtract from j.
                    # Wait, h[j] changes by -delta? 
                    # Let's define: h[i] += delta, h[j] -= delta.
                    # If delta > 0, h[j] decreases. Need h[j] >= delta.
                    # If delta < 0, h[i] decreases. Need h[i] >= -delta.
            
            # Let's simplify: 
            # Pick amount to move from j to i.
            # amount = np.random.uniform(0, min(h[j], 0.01/N))
            # h[i] += amount
            # h[j] -= amount
            
            amount = np.random.uniform(0, min(h[j], 0.05/N))
            if amount < 1e-12:
                continue
                
            h[i] += amount
            h[j] -= amount
            
            # Calculate new score
            new_score = _score(h)
            
            # Acceptance
            if new_score < current_score:
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_h = h.copy()
            else:
                # Metropolis criterion
                # delta_E = new_score - current_score
                # prob = exp(-delta_E / T)
                # To avoid overflow/underflow, compare log(prob) with log(random)
                # log(prob) = -delta_E / T
                # log(rand)
                if - (new_score - current_score) / T > np.log(np.random.rand()):
                    current_score = new_score
                    # Do not update best_local_h unless it's actually better? 
                    # Usually we track best seen.
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_h = h.copy()
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
                
            it += 1
            # Safety break
            if it > 20000:
                break
                
        if best_local_score < best_score:
            best_score = best_local_score
            best_h = best_local_h.copy()
            
    if best_h is None:
        return (np.ones(N) / N).tolist()
        
    return best_h.tolist()
