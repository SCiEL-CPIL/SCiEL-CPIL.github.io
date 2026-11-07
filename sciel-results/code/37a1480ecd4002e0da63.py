import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(50):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Phase 1: Generate diverse combinatorial starting points
    # The "sawtooth" pattern (blocks of 1s and 0s) is known to be effective
    # for minimizing max correlation in this type of problem.
    
    candidates = []
    
    # 1. Pure alternating (period 2)
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    candidates.append(_project_to_feasible(h_alt))
    
    # 2. Block patterns with various periods
    for period in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25]:
        h_block = np.zeros(N)
        # Try different duty cycles within the period
        for duty in range(1, period):
            h_block = np.zeros(N)
            for i in range(N):
                if (i % period) < duty:
                    h_block[i] = 1.0
            # Ensure sum is N/2 by scaling or adjusting? 
            # No, we project. But pure 0/1 with sum != N/2 will be projected to 0/1-ish.
            # Actually, for 0/1 patterns, sum must be exactly N/2 to stay 0/1.
            # If sum(h_block) != N/2, projection will make it non-binary.
            # Let's only keep binary patterns that naturally have sum N/2 or close.
            if abs(np.sum(h_block) - N/2.0) < 1e-9:
                candidates.append(h_block.copy())
            else:
                # Project it to get a valid starting point (will be non-binary)
                candidates.append(_project_to_feasible(h_block))
                
    # 3. Random 0/1 patterns with correct sum
    rng = np.random.default_rng(42)
    for _ in range(20):
        h_rand = np.zeros(N)
        # Choose N/2 indices to be 1
        indices = rng.choice(N, size=N//2, replace=False)
        h_rand[indices] = 1.0
        candidates.append(h_rand.copy())
        
    # 4. Smooth sine waves (for gradient-based refinement starting points)
    for freq in [1, 2, 3, 5, 7, 10, 13]:
        h_sin = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        candidates.append(_project_to_feasible(h_sin))
        
    # 5. The seed (all 0.5)
    candidates.append(np.full(N, 0.5))
    
    best_h = candidates[0].copy()
    best_val = _correlation_value(best_h)
    
    # Phase 2: Stochastic Hill Climbing (Pair Swaps)
    # This is the core "different strategy". We don't use gradients.
    # We try to swap values between two positions i and j.
    # To maintain sum, if we change h[i] by d, we must change h[j] by -d.
    # Since we want to explore the 0/1 landscape effectively, we can try
    # discrete swaps (1,0) -> (0,1) or continuous small adjustments.
    # Let's do a mix: primarily discrete swaps of 0.5 units or full swaps,
    # but also small continuous perturbations.
    
    # We'll iterate through candidates and improve them.
    for h_start in candidates:
        if time.time() > deadline - 10.0:
            break
            
        h_curr = h_start.copy()
        val_curr = _correlation_value(h_curr)
        
        # Local search loop
        no_improve_count = 0
        max_no_improve = 50
        step_sizes = [0.1, 0.2, 0.5, 1.0]  # 1.0 allows 0<->1 flips
        
        while no_improve_count < max_no_improve:
            improved = False
            # Try random pairs
            for _ in range(100):  # 100 random attempts per iteration
                if time.time() > deadline - 5.0:
                    break
                
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i == j:
                    continue
                    
                d = step_sizes[rng.integers(0, len(step_sizes))]
                sign = 1.0 if rng.random() > 0.5 else -1.0
                
                new_i = h_curr[i] + sign * d
                new_j = h_curr[j] - sign * d
                
                if new_i < -1e-9 or new_i > 1 + 1e-9 or new_j < -1e-9 or new_j > 1 + 1e-9:
                    continue
                    
                new_i = np.clip(new_i, 0, 1)
                new_j = np.clip(new_j, 0, 1)
                
                # If clipping changed the value, the sum might be off.
                # We rely on the fact that small d keeps us in bounds usually.
                # If we clip, we should re-project, but that's expensive.
                # Let's just skip if clipping occurred significantly.
                if abs(new_i - (h_curr[i] + sign*d)) > 1e-9 or abs(new_j - (h_curr[j] - sign*d)) > 1e-9:
                    continue
                    
                h_curr[i] = new_i
                h_curr[j] = new_j
                
                new_val = _correlation_value(h_curr)
                
                if new_val < val_curr - 1e-12:
                    val_curr = new_val
                    improved = True
                    no_improve_count = 0
                else:
                    # Revert
                    h_curr[i] -= sign * d
                    h_curr[j] += sign * d
                    
            if not improved:
                no_improve_count += 1
                
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()
            
    # Phase 3: Gradient-based polish on the best combinatorial result
    # The combinatorial search finds a good "shape", but SLSQP can refine the boundaries.
    
    if time.time() < deadline - 5.0:
        try:
            def exact_obj(x):
                corr = np.correlate(x, 1 - x, mode="full")
                return np.max(corr)
            
            res = minimize(
                exact_obj,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 4: One final quick local search on the polished result
    if time.time() < deadline - 2.0:
        h_curr = best_h.copy()
        val_curr = best_val
        for _ in range(500):
            if time.time() > deadline - 1.0:
                break
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i == j: continue
            d = 0.01 * (1 if rng.random() > 0.5 else -1)
            ni = h_curr[i] + d
            nj = h_curr[j] - d
            if ni < 0 or ni > 1 or nj < 0 or nj > 1: continue
            h_curr[i] = ni
            h_curr[j] = nj
            nv = _correlation_value(h_curr)
            if nv < val_curr:
                val_curr = nv
            else:
                h_curr[i] -= d
                h_curr[j] += d
        
        if val_curr < best_val:
            best_val = val_curr
            best_h = h_curr.copy()

    return best_h
