# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5),
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1),
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    
    # Precompute edge masks for CW and CCW routing for each demand
    cw_masks = np.zeros((D, N))
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            edges = range(s, t)
        else:
            edges = list(range(s, N)) + list(range(0, t))
        for e in edges:
            cw_masks[i, e] = 1
    
    ccw_masks = 1.0 - cw_masks
    weights = np.array([d[2] for d in demands])
    
    # Precompute base_load and weighted_delta for fast evaluation
    base_load = (cw_masks * weights[:, None]).sum(axis=0)  # shape (N,)
    weighted_delta = (ccw_masks - cw_masks) * weights[:, None]  # shape (D, N)
    
    def evaluate(routing):
        """Compute max edge load given routing vector (0=CW, 1=CCW)."""
        total_load = base_load + (weighted_delta * routing[:, None]).sum(axis=0)
        return np.max(total_load)
    
    def evaluate_loads(routing):
        """Return full load vector."""
        return base_load + (weighted_delta * routing[:, None]).sum(axis=0)
    
    # Simulated annealing
    best_routing = np.zeros(D, dtype=np.float64)
    best_score = evaluate(best_routing)
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with simulated annealing
    for restart in range(200):
        if time.time() > deadline - 20:
            break
        
        # Start from best found so far or random
        if restart == 0:
            routing = np.zeros(D, dtype=np.float64)
        elif restart % 5 == 0:
            # Occasionally start from best to refine
            routing = best_routing.copy()
        else:
            routing = rng.random(D).round()
        
        current_score = evaluate(routing)
        if current_score < best_score:
            best_score = current_score
            best_routing = routing.copy()
        
        # Simulated annealing with adaptive cooling
        T = 3.0
        T_min = 0.005
        alpha = 0.9998
        max_steps = 20000
        no_improve_count = 0
        patience = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 20:
                break
            
            # Pick a random demand to flip
            d = rng.integers(0, D)
            old_val = routing[d]
            routing[d] = 1.0 - old_val
            
            new_score = evaluate(routing)
            delta = new_score - current_score
            
            if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_routing = routing.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                routing[d] = old_val
                no_improve_count += 1
            
            T *= alpha
            if T < T_min:
                break
            if no_improve_count > patience:
                break
    
    # Final polish: try 1, 2, and 3-flip moves
    def polish(routing, max_flip=3):
        """Try all combinations of up to max_flip flips and return best."""
        best_r = routing.copy()
        best_s = evaluate(routing)
        
        # Single flips
        for i in range(D):
            if time.time() > deadline - 10:
                break
            trial = best_r.copy()
            trial[i] = 1.0 - trial[i]
            s = evaluate(trial)
            if s < best_s - 1e-10:
                best_s = s
                best_r = trial
        
        # Double flips
        for i in range(D):
            if time.time() > deadline - 10:
                break
            for j in range(i+1, D):
                trial = best_r.copy()
                trial[i] = 1.0 - trial[i]
                trial[j] = 1.0 - trial[j]
                s = evaluate(trial)
                if s < best_s - 1e-10:
                    best_s = s
                    best_r = trial
        
        # Triple flips
        if max_flip >= 3:
            for i in range(D):
                if time.time() > deadline - 10:
                    break
                for j in range(i+1, D):
                    if time.time() > deadline - 10:
                        break
                    for k in range(j+1, D):
                        trial = best_r.copy()
                        trial[i] = 1.0 - trial[i]
                        trial[j] = 1.0 - trial[j]
                        trial[k] = 1.0 - trial[k]
                        s = evaluate(trial)
                        if s < best_s - 1e-10:
                            best_s = s
                            best_r = trial
        
        return best_r, best_s
    
    # Multiple polish passes to ensure convergence
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        best_routing, best_score = polish(best_routing, max_flip=3)
    
    # Return: >= 0.5 means CCW, < 0.5 means CW
    return best_routing

# EVOLVE-BLOCK-END
