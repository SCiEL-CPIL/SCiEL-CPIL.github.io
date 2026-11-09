"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_objective(f):
    """Compute max |autoconv| / (sum f)^2"""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.inf
    # Full linear autoconvolution
    ac = np.convolve(f, f, mode='full')
    return np.max(np.abs(ac)) / (S * S)


def _local_search(f, deadline, max_iters=50000):
    """Local search: try flipping signs and small perturbations"""
    best = f.copy()
    best_obj = _compute_objective(best)
    n = len(f)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Try flipping a random sign
        idx = np.random.randint(n)
        trial = best.copy()
        trial[idx] = -trial[idx]
        obj = _compute_objective(trial)
        if obj < best_obj:
            best = trial
            best_obj = obj
        # Try small perturbation
        if it % 3 == 0:
            idx2 = np.random.randint(n)
            trial2 = best.copy()
            trial2[idx2] += np.random.uniform(-0.5, 0.5)
            obj2 = _compute_objective(trial2)
            if obj2 < best_obj:
                best = trial2
                best_obj = obj2
    
    return best, best_obj


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    # Start from all ones
    f = np.ones(N)
    best_f = f.copy()
    best_obj = _compute_objective(best_f)
    
    # Multiple restarts with different random initializations
    np.random.seed(42)
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            f = np.ones(N)
        else:
            # Random signed sequence
            f = np.random.choice([-1.0, 1.0], size=N)
            # Ensure sum is positive
            if np.sum(f) < 0:
                f = -f
        
        f_opt, obj = _local_search(f, deadline, max_iters=20000)
        if obj < best_obj:
            best_f = f_opt
            best_obj = obj
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 5.0:
        f_polish = best_f.copy()
        obj_polish = best_obj
        for _ in range(10000):
            if time.time() > deadline - 2.0:
                break
            idx = np.random.randint(N)
            trial = f_polish.copy()
            trial[idx] += np.random.normal(0, 0.1)
            obj = _compute_objective(trial)
            if obj < obj_polish:
                f_polish = trial
                obj_polish = obj
        if obj_polish < best_obj:
            best_f = f_polish
            best_obj = obj_polish
    
    return best_f
# EVOLVE-BLOCK-END
