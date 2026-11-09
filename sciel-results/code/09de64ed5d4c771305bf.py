"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import log, exp

def _compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for i, a in enumerate(U):
        for b in U[i:]:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def _sa_search(deadline):
    """Simulated annealing search for optimal set U."""
    # Start with known good structure
    best_U = [0, 1, 4, 9]
    best_score = _compute_score(best_U)
    
    current_U = list(best_U)
    current_score = best_score
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.001
    cooling = 0.995
    
    T = T_start
    step = 0
    max_steps = 50000
    
    while time.time() < deadline and step < max_steps and T > T_end:
        step += 1
        
        # Generate perturbation
        U = list(current_U)
        pert_type = np.random.randint(0, 4)
        
        if pert_type == 0:  # Add an element
            max_u = max(U) if U else 0
            new_val = np.random.randint(1, max_u + 10)
            if new_val not in U:
                U.append(int(new_val))
        elif pert_type == 1:  # Remove an element (not 0)
            if len(U) > 1:
                idx = np.random.randint(1, len(U))
                U.pop(idx)
        elif pert_type == 2:  # Shift a non-zero element
            if len(U) > 1:
                idx = np.random.randint(1, len(U))
                U[idx] += np.random.randint(-3, 4)
                if U[idx] < 0:
                    U[idx] = 0
                # Remove duplicates
                U = list(set(U))
                if 0 not in U:
                    U.append(0)
                U = sorted(U)
        else:  # Randomize one element
            if len(U) > 1:
                idx = np.random.randint(1, len(U))
                max_u = max(U)
                U[idx] = int(np.random.randint(1, max_u + 5))
                U = list(set(U))
                if 0 not in U:
                    U.append(0)
                U = sorted(U)
        
        if not U or 0 not in U:
            continue
        
        new_score = _compute_score(U)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score >= current_score:
            current_U = U
            current_score = new_score
            if new_score > best_score:
                best_U = list(U)
                best_score = new_score
        else:
            delta = current_score - new_score
            if delta < 100 * T:  # Avoid overflow
                if np.random.random() < exp(-delta / T):
                    current_U = U
                    current_score = new_score
        
        T *= cooling
    
    return best_U, best_score

def construct():
    # Deadline: 120s limit, use 100s for safety
    deadline = time.time() + 100
    
    # Try simulated annealing
    best_U, best_score = _sa_search(deadline)
    
    # Verify and return
    U = sorted(set(best_U))
    if 0 not in U:
        U = [0] + U
    if not U or max(U) == 0:
        return [0, 1, 4, 9]
    
    return U
# EVOLVE-BLOCK-END
