# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses Simulated Annealing with multiple restarts and a deadline guard.
    """
    m = 205
    k = 2
    
    # Compute forbidden differences (nonzero quadratic residues mod 205)
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # Precompute conflict matrix: conflict[i][j] is True if (i-j) % m or (j-i) % m is in forbidden
    # i.e., if i and j cannot both be in the set
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            if d1 in forbidden:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Deadline guard: 630s limit, leave 30s safety margin
    start_time = time.time()
    deadline = start_time + 600.0
    
    def is_valid_solution(mask):
        """Check if mask (boolean array of length m) represents a valid solution."""
        selected = np.where(mask)[0]
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a, b = selected[i], selected[j]
                if conflict[a, b]:
                    return False
        return True
    
    def objective(mask):
        """
        Objective function for SA: minimize conflicts.
        Returns number of conflicting pairs in the current mask.
        """
        count = 0
        selected = np.where(mask)[0]
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if conflict[selected[i], selected[j]]:
                    count += 1
        return count
    
    def greedy_init(rng):
        """Greedy initialization: add elements in random order if no conflict."""
        mask = np.zeros(m, dtype=bool)
        candidates = list(range(m))
        rng.shuffle(candidates)
        for cand in candidates:
            # Check if cand conflicts with any already selected
            can_add = True
            for s in np.where(mask)[0]:
                if conflict[cand, s]:
                    can_add = False
                    break
            if can_add:
                mask[cand] = True
        return mask
    
    def simulated_annealing(rng, initial_mask, max_time=10.0):
        """Run simulated annealing starting from initial_mask."""
        mask = initial_mask.copy()
        current_obj = objective(mask)
        best_mask = mask.copy()
        best_obj = current_obj
        
        T_start = 1.0
        T_end = 0.01
        cooling_rate = 0.9995
        T = T_start
        sa_start = time.time()
        
        while T > T_end:
            if time.time() > deadline - 30.0:
                break
            if time.time() - sa_start > max_time:
                break
            
            # Choose a random element to flip
            idx = rng.randint(0, m - 1)
            
            # Compute delta in objective
            old_val = mask[idx]
            mask[idx] = not old_val
            
            # Recompute objective (could be optimized with incremental updates)
            new_obj = objective(mask)
            delta = new_obj - current_obj
            
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                # Accept move
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_mask = mask.copy()
            else:
                # Reject move
                mask[idx] = old_val
            
            T *= cooling_rate
        
        return best_mask, best_obj
    
    # Run multiple restarts
    best_mask_overall = np.zeros(m, dtype=bool)
    best_obj_overall = m * (m - 1)  # worst case
    
    # First, try a few greedy inits
    for restart in range(20):
        if time.time() > deadline - 30.0:
            break
        
        rng = random.Random(restart * 1000 + 42)
        init_mask = greedy_init(rng)
        init_obj = objective(init_mask)
        
        if init_obj == 0:
            # Already valid
            if np.sum(init_mask) > np.sum(best_mask_overall):
                best_mask_overall = init_mask
                best_obj_overall = 0
        
        # Run SA from this init
        sa_mask, sa_obj = simulated_annealing(rng, init_mask, max_time=5.0)
        if sa_obj < best_obj_overall:
            best_obj_overall = sa_obj
            best_mask_overall = sa_mask
    
    # If we still have conflicts, try to repair by removing conflicting elements
    if best_obj_overall > 0:
        mask = best_mask_overall.copy()
        # Greedy repair: remove one element from each conflicting pair
        changed = True
        while changed and time.time() < deadline - 30.0:
            changed = False
            selected = np.where(mask)[0]
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if conflict[selected[i], selected[j]]:
                        # Remove the one with higher index (arbitrary choice)
                        mask[selected[j]] = False
                        changed = True
                        break
                if changed:
                    break
        best_mask_overall = mask
        best_obj_overall = objective(mask)
    
    # Extract result
    result = np.where(best_mask_overall)[0].astype(np.int64)
    
    # Final validity check
    if not is_valid_solution(best_mask_overall):
        # Fallback to a known small valid set
        result = np.array([0], dtype=np.int64)
    
    return result


# EVOLVE-BLOCK-END
