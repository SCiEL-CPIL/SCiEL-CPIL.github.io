# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares).

    Uses Simulated Annealing with random restarts to find large independent sets in the
    conflict graph where edges connect elements with forbidden differences.
    """
    m = 205
    deadline = time.time() + 600  # 600 second deadline with margin
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency: conflict[i] is set of j such that (i-j) mod m is forbidden
    conflict = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                conflict[i].add(j)
    
    def is_valid_subset(subset):
        """Check if subset is a valid independent set."""
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                d1 = (subset[i] - subset[j]) % m
                if d1 in forbidden:
                    return False
        return True
    
    def greedy_init_random(rng):
        """Randomized greedy initialization."""
        order = list(range(m))
        rng.shuffle(order)
        selected = []
        for candidate in order:
            ok = True
            for s in selected:
                d1 = (candidate - s) % m
                if d1 in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected
    
    def simulated_annealing(initial_subset, rng, max_iters=500000, start_temp=1.0, end_temp=0.001):
        """Simulated annealing to improve the subset."""
        current = list(initial_subset)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        
        # Precompute for fast conflict checking
        current_set = set(current)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
            
            # Random move: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add a random element not in current
                candidates = [x for x in range(m) if x not in current_set]
                if not candidates:
                    continue
                x = rng.choice(candidates)
                # Check if adding x is valid
                valid = True
                for s in current:
                    d = (x - s) % m
                    if d in forbidden:
                        valid = False
                        break
                if valid:
                    current.append(x)
                    current_set.add(x)
                    new_size = len(current)
                    if new_size > best_size:
                        best = list(current)
                        best_size = new_size
                # If invalid, we could still accept with probability, but for now skip
                
            elif move_type < 0.7:
                # Try to remove a random element
                if not current:
                    continue
                idx = rng.randrange(len(current))
                x = current[idx]
                current.pop(idx)
                current_set.remove(x)
                new_size = len(current)
                # Accept removal with probability based on temperature (always accept if it helps later)
                # Actually, removal decreases size, so only accept if we think it leads to better solution
                # For simplicity, accept removal with some probability
                if rng.random() > temp * 0.5:
                    # Reject, put back
                    current.append(x)
                    current_set.add(x)
                    
            else:
                # Swap: remove one, try to add another
                if not current:
                    continue
                idx = rng.randrange(len(current))
                x_remove = current[idx]
                candidates = [x for x in range(m) if x not in current_set and x != x_remove]
                if not candidates:
                    continue
                x_add = rng.choice(candidates)
                
                # Check if after removing x_remove and adding x_add, the set is valid
                temp_current = [s for i, s in enumerate(current) if i != idx]
                valid = True
                for s in temp_current:
                    d = (x_add - s) % m
                    if d in forbidden:
                        valid = False
                        break
                if valid:
                    current[idx] = x_add
                    current_set.remove(x_remove)
                    current_set.add(x_add)
                    new_size = len(current)
                    if new_size > best_size:
                        best = list(current)
                        best_size = new_size
        
        return best
    
    rng = random.Random(42)
    
    best_overall = []
    best_size = 0
    
    # Multiple restarts
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        initial = greedy_init_random(rng)
        if len(initial) > best_size:
            best_size = len(initial)
            best_overall = list(initial)
        
        # Simulated annealing
        improved = simulated_annealing(initial, rng, max_iters=200000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_overall = list(improved)
        
        # Also try to improve best_overall directly
        if len(best_overall) > 0:
            improved2 = simulated_annealing(best_overall, rng, max_iters=100000)
            if len(improved2) > best_size:
                best_size = len(improved2)
                best_overall = list(improved2)
    
    # Final verification
    if not is_valid_subset(best_overall):
        # Fallback to a known valid solution
        best_overall = [0]
    
    return np.array(best_overall, dtype=np.int64)


# EVOLVE-BLOCK-END
