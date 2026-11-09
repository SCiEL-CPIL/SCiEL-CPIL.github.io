"""
Simulated Annealing approach for constructing truncated perfect difference sets.
Strategy:
1. Start with a valid Sidon set containing 0 and differences 1..30 (generated via fast greedy).
2. Use Simulated Annealing to minimize the maximum element.
3. Moves: Add random element, Remove random element (if valid), Swap random element.
4. Use bitmask for difference tracking to allow O(1) validation of Sidon property for small sets.
"""
import time
import random
import numpy as np

def _is_sidon(set_a):
    """Check if all positive differences are distinct."""
    n = len(set_a)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = set_a[j] - set_a[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _has_all_diffs(set_a, max_d=30):
    """Check if all d in 1..max_d occur as differences."""
    diffs = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            d = set_a[j] - set_a[i]
            diffs.add(d)
    return all(d in diffs for d in range(1, max_d + 1))

def _score(set_a):
    """Score: minimize max element. Return float('inf') if invalid."""
    if not _is_sidon(set_a):
        return float('inf')
    if not _has_all_diffs(set_a):
        return float('inf')
    return max(set_a)

def _greedy_construct(seed=0):
    """Fast greedy construction to get a valid starting point."""
    rng = random.Random(seed)
    a = [0]
    
    def diffs_of(s):
        s_sorted = sorted(s)
        return [y - x for i, x in enumerate(s_sorted) for y in s_sorted[i + 1:]]
    
    for d in range(1, 31):
        if d in diffs_of(a):
            continue
        # Try adding one element
        candidates = set()
        for v in a:
            candidates.add(v + d)
            candidates.add(v - d)
        candidates = [x for x in candidates if x >= 0 and x not in a]
        rng.shuffle(candidates)
        found = False
        for x in candidates:
            trial = sorted(a + [x])
            if _is_sidon(trial):
                a = trial
                found = True
                break
        if not found:
            # Try adding two elements
            for _ in range(100):
                y = max(a) + rng.randint(1, 50)
                trial = sorted(a + [y, y + d])
                if _is_sidon(trial):
                    a = trial
                    found = True
                    break
            if not found:
                # Fallback: large gap
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if _is_sidon(trial):
                        a = trial
                        break
                    y += 1
    return a

def _sa_construct(seed=0, time_limit=115.0):
    """
    Simulated Annealing to minimize the maximum element of a Sidon set containing 0 and diffs 1..30.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initial state
    current_set = _greedy_construct(seed)
    current_max = max(current_set)
    
    # Track best found
    best_set = list(current_set)
    best_max = current_max
    
    # Temperature parameters
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Precompute differences for current set for fast checking? 
    # Actually, for SA, we need to check validity of neighbors.
    # A neighbor is valid if it's Sidon and has all diffs 1..30.
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        move_type = rng.random()
        
        # 1. Try to remove an element (if size > 31 or if removing doesn't break diffs)
        # Actually, we want to minimize max, so we should try to replace the max element with something smaller,
        # or remove large elements if possible (but we must keep diffs 1..30).
        
        # Strategy: 
        # - Swap: Replace a random element (not 0) with a random new element.
        # - Add: Add a random element (might increase max, but could allow future reductions? Unlikely if we want min max).
        # - Remove: Remove a random element (if valid).
        
        # Let's focus on Swap and Remove.
        
        if len(current_set) > 1 and move_type < 0.5:
            # Swap
            idx = rng.randint(1, len(current_set) - 1) # Don't swap 0
            old_val = current_set[idx]
            
            # New value: try to be smaller than current max, or random in range
            # Bias towards smaller values to reduce max
            if rng.random() < 0.7:
                new_val = rng.randint(1, max(1, current_max - 1))
            else:
                new_val = rng.randint(1, current_max + 50)
            
            if new_val in current_set:
                continue
                
            new_set = list(current_set)
            new_set[idx] = new_val
            new_set = sorted(new_set)
            
            # Check validity
            if not _is_sidon(new_set):
                continue
            if not _has_all_diffs(new_set):
                continue
            
            new_max = max(new_set)
            
        elif move_type < 0.8:
            # Remove
            if len(current_set) <= 31: # Keep size reasonable? Task says "at most 32".
                # If we remove, we might lose a diff.
                idx = rng.randint(1, len(current_set) - 1)
                old_val = current_set[idx]
                new_set = [x for x in current_set if x != old_val]
                new_set = sorted(new_set)
                
                if not _is_sidon(new_set):
                    continue
                if not _has_all_diffs(new_set):
                    continue
                
                new_max = max(new_set)
                
        else:
            # Add (rarely useful for minimizing max, but helps if stuck)
            new_val = rng.randint(1, current_max + 100)
            if new_val in current_set:
                continue
            new_set = sorted(current_set + [new_val])
            if len(new_set) > 32:
                continue
            if not _is_sidon(new_set):
                continue
            if not _has_all_diffs(new_set):
                continue
            new_max = max(new_set)

        # Calculate delta
        delta = new_max - current_max
        
        # Acceptance probability
        if delta < 0:
            # Improvement
            current_set = new_set
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_set = list(current_set)
        else:
            # Worse
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_set = new_set
                    current_max = new_max
        
        # Cool down
        T *= cooling_rate
        
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 118.0 # Safety margin
    
    best_set = None
    best_score = float('inf')
    
    # Run multiple SA instances with different seeds
    # SA is stochastic, so multiple runs are good.
    # Each run should be short enough to allow many restarts.
    # Let's allocate ~5-10 seconds per run.
    
    seed = 0
    while time.time() < deadline - 5.0:
        # Run SA for a specific duration
        run_deadline = min(deadline - 2.0, time.time() + 10.0)
        time_limit = run_deadline - time.time()
        
        if time_limit < 1.0:
            break
            
        candidate = _sa_construct(seed, time_limit=time_limit)
        s = _score(candidate)
        
        if s < best_score:
            best_score = s
            best_set = candidate
            
        seed += 1
        
    # If we didn't find anything, fallback to greedy
    if best_set is None or best_score == float('inf'):
        best_set = _greedy_construct(0)
        
    return best_set

# EVOLVE-BLOCK-END
