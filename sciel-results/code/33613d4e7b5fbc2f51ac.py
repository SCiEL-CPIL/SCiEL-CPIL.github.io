"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # Filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict matrix using NumPy for speed
    # conflict_matrix[i][j] = 1 if valid_numbers[i] + valid_numbers[j] is a perfect square (i != j)
    conflict_matrix = np.zeros((num_valid, num_valid), dtype=np.uint8)
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                conflict_matrix[i][j] = 1
    
    # Convert to sets for faster membership tests in greedy
    conflict_sets = [set(np.where(conflict_matrix[i] == 1)[0].tolist()) for i in range(num_valid)]
    
    def greedy_min_degree_randomized(seed=None):
        if seed is not None:
            random.seed(seed)
        
        remaining = set(range(num_valid))
        selected = []
        
        while remaining:
            # Find elements with minimum degree among remaining
            min_deg = float('inf')
            candidates = []
            for idx in remaining:
                deg = len(conflict_sets[idx] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [idx]
                elif deg == min_deg:
                    candidates.append(idx)
            
            # Pick randomly among min-degree candidates
            best_idx = random.choice(candidates)
            selected.append(best_idx)
            remaining.discard(best_idx)
            for neighbor in conflict_sets[best_idx]:
                remaining.discard(neighbor)
        
        return [valid_numbers[i] for i in selected]
    
    def is_valid_set(s):
        """Check if a set is valid (no pair sums to a perfect square)."""
        s_set = set(s)
        for a in s_set:
            if 2 * a in squares:
                return False
            for b in s_set:
                if a <= b and (a + b) in squares:
                    return False
        return True
    
    # Run multiple randomized restarts
    best_result = None
    best_size = 0
    num_restarts = 0
    
    while time.time() < deadline:
        result = greedy_min_degree_randomized()
        result_size = len(result)
        num_restarts += 1
        
        if result_size > best_size:
            best_size = result_size
            best_result = result
        
        # Early stop if we find a very large set
        if best_size >= 120:
            break
    
    # Local search improvement on the best result
    if best_result is not None and time.time() < deadline - 5:
        current = set(best_result)
        current_size = len(current)
        improved = False
        
        # Try 1-swap local search: remove one, try to add another
        for to_remove in list(current):
            if time.time() > deadline - 2:
                break
            
            trial_set = current - {to_remove}
            # Try adding each valid number not in trial_set
            for candidate in valid_numbers:
                if candidate in trial_set:
                    continue
                # Check if candidate can be added
                can_add = True
                for s in squares:
                    other = s - candidate
                    if other != candidate and other in trial_set:
                        can_add = False
                        break
                if can_add:
                    new_size = len(trial_set) + 1
                    if new_size > current_size:
                        current = trial_set | {candidate}
                        current_size = new_size
                        improved = True
                        break
        
        if improved:
            best_result = sorted(current)
            best_size = current_size
    
    # Final validation
    if best_result is None:
        # Fallback
        fallback = [a for a in range(1, 257) if 2 * a not in squares]
        # Greedy filter
        result = []
        for a in fallback:
            ok = True
            for b in result:
                if (a + b) in squares:
                    ok = False
                    break
            if ok:
                result.append(a)
        return sorted(result)
    
    # Verify
    if not is_valid_set(best_result):
        # Rebuild from scratch with validation
        result = []
        for a in sorted(valid_numbers):
            ok = True
            for b in result:
                if (a + b) in squares:
                    ok = False
                    break
            if ok:
                result.append(a)
        return sorted(result)
    
    return sorted(best_result)
# EVOLVE-BLOCK-END
