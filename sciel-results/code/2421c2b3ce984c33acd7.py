import time
import math
import numpy as np
import random

def fast_score(U):
    """
    Computes the score using NumPy for speed.
    Score: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    if len(U) < 2:
        return 0.0
    
    U = list(set(U))
    if 0 not in U:
        U.append(0)
        
    maxU = max(U)
    if maxU == 0:
        return 0.0
    
    # Use boolean arrays for efficiency if maxU is reasonable
    # If maxU is very large, fall back to sets
    if maxU > 50000:
        # Fallback for very sparse sets
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        len_sum = len(sum_set)
        len_diff = len(diff_set)
    else:
        # Bitmask approach
        # U+U is in [0, 2*maxU]
        # U-U is in [-maxU, maxU]
        
        # Create mask for U
        # We can use numpy convolve or just shifts.
        # Shifts are faster for sparse-ish sets if we use integers?
        # Let's stick to sets for simplicity if maxU is not huge, 
        # but the prompt says NumPy is available.
        # Let's try a hybrid: if N is small, sets are fine. 
        # The bottleneck in the previous code was likely the search strategy, not just speed.
        # However, fast scoring allows more restarts.
        
        # Let's use the set method but optimize the loop?
        # Actually, for N < 200, sets are fine.
        # Let's just use sets but ensure we don't do redundant work.
        
        sum_set = set()
        diff_set = set()
        # Optimization: sort U
        U.sort()
        for i, a in enumerate(U):
            for b in U[i:]: # Sums are commutative, but we need all pairs for set size? 
                             # No, a+b is same as b+a. 
                             # But for diff, a-b != b-a.
                             # So we need all pairs for diff.
                             # For sum, we can do i <= j.
                sum_set.add(a + b)
                diff_set.add(a - b)
                if i != j:
                    diff_set.add(b - a)
        
        len_sum = len(sum_set)
        len_diff = len(diff_set)
        
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    
    if log_denom == 0:
        return 0.0
        
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_score = -1
    best_U = [0, 1, 3] # Default seed
    
    # Strategy: Randomized Local Search with Restart
    # We explore different "shapes" of sets.
    # 1. Dense sets (intervals)
    # 2. Semi-dense (intervals with gaps)
    # 3. Sparse (random selection)
    
    # Helper to generate a random set
    def generate_random_set(n, max_val):
        # Always include 0
        # Randomly pick n-1 other numbers
        if n <= 1:
            return [0]
        # Ensure 0 is included
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        chosen = candidates[:n-1]
        return [0] + sorted(chosen)
    
    # Helper for local search
    def local_search(initial_U, max_iters, timeout_check):
        current_U = initial_U[:]
        current_score = fast_score(current_U)
        
        for _ in range(max_iters):
            if time.time() > timeout_check:
                break
                
            # Choose a move: Add, Remove, or Swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_U = current_U[:]
            max_val = max(new_U) if new_U else 1
            
            if move_type == 'add':
                # Add a number near the current max or in a gap
                # Bias towards extending the range or filling gaps
                if random.random() < 0.5:
                    # Extend range
                    new_val = random.randint(max_val, max_val * 2 + 10)
                else:
                    # Fill gap
                    new_val = random.randint(0, max_val)
                
                if new_val not in new_U:
                    new_U.append(new_val)
                    new_U = list(set(new_U))
            elif move_type == 'remove':
                if len(new_U) > 2:
                    # Don't remove 0
                    removable = [x for x in new_U if x != 0]
                    if removable:
                        idx_to_remove = random.choice(removable)
                        new_U.remove(idx_to_remove)
            elif move_type == 'swap':
                if len(new_U) > 2:
                    # Remove one, add another
                    removable = [x for x in new_U if x != 0]
                    if removable:
                        idx_to_remove = random.choice(removable)
                        new_U.remove(idx_to_remove)
                        new_val = random.randint(0, max_val * 2)
                        if new_val not in new_U:
                            new_U.append(new_val)
                            new_U = list(set(new_U))
            
            if len(new_U) == len(current_U) and new_U == current_U:
                continue
                
            new_score = fast_score(new_U)
            
            if new_score > current_score:
                current_score = new_score
                current_U = new_U
            else:
                # With some probability, accept worse moves (Simulated Annealing style)
                # But simple hill climbing is often sufficient for this metric if restarts are used
                pass
                
        return current_U, current_score
    
    # Restart loop
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        
        # Vary the parameters for each restart
        # Size N between 5 and 100
        # Max value M between 10 and 10000
        
        n = random.randint(5, 100)
        # Correlate N and M. 
        # If N is large, M should be large enough to allow distinct sums/diffs.
        # If N is small, M can be smaller.
        min_m = n
        max_m = n * 100 + 100
        m = random.randint(min_m, max_m)
        
        # Initial set
        initial_U = generate_random_set(n, m)
        
        # Run local search
        # Budget per restart: ~1000 iterations or 1 second
        max_iters = 500
        u, s = local_search(initial_U, max_iters, deadline - 5)
        
        if s > best_score:
            best_score = s
            best_U = u
            
    return best_U

if __name__ == "__main__":
    # Quick self-test
    # import sys
    # print(construct())
    pass
