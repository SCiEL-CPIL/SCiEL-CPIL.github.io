"""Rewritten program: search for optimal finite sets maximizing the sumset/difference-set exponent lower bound."""
import math
import random
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    
    def score_of(U):
        U = np.array(sorted(set(U)), dtype=int)
        if len(U) == 0 or U[0] != 0:
            return 0.0
        if len(U) == 1:
            return 1.0
        
        # Compute sumset and difference set sizes
        sums = U[:, None] + U[None, :]
        diffs = U[:, None] - U[None, :]
        
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
        
        if n_sums == 0:
            return 0.0
            
        ratio = n_diffs / n_sums
        if ratio <= 0:
            return 0.0
            
        max_u = U[-1]
        if max_u == 0:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0.0
            
        return 1 + math.log(ratio) / denom

    # Simulated Annealing Parameters
    start_temp = 0.1
    end_temp = 0.001
    cool_rate = 0.995
    max_val = 200
    max_set_size = 25
    
    def random_set():
        size = random.randint(2, 8)
        vals = sorted(random.sample(range(1, max_val), size - 1))
        return [0] + vals

    def move_set(current):
        """Perform a random move on the current set."""
        r = random.random()
        if r < 0.4 and len(current) < max_set_size:
            # Add a random element
            x = random.randint(1, max_val)
            if x not in current:
                return current + [x]
        elif r < 0.7 and len(current) > 2:
            # Remove a random element (not 0)
            idx = random.randint(1, len(current) - 1)
            return current[:idx] + current[idx+1:]
        elif r < 0.9 and len(current) > 2:
            # Replace a random element
            idx = random.randint(1, len(current) - 1)
            x = random.randint(1, max_val)
            if x != current[idx] and x not in current:
                new_set = current[:idx] + [x] + current[idx+1:]
                return sorted(new_set)
        else:
            # Swap or shift? Just try a random small change
            idx = random.randint(1, len(current) - 1)
            old = current[idx]
            delta = random.randint(-5, 5)
            x = old + delta
            if x > 0 and x != old and x not in current:
                new_set = current[:idx] + [x] + current[idx+1:]
                return sorted(new_set)
        
        return current # No change

    best_set = [0, 1, 3]
    best_score = score_of(best_set)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Run multiple restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        current = random_set()
        current_score = score_of(current)
        
        # If we start with a better score, great.
        if current_score > best_score:
            best_score = current_score
            best_set = current[:]
            
        temp = start_temp
        
        while temp > end_temp:
            if time.time() - start_time > time_limit:
                break
                
            new_set = move_set(current)
            if new_set == current:
                temp *= cool_rate
                continue
                
            new_score = score_of(new_set)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / temp):
                current = new_set
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current[:]
            
            temp *= cool_rate
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
