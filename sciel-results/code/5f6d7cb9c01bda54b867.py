"""Rewritten program: search for optimal finite sets maximizing the sumset/difference-set exponent lower bound."""
import math
import random
import numpy as np

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

    best_set = [0, 1, 3]
    best_score = score_of(best_set)
    
    # Candidate seed sets
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 5, 10],
        [0, 1, 4, 6, 9],
    ]
    
    # Add random seeds
    random.seed(42)
    for _ in range(30):
        size = random.randint(3, 8)
        vals = sorted(random.sample(range(1, 150), size - 1))
        candidates.append([0] + vals)
        
    for start in candidates:
        current = list(start)
        current_score = score_of(current)
        
        # Local Search
        improved = True
        max_val = 150
        while improved:
            improved = False
            # Try adding
            for x in range(1, max_val + 1):
                if x not in current:
                    new_set = current + [x]
                    s = score_of(new_set)
                    if s > current_score + 1e-12:
                        current = new_set
                        current_score = s
                        improved = True
                        break
            if improved: continue
            
            # Try removing
            for i in range(1, len(current)):
                new_set = current[:i] + current[i+1:]
                if len(new_set) < 2: continue
                s = score_of(new_set)
                if s > current_score + 1e-12:
                    current = new_set
                    current_score = s
                    improved = True
                    break
            if improved: continue
            
            # Try replacing
            for i in range(1, len(current)):
                old = current[i]
                for x in range(1, max_val + 1):
                    if x == old or x in current: continue
                    new_set = current[:i] + [x] + current[i+1:]
                    s = score_of(new_set)
                    if s > current_score + 1e-12:
                        current = new_set
                        current_score = s
                        improved = True
                        break
                if improved: break
                
        if current_score > best_score:
            best_score = current_score
            best_set = current
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
