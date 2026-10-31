"""New program: Simulated Annealing / Enhanced Local Search with larger range."""
import math
import random
import numpy as np

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
    # Include geometric, arithmetic, and mixed
    candidates = []
    for k in range(3, 10):
        candidates.append([0] + [2**i - 1 for i in range(1, k)])
        candidates.append([0] + [i for i in range(1, k)])
        
    # Random seeds with larger range
    random.seed(42)
    for _ in range(50):
        size = random.randint(4, 12)
        max_val = random.choice([100, 500, 2000, 10000])
        vals = sorted(random.sample(range(1, max_val), size - 1))
        candidates.append([0] + vals)
        
    # Enhanced Local Search
    for start in candidates:
        current = list(start)
        current_score = score_of(current)
        
        # Simulated Annealing / Hill Climbing with restarts
        # We will use a simple hill climbing with random restarts of the search space
        # and a larger max_val
        
        max_val = 20000
        improved = True
        iterations = 0
        max_iterations = 5000
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # 1. Try adding a new element
            # Randomly sample a few candidates for addition to avoid O(N*MaxVal)
            # But N is small, so we can iterate over a range or sample.
            # Let's sample 100 random values from 1 to max_val
            for _ in range(50):
                x = random.randint(1, max_val)
                if x not in current:
                    new_set = current + [x]
                    s = score_of(new_set)
                    if s > current_score + 1e-12:
                        current = new_set
                        current_score = s
                        improved = True
                        break
            
            if improved: continue
            
            # 2. Try removing an element
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
            
            # 3. Try replacing an element with a random value
            for i in range(1, len(current)):
                old = current[i]
                # Try a few random replacements
                for _ in range(10):
                    x = random.randint(1, max_val)
                    if x == old or x in current: continue
                    new_set = current[:i] + [x] + current[i+1:]
                    s = score_of(new_set)
                    if s > current_score + 1e-12:
                        current = new_set
                        current_score = s
                        improved = True
                        break
                if improved: break
            
            # 4. Try shifting the non-zero elements
            # U' = {0} U {u + d for u in U \ {0}}
            # This changes the range significantly.
            if len(current) > 2:
                d = random.randint(-100, 100)
                if d != 0:
                    new_non_zero = [u + d for u in current[1:]]
                    # Check if valid (non-negative, distinct, no 0 collision)
                    valid = True
                    if 0 in new_non_zero:
                        valid = False
                    for v in new_non_zero:
                        if v < 0 or v in current:
                            valid = False
                            break
                    if valid:
                        new_set = [0] + sorted(new_non_zero)
                        s = score_of(new_set)
                        if s > current_score + 1e-12:
                            current = new_set
                            current_score = s
                            improved = True

        if current_score > best_score:
            best_score = current_score
            best_set = current
            
    return sorted(best_set)
