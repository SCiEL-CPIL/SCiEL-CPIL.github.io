import math
import random

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    
    def score_of(U):
        U = sorted(set(U))
        if not U or U[0] != 0:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0.0
        return 1 + math.log(ratio) / denom
    
    random.seed(12345)
    
    best_set = [0, 1, 3]
    best_score = score_of(best_set)
    
    # Simulated annealing with multiple restarts
    for trial in range(30):
        n = random.randint(3, 15)
        max_val = random.randint(8, 80)
        
        # Initialize: random set
        current = [0]
        while len(current) < n:
            x = random.randint(1, max_val)
            if x not in current:
                current.append(x)
        current = sorted(current)
        current_score = score_of(current)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current[:]
        
        # Simulated annealing
        T = 0.3
        cooling = 0.997
        for it in range(5000):
            T *= cooling
            if T < 1e-7:
                break
            
            # Generate neighbor
            r = random.random()
            if r < 0.35:
                # Add
                x = random.randint(1, max_val + 5)
                if x not in current:
                    new_set = sorted(current + [x])
                else:
                    continue
            elif r < 0.65:
                # Remove
                if len(current) > 2:
                    idx = random.randint(1, len(current) - 1)
                    new_set = current[:idx] + current[idx+1:]
                else:
                    continue
            else:
                # Replace
                if len(current) > 2:
                    idx = random.randint(1, len(current) - 1)
                    old = current[idx]
                    x = random.randint(1, max_val + 5)
                    if x == old or x in current:
                        continue
                    new_set = sorted(current[:idx] + [x] + current[idx+1:])
                else:
                    continue
            
            new_score = score_of(new_set)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current[:]
    
    # Final local search polish
    current = best_set[:]
    current_score = score_of(current)
    improved = True
    while improved:
        improved = False
        # Try adding
        for x in range(1, max(current) + 10):
            if x not in current:
                new_set = sorted(current + [x])
                s = score_of(new_set)
                if s > current_score + 1e-12:
                    current = new_set
                    current_score = s
                    improved = True
                    break
        if improved:
            continue
        # Try removing
        for i in range(1, len(current)):
            new_set = current[:i] + current[i+1:]
            if len(new_set) < 2:
                continue
            s = score_of(new_set)
            if s > current_score + 1e-12:
                current = new_set
                current_score = s
                improved = True
                break
        if improved:
            continue
        # Try replacing
        for i in range(1, len(current)):
            old = current[i]
            for x in range(1, max(current) + 10):
                if x == old or x in current:
                    continue
                new_set = sorted(current[:i] + [x] + current[i+1:])
                s = score_of(new_set)
                if s > current_score + 1e-12:
                    current = new_set
                    current_score = s
                    improved = True
                    break
            if improved:
                break
    
    if current_score > best_score:
        best_set = current
    
    return sorted(best_set)
