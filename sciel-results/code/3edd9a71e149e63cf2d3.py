import math
import time
import random

def _score(U):
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    if len(U) < 2:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    plus = set()
    minus = set()
    # Optimization: Use sets
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
            
    n_plus = len(plus)
    n_minus = len(minus)
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
        
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 2, 4, 5, 7, 8, 10], # Some random small ones
        list(range(10)),
        [0, 1, 3, 4, 6, 7, 9]
    ]
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # Local Search
    current_set = best_set[:]
    current_score = best_score
    
    # Heuristic: Try to expand or modify
    # We will try to add elements or swap elements
    
    while time.time() < deadline:
        # Generate a batch of neighbors
        improved = False
        for _ in range(50):
            if time.time() > deadline:
                break
                
            new_set = current_set[:]
            op = random.choice(['add', 'remove', 'replace', 'shift'])
            
            if op == 'add':
                # Add a number near the max or in a gap
                max_val = max(new_set)
                # Try numbers in [1, max_val + 5]
                x = random.randint(1, max_val + 5)
                if x not in new_set:
                    new_set.append(x)
            elif op == 'remove':
                if len(new_set) > 2:
                    # Don't remove 0
                    removable = [x for x in new_set if x != 0]
                    if removable:
                        x = random.choice(removable)
                        new_set.remove(x)
            elif op == 'replace':
                if len(new_set) > 1:
                    removable = [x for x in new_set if x != 0]
                    if removable:
                        x = random.choice(removable)
                        new_set.remove(x)
                        # Replace with x +/- delta
                        delta = random.choice([1, -1, 2, -2, 3])
                        y = x + delta
                        if y >= 0 and y not in new_set:
                            new_set.append(y)
                        else:
                            # If collision, try another
                            y2 = x + random.randint(1, 5)
                            if y2 not in new_set:
                                new_set.append(y2)
            elif op == 'shift':
                # Shift all non-zero elements by 1? Might break 0 constraint if we shift 0, but we don't.
                # This is risky. Let's stick to add/remove/replace.
                pass
                
            # Ensure 0 is in new_set
            if 0 not in new_set:
                new_set.append(0)
                
            new_set = list(set(new_set))
            new_score = _score(new_set)
            
            if new_score > current_score:
                current_score = new_score
                current_set = new_set
                improved = True
                
        # If no improvement in this batch, try a random restart from best_set or a perturbed best_set
        if not improved:
            # Perturb best_set
            current_set = best_set[:]
            # Add a random element
            max_val = max(current_set)
            x = random.randint(1, max_val + 10)
            current_set.append(x)
            current_set = list(set(current_set))
            current_score = _score(current_set)
            
            # If this perturbation is worse than best, revert to best and try more aggressive changes?
            # Actually, just keep exploring from current.
            
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]

    return best_set
