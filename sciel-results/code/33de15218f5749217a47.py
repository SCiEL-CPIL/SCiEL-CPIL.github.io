import time
from math import log, exp, floor, sqrt
import random
import numpy as np

def score_set(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    # Optimization: Use lists for speed, then convert to sets
    # But for small n, sets are fine.
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            if d != 0:
                diffs.add(d)
                diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Initial diverse set of candidates
    # Based on known good structures and random geometric starts
    candidates = []
    
    # 1. Small exhaustive-like seeds
    candidates.append([0, 1, 4, 9, 11])
    candidates.append([0, 1, 3, 7, 15, 31]) # Powers of 2 minus 1 style
    candidates.append([0, 1, 2, 4, 8, 16, 32]) # Powers of 2
    
    # 2. Random geometric sets
    for _ in range(10):
        m = random.randint(5, 15)
        r = random.uniform(1.2, 2.5)
        s = random.uniform(1.0, 5.0)
        U = [0]
        current = s
        for _ in range(m):
            current *= r
            val = int(floor(current))
            if val > 0 and val not in U:
                U.append(val)
        candidates.append(sorted(U))
        
    # 3. Random "sparse" sets
    for _ in range(10):
        max_val = random.randint(50, 500)
        n_elems = random.randint(5, 15)
        # Pick random distinct values
        U = [0] + sorted(random.sample(range(1, max_val), n_elems - 1))
        candidates.append(U)
        
    # Evaluate all candidates to find the best initial state
    best_set = None
    best_score = -1
    for U in candidates:
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    if best_set is None:
        best_set = [0, 1, 4, 9, 11]
        best_score = score_set(best_set)

    # Simulated Annealing
    current_set = list(best_set)
    current_score = best_score
    
    # SA Parameters
    T = 1.0  # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.995
    
    # We will run SA until time limit or temperature is too low
    # To make it effective, we need to allow "jumps"
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Generate a neighbor
        # Operations:
        # 1. Shift a random element
        # 2. Add a random element
        # 3. Remove a random element (if size > 3)
        
        op = random.random()
        n = len(current_set)
        
        new_set = list(current_set)
        
        if n > 3 and op < 0.1:
            # Remove element (not 0)
            idx = random.randint(1, n - 1)
            new_set.pop(idx)
        elif op < 0.3:
            # Add element
            max_val = max(current_set) if current_set else 10
            # Try adding near existing gaps or extending range
            if random.random() < 0.5:
                # Near max
                new_val = max_val + random.randint(1, 10)
            else:
                # Random in range
                new_val = random.randint(1, max_val)
            
            if new_val not in new_set and new_val > 0:
                new_set.append(new_val)
                new_set.sort()
        else:
            # Shift element
            if n > 1:
                idx = random.randint(1, n - 1)
                old_val = new_set[idx]
                # Shift by small amount
                delta = random.randint(-5, 5)
                new_val = old_val + delta
                if new_val > 0 and new_val not in new_set:
                    new_set[idx] = new_val
                    new_set.sort()
                else:
                    # Invalid move, skip
                    continue
            else:
                continue
        
        new_score = score_set(new_set)
        
        # Acceptance probability
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            # Metropolis criterion
            # Delta E is negative (score decreased)
            delta = new_score - current_score
            if delta == 0:
                continue
            prob = exp(delta / T)
            if random.random() < prob:
                current_set = new_set
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        # Occasionally restart temperature if stuck? 
        # Or just let it cool.
        
    # Final local polish on the best set found by SA
    # This fixes any minor rounding or local traps SA might have left
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            n = len(current)
            
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try small deltas
                for delta in range(-3, 4):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved and n < 25 and time.time() < deadline - 1:
                max_val = max(current)
                # Try values near the gap or max
                for new_val in range(1, max_val + 5):
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                        
            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
