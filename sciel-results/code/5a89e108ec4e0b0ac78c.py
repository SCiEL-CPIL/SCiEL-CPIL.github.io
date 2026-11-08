import math
import random
import time

def score_set(U):
    if not U: return 0
    max_u = max(U)
    if max_u == 0: return 1.0
    
    sum_set = set()
    diff_set = set()
    
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0: return 1.0
    
    ratio = size_diff / size_sum
    # log base 2 or e? Formula says log. Usually natural log in math, but ratio is invariant.
    # Denominator log(2 max U + 1)
    denom = math.log(2 * max_u + 1)
    if denom == 0: return 1.0
    
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Initialize with some good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 5],
        [0, 1, 4, 6],
        [0, 1, 2, 4, 7],
        [0, 1, 3, 4, 7],
        [0, 1, 2, 5, 6]
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # Local Search
    # We will search for sets of size up to 12
    # Represent set as sorted list
    
    def get_neighbors(U):
        neighbors = []
        n = len(U)
        max_val = max(U) if U else 10
        # 1. Move an element
        for i in range(n):
            for delta in [-2, -1, 1, 2, 3]:
                new_val = U[i] + delta
                if new_val >= 0:
                    new_U = U[:]
                    new_U[i] = new_val
                    if new_val not in new_U[:i] + new_U[i+1:]: # Ensure unique
                         neighbors.append(sorted(new_U))
        # 2. Add an element
        if n < 12:
            for val in range(0, max_val + 5):
                if val not in U:
                    new_U = U + [val]
                    neighbors.append(sorted(new_U))
        # 3. Remove an element
        if n > 1:
            for i in range(n):
                if U[i] == 0 and n == 1: continue # Must contain 0? Task says "containing 0".
                # If we remove 0, we must ensure 0 is added back or start with 0.
                # Easiest: never remove 0.
                if U[i] == 0: continue
                new_U = U[:i] + U[i+1:]
                neighbors.append(sorted(new_U))
        return neighbors

    # Simulated Annealing
    current_set = best_set[:]
    current_score = best_score
    T = 0.1
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        neighbors = get_neighbors(current_set)
        if not neighbors:
            break
        next_set = random.choice(neighbors)
        next_score = score_set(next_set)
        
        delta = next_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_set = next_set
            current_score = next_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        
        T *= cooling
        steps += 1
        
        # Occasionally restart with a random set to avoid local minima
        if steps % 500 == 0:
             # Generate random set of size 4-8
             n = random.randint(4, 8)
             vals = random.sample(range(0, 20), n)
             if 0 not in vals:
                 vals[0] = 0
             r_set = sorted(vals)
             r_score = score_set(r_set)
             if r_score > current_score:
                 current_set = r_set
                 current_score = r_score
                 if current_score > best_score:
                     best_score = current_score
                     best_set = current_set[:]

    return best_set
