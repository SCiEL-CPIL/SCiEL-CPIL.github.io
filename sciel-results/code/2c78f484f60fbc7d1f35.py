"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _score_set(U):
    """Calculate the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    
    n = len(U)
    sums = set()
    diffs = set()
    
    # Optimization: use list iteration
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
            
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0 or n_diffs == 0:
        return -1
        
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return -1
        
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
        
    return 1 + math.log(ratio) / denom

def _local_search(U, deadline, max_iter=5000, temp=1.0):
    """Local search with simulated annealing-like acceptance and random swaps."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    best_U = U[:]
    best_score = current_score
    max_u = max(U)
    
    for i in range(max_iter):
        if time.time() > deadline - 2:
            break
            
        # Choose a random move
        move_type = random.choice(['add', 'remove', 'swap', 'randomize'])
        
        new_U = U[:]
        
        if move_type == 'add':
            # Add a random element near the current range
            # Range extended slightly
            low = 0
            high = max_u + 20
            new_elem = random.randint(low, high)
            if new_elem not in U:
                new_U.append(new_elem)
                new_U.sort()
            else:
                continue # Skip if no change
                
        elif move_type == 'remove':
            if len(U) <= 2:
                continue
            # Remove a random non-zero element
            idx = random.randint(1, len(U) - 1)
            new_U.pop(idx)
            
        elif move_type == 'swap':
            if len(U) <= 1:
                continue
            idx = random.randint(1, len(U) - 1)
            old_elem = U[idx]
            # Replace with a random element
            new_elem = random.randint(1, max_u + 30)
            if new_elem == old_elem:
                continue
            if new_elem in U:
                continue
            new_U[idx] = new_elem
            new_U.sort()
            
        elif move_type == 'randomize':
            # Occasionally randomize one element significantly to jump out of local minima
            if len(U) <= 1:
                continue
            idx = random.randint(1, len(U) - 1)
            new_elem = random.randint(1, max_u * 2 + 50)
            if new_elem in U:
                continue
            new_U[idx] = new_elem
            new_U.sort()

        new_score = _score_set(new_U)
        
        if new_score > current_score:
            U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
            max_u = max(U)
        else:
            # Accept worse moves with some probability (simulated annealing)
            # Probability decreases over time
            if random.random() < 0.05: # Fixed small probability for exploration
                U = new_U
                current_score = new_score
                max_u = max(U)
                
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # 1. Systematic search for small sizes
    # Size 3
    for a in range(1, 30):
        for b in range(a + 1, 50):
            if time.time() > deadline - 10:
                return best_set
            U = [0, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    # Size 4
    for a in range(1, 20):
        for b in range(a + 1, 35):
            for c in range(b + 1, 60):
                if time.time() > deadline - 10:
                    return best_set
                U = [0, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
                    
    # Size 5 (limited search space)
    for a in range(1, 15):
        for b in range(a + 1, 25):
            for c in range(b + 1, 40):
                for d in range(c + 1, 60):
                    if time.time() > deadline - 10:
                        return best_set
                    U = [0, a, b, c, d]
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U[:]

    # 2. Local search from best found
    if time.time() < deadline - 20:
        improved_set, improved_score = _local_search(best_set, deadline, max_iter=2000)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score

    # 3. Local search from several promising starting points
    # Generate some structured starting points
    starting_points = []
    
    # Golomb rulers / Sidon-like sets
    for n in range(4, 10):
        # Simple construction: powers of 2 (not great for sums, but good diffs)
        # Or just random sets
        for _ in range(5):
            U = [0]
            max_val = 0
            for _ in range(n - 1):
                # Pick a number that maximizes difference uniqueness
                # Heuristic: pick large numbers
                candidate = random.randint(max_val + 1, max_val + 20)
                # Check if it's a good addition
                # Just add for now, local search will fix it
                U.append(candidate)
                max_val = candidate
            U.sort()
            starting_points.append(U)
            
    # Add the best set found so far
    starting_points.append(best_set[:])
    
    # Add some hardcoded good candidates if known
    # Based on small set analysis, sets like [0, 1, 4, 6] are decent
    starting_points.extend([
        [0, 1, 4, 6],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 2, 5, 9],
        [0, 1, 5, 8],
        [0, 1, 4, 7],
        [0, 2, 4, 7],
        [0, 1, 3, 5],
        [0, 1, 3, 4, 8],
        [0, 1, 2, 4, 8],
    ])
    
    for start in starting_points:
        if time.time() > deadline - 5:
            break
        s = _score_set(start)
        if s > best_score - 0.1: # Only search from promising starts
            improved_set, improved_score = _local_search(start, deadline, max_iter=1500)
            if improved_score > best_score:
                best_set = improved_set
                best_score = improved_score
                
    if time.time() > deadline - 1:
        return best_set
    
    return best_set

# EVOLVE-BLOCK-END
