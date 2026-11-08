import time
import random
import numpy as np
from math import log

def score_set(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all pairwise sums (with repetition)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    # Difference set: all pairwise differences (with sign)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = int(U_arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def generate_powers(base, k):
    """Generate set {0, 1, base, base^2, ..., base^(k-1)}"""
    U = [0, 1]
    val = 1
    for _ in range(k - 2):
        val *= base
        U.append(val)
    return sorted(set(U))

def generate_squares(k):
    """Generate set {0, 1, 4, 9, ..., k^2}"""
    return sorted([i*i for i in range(k+1)])

def generate_geometric_sparse(base, n, start=1):
    """Generate a geometric sequence set."""
    U = [0]
    val = float(start)
    seen = {0}
    for _ in range(n):
        val *= base
        iv = int(round(val))
        if iv not in seen and iv > 0:
            U.append(iv)
            seen.add(iv)
    return sorted(U)

def local_search(initial_U, time_limit):
    """
    Perform local search (hill climbing) starting from initial_U.
    """
    current_U = list(initial_U)
    current_score = score_set(current_U)
    best_U = list(current_U)
    best_score = current_score
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        neighbor = list(current_U)
        max_val = max(neighbor) if neighbor else 10
        
        move_type = random.choice(['add', 'remove', 'modify', 'shift'])
        
        if move_type == 'add':
            # Add a new element
            # Try near existing elements or at the end
            if random.random() < 0.5 and neighbor:
                base_idx = random.randint(0, len(neighbor)-1)
                base_val = neighbor[base_idx]
                # Small perturbation
                new_val = base_val + random.randint(-max(1, int(base_val*0.2)), max(1, int(base_val*0.2)))
            else:
                # Larger jump
                new_val = random.randint(1, max_val * 2)
            
            if new_val >= 0 and new_val not in neighbor:
                neighbor.append(new_val)
                
        elif move_type == 'remove' and len(neighbor) > 2:
            # Remove a random non-zero element
            idx = random.randint(1, len(neighbor)-1)
            neighbor.pop(idx)
            
        elif move_type == 'modify' and len(neighbor) > 1:
            # Modify a random non-zero element
            idx = random.randint(1, len(neighbor)-1)
            old_val = neighbor[idx]
            delta = random.randint(-max(1, int(old_val*0.1)), max(1, int(old_val*0.1)))
            new_val = old_val + delta
            if new_val >= 0 and new_val not in neighbor:
                neighbor[idx] = new_val
            else:
                # If invalid, try a small random change
                new_val = random.randint(0, max_val)
                if new_val not in neighbor:
                    neighbor[idx] = new_val
                else:
                    continue # Skip this move

        elif move_type == 'shift':
            # Shift the entire set slightly (scale)
            factor = random.uniform(0.9, 1.1)
            new_neighbor = [0] + [int(v * factor) for v in neighbor if v > 0]
            # Deduplicate
            new_neighbor = sorted(set(new_neighbor))
            if len(new_neighbor) > 1:
                neighbor = new_neighbor
            else:
                continue

        # Ensure 0 is present and sorted
        if 0 not in neighbor:
            neighbor.append(0)
        neighbor = sorted(set(neighbor))
        
        if len(neighbor) < 2:
            continue

        new_score = score_set(neighbor)
        
        if new_score > current_score:
            current_U = neighbor
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(current_U)
        else:
            # Simulated annealing-like acceptance? 
            # For simplicity and speed, strict hill climbing.
            # We can occasionally accept worse moves to escape local minima.
            if random.random() < 0.05: # 5% chance to accept worse
                current_U = neighbor
                current_score = new_score

    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Leave 5s buffer
    
    # Generate diverse initial seeds
    seeds = []
    
    # 1. Powers of small bases
    for base in [2, 3, 4, 5, 7, 10]:
        for k in range(5, 15):
            seeds.append(generate_powers(base, k))
            
    # 2. Squares
    for k in range(5, 20):
        seeds.append(generate_squares(k))
        
    # 3. Geometric sparse
    for _ in range(20):
        base = random.uniform(1.5, 4.0)
        n = random.randint(10, 30)
        seeds.append(generate_geometric_sparse(base, n))
        
    # 4. Random sparse
    for _ in range(20):
        n = random.randint(10, 30)
        U = [0]
        last = 0
        for _ in range(n-1):
            step = random.randint(1, max(1, int(last * random.uniform(0.5, 2.0)) + 1))
            last += step
            U.append(last)
        seeds.append(sorted(set(U)))
        
    # Run local search on a subset of seeds
    # We have ~100 seeds. Let's pick 10 best or random 10.
    # To maximize coverage, pick random 10 and also the "classic" ones.
    selected_seeds = random.sample(seeds, min(10, len(seeds)))
    
    best_overall_U = None
    best_overall_score = -1
    
    # Allocate time per seed
    time_per_seed = 10.0 # 10 seconds per seed, 10 seeds = 100s
    
    for seed in selected_seeds:
        if time.time() > deadline - 5:
            break
            
        remaining_time = deadline - time.time() - 5
        current_time_limit = min(time_per_seed, remaining_time)
        
        if current_time_limit <= 1:
            continue
            
        U, s = local_search(seed, current_time_limit)
        
        if s > best_overall_score:
            best_overall_score = s
            best_overall_U = U
            
    if best_overall_U is None:
        # Fallback
        best_overall_U = [0, 1, 3]
        
    return best_overall_U

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
