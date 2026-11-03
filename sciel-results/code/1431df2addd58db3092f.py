import time
import random
from math import log
import itertools
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    # Vectorized calculation for speed
    U_arr = np.array(U)
    
    # Sum set
    # To save time/memory, we can compute sums of pairs i<=j
    # But for small N (< 30), O(N^2) is fine.
    # Let's use a set for exact counting.
    sums = set()
    diffs = set()
    
    # Optimization: If N is small, pure python sets are fast enough.
    # If N is large, we might need numpy unique.
    # Given the 120s limit, N will likely be < 20 for SA.
    
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
        if i < n - 1:
            for j in range(i + 1, n):
                d = U[j] - U[i]
                diffs.add(d)
                diffs.add(-d)
    
    diffs.add(0)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def fast_score(U):
    """
    Faster scoring using numpy for larger sets, falling back to python for small.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums
    # i <= j
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences
    # i < j
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    # Unique positive diffs
    num_pos_diffs = len(np.unique(diffs))
    # Total diffs = 2 * num_pos_diffs + 1 (for 0)
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def simulated_annealing(U, deadline, max_iter=2000):
    """
    Simulated annealing to improve the set U.
    Moves:
    1. Perturb an element.
    2. Add an element.
    3. Remove an element.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current
    best_score = current_score
    
    T = 0.1  # Initial temperature
    cooling = 0.995
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        move_type = random.choice(['perturb', 'add', 'remove', 'perturb'])
        
        new_set = current[:]
        
        if move_type == 'perturb' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
            # Try to move within a reasonable range
            # Range depends on the scale of the set
            max_val = max(current)
            scale = max(1, int(max_val * 0.1))
            new_val = old_val + random.randint(-scale, scale)
            if new_val <= 0:
                new_val = old_val + random.randint(1, scale)
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue # Skip if collision caused size change (unless we accept it)
                # Actually, if we want to keep size fixed for perturb, we skip.
                # But maybe changing size is allowed? 
                # Let's stick to fixed size for perturb to focus on shape.
        
        elif move_type == 'add' and n < 25:
            max_val = max(current)
            # Try adding near existing elements or gaps
            if random.random() < 0.5:
                # Near max
                new_val = max_val + random.randint(1, max(2, int(max_val * 0.2)))
            else:
                # Random in range
                new_val = random.randint(1, max_val * 2)
            new_set.append(new_val)
            new_set = sorted(set(new_set))
            if len(new_set) == n:
                continue # Duplicate
        
        elif move_type == 'remove' and n > 4:
            i = random.randint(1, n-1)
            new_set.pop(i)
        
        else:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score < 0:
            # Invalid move
            if random.random() < T:
                current = new_set
                current_score = new_score
        else:
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current
            else:
                # Accept worse move with probability
                delta = current_score - new_score
                if random.random() < np.exp(-delta / T):
                    current = new_set
                    current_score = new_score
        
        T *= cooling
        if T < 0.001:
            T = 0.001 # Floor for temperature to allow some exploration
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    best_set = [0, 1, 4, 9, 11]
    best_score = fast_score(best_set)
    
    # Known Golomb Rulers (0-indexed)
    # N=1: [0]
    # N=2: [0, 1]
    # N=3: [0, 1, 3]
    # N=4: [0, 1, 4, 6]
    # N=5: [0, 1, 4, 9, 11]
    # N=6: [0, 1, 4, 10, 12, 17]
    # N=7: [0, 1, 4, 10, 18, 23, 25]
    # N=8: [0, 1, 4, 10, 18, 23, 29, 34]
    # N=9: [0, 1, 4, 10, 18, 23, 29, 37, 41]
    # N=10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52]
    
    golomb_rulers = [
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
    ]
    
    # Add some random perturbations of these rulers
    candidates = list(golomb_rulers)
    
    # Generate some random sparse sets
    random.seed(42)
    for _ in range(20):
        n_elem = random.randint(4, 12)
        max_val = random.randint(50, 200)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
        
    # Geometric sets
    for ratio in [1.5, 2.0, 2.5, 3.0]:
        U = [0]
        val = 1
        for _ in range(10):
            U.append(val)
            val = int(val * ratio)
            if val > 1000:
                break
        candidates.append(U)
        
    # Evaluate candidates
    scored_candidates = []
    for U in candidates:
        s = fast_score(U)
        if s > 0:
            scored_candidates.append((s, U))
            
    scored_candidates.sort(reverse=True)
    
    # Run SA on top candidates
    for score, U in scored_candidates[:20]:
        if time.time() > deadline - 10:
            break
        refined, refined_score = simulated_annealing(U, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
    # Final polish on the best set
    if time.time() < deadline - 5:
        refined, refined_score = simulated_annealing(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
    return best_set

# Test
if __name__ == "__main__":
    print(f"Best set: {construct()}")
