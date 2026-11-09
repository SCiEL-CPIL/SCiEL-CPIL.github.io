import time
import random
from itertools import combinations

def _is_sidon_with_diagonals(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _largest_sidon_subset_size(arr):
    n = len(arr)
    for size in range(n, 0, -1):
        for combo in combinations(range(n), size):
            if _is_sidon_with_diagonals([arr[i] for i in combo]):
                return size
    return 1

def _count_sum_collisions(arr):
    """Count the number of sum collisions in the set.
    A collision occurs when a+b = c+d where {a,b} != {c,d} (as multisets).
    More collisions means harder to find large Sidon subsets."""
    n = len(arr)
    sum_count = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sum_count[s] = sum_count.get(s, 0) + 1
    
    # Count pairs of pairs that have the same sum
    collisions = 0
    for count in sum_count.values():
        if count > 1:
            # Number of ways to choose 2 pairs from count pairs
            collisions += count * (count - 1) // 2
    
    return collisions

def construct():
    start_time = time.time()
    deadline = start_time + 70
    
    random.seed(777)
    
    best_set = None
    best_score = 13  # Start high
    best_collision = float('inf')
    
    # Try multiple initial solutions and local searches
    initial_solutions = [
        # Symmetric pairs around 64
        [2, 126, 4, 124, 6, 122, 8, 120, 10, 118, 12, 116],
        # Clumped values
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Spread with structure
        [0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98],
        # Arithmetic progression
        [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 128],
        # Quadratic-like
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121],
    ]
    
    for init in initial_solutions:
        if time.time() > deadline:
            break
        current_set = init[:]
        current_score = _largest_sidon_subset_size(current_set)
        current_collision = _count_sum_collisions(current_set)
        
        if current_score < best_score or (current_score == best_score and current_collision < best_collision):
            best_score = current_score
            best_set = current_set[:]
            best_collision = current_collision
        
        # Local search: optimize both score and collision count
        all_values = set(range(129))
        no_improve = 0
        while time.time() < deadline and no_improve < 300:
            i = random.randint(0, 11)
            candidates = [v for v in all_values if v not in current_set]
            if not candidates:
                break
            new_val = random.choice(candidates)
            old_val = current_set[i]
            current_set[i] = new_val
            
            new_score = _largest_sidon_subset_size(current_set)
            new_collision = _count_sum_collisions(current_set)
            
            # Accept if strictly better in score, or same score with fewer collisions
            if new_score < current_score or (new_score == current_score and new_collision < current_collision):
                current_score = new_score
                current_collision = new_collision
                no_improve = 0
                if new_score < best_score or (new_score == best_score and new_collision < best_collision):
                    best_score = new_score
                    best_set = current_set[:]
                    best_collision = new_collision
            else:
                current_set[i] = old_val
                no_improve += 1
    
    if best_set is None:
        best_set = initial_solutions[0]
    
    return best_set
