import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if subset has all unordered pair sums (including diagonals) unique."""
    s = sorted(subset)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _max_sidon_size(arr):
    """Find the maximum size of a Sidon subset of arr."""
    n = len(arr)
    for size in range(n, 2, -1):
        for subset in combinations(arr, size):
            if _is_sidon(list(subset)):
                return size
    for subset in combinations(arr, 3):
        if _is_sidon(list(subset)):
            return 3
    return 2

def construct():
    deadline = time.time() + 80.0
    
    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)
    
    if best_score <= 3:
        return best_set
    
    random.seed(12345)
    
    current_set = list(best_set)
    current_score = best_score
    
    def evaluate(s):
        return _max_sidon_size(s)
    
    # Local search with swap and insert moves
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        indices = list(range(12))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline or current_score <= 3:
                break
            
            # Try replacing element i with a random value
            for attempt in range(8):
                new_val = random.randint(0, 128)
                if new_val in current_set:
                    continue
                
                old_val = current_set[i]
                current_set[i] = new_val
                new_score = evaluate(current_set)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_set = list(current_set)
                    if current_score <= 3:
                        break
                else:
                    current_set[i] = old_val
            
            if current_score <= 3:
                break
        
        if current_score <= 3:
            break
        
        if not improved:
            no_improve_count += 1
            # Perturbation: change 2-3 elements
            num_changes = random.randint(2, 4)
            change_indices = random.sample(range(12), num_changes)
            for idx in change_indices:
                current_set[idx] = random.randint(0, 128)
            # Ensure uniqueness
            attempts = 0
            while len(set(current_set)) < 12 and attempts < 100:
                dup_idx = random.randint(0, 11)
                current_set[dup_idx] = random.randint(0, 128)
                attempts += 1
            if len(set(current_set)) < 12:
                current_set = list(best_set)
                current_score = best_score
            else:
                current_score = evaluate(current_set)
                if current_score < best_score:
                    best_score = current_score
                    best_set = list(current_set)
            
            if no_improve_count > 50:
                # Full restart
                current_set = random.sample(range(129), 12)
                current_score = evaluate(current_set)
                if current_score < best_score:
                    best_score = current_score
                    best_set = list(current_set)
                no_improve_count = 0
        else:
            no_improve_count = 0
    
    return best_set
