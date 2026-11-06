import time
import random
from itertools import combinations

def is_sidon_with_diagonals(subset):
    """
    Check if a subset of integers has all unordered pair sums (including diagonals) unique.
    """
    sums = set()
    k = len(subset)
    for i in range(k):
        for j in range(i, k):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidate_set):
    """
    Find the size of the largest subset of candidate_set that is Sidon with diagonals.
    candidate_set is a list of 12 distinct integers.
    """
    n = len(candidate_set)
    # Optimization: Check from largest k down
    for k in range(n, 0, -1):
        found = False
        for comb in combinations(candidate_set, k):
            if is_sidon_with_diagonals(comb):
                found = True
                break
        if found:
            return k
    return 0

def collision_count(a, b, c, d):
    """
    Count how many ways a+b = c+d (with a<=b, c<=d) given 4 distinct values.
    """
    count = 0
    if a + b == c + d:
        count += 1
    if a + c == b + d:
        count += 1
    if a + d == b + c:
        count += 1
    return count

def spectral_select(initial_pool, num_to_select=12):
    """
    Select num_to_select integers from initial_pool using a spectral-like greedy approach
    that maximizes local collision density.
    """
    if len(initial_pool) <= num_to_select:
        return initial_pool[:]
    
    # Compute a "collision score" for each element based on how many additive quadruples
    # it participates in with other elements in the pool.
    # Higher score means more collisions, which is what we want (to break Sidon sets).
    
    pool_size = len(initial_pool)
    scores = [0] * pool_size
    
    # For each pair (i, j), check how many other pairs (k, l) satisfy pool[i]+pool[j] = pool[k]+pool[l]
    # This is O(n^4) which might be too slow for large pools, but we'll limit pool size.
    # Actually, let's use a simpler heuristic: count how many other elements x in the pool
    # create a collision with a fixed pair (a, b). i.e., count x such that a+b = x+y for some y.
    
    # Simpler: for each element x, count how many pairs (a, b) with a != b, a != x, b != x
    # such that a + b = x + y for some y in pool. This is getting complex.
    
    # Let's use a direct but efficient approach:
    # For each element x in the pool, compute the number of "conflicts" it has.
    # A conflict for x is a pair (a, b) such that a + b = x + c for some c in pool, c != x, a != b, a != c, b != c.
    # Actually, let's just compute the number of additive quadruples involving x.
    
    # To keep it fast, we'll limit the pool to at most 50 elements.
    if pool_size > 50:
        # Randomly sample 50
        initial_pool = random.sample(initial_pool, 50)
        pool_size = 50
    
    # Compute scores
    for i in range(pool_size):
        xi = initial_pool[i]
        for j in range(i + 1, pool_size):
            xj = initial_pool[j]
            s = xi + xj
            # Count how many pairs (k, l) with k < l, k != i, l != i, k != j, l != j
            # such that initial_pool[k] + initial_pool[l] == s
            for k in range(pool_size):
                if k == i or k == j:
                    continue
                for l in range(k + 1, pool_size):
                    if l == i or l == j:
                        continue
                    if initial_pool[k] + initial_pool[l] == s:
                        scores[i] += 1
                        scores[j] += 1
    
    # Select the top num_to_select elements by score
    indexed_scores = [(score, idx) for idx, score in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[0], reverse=True)
    
    selected_indices = [idx for score, idx in indexed_scores[:num_to_select]]
    selected = [initial_pool[i] for i in selected_indices]
    selected.sort()
    
    return selected

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # Initial candidates with high collision density
    candidates = [
        list(range(12)),
        [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],
        [0, 2, 4, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20],
        [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21],
        [0, 1, 2, 3, 14, 15, 16, 17, 18, 19, 20, 21],
        # Add spectral-selected candidates
    ]
    
    # Use spectral selection to generate additional candidates
    # Pool: all integers in [0, 128]
    full_pool = list(range(129))
    for _ in range(5):
        # Randomly sample a subset of the pool to avoid always picking the same
        sampled_pool = random.sample(full_pool, 60)
        selected = spectral_select(sampled_pool, 12)
        if len(set(selected)) == 12 and all(0 <= x <= 128 for x in selected):
            candidates.append(selected)
    
    best_set = None
    best_score = 13
    
    for cand in candidates:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = max_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        best_set = list(range(12))
        best_score = max_sidon_size(best_set)
        
    if best_score <= 3:
        return best_set
        
    current_set = best_set[:]
    current_score = best_score
    
    # Local search with spectral-guided moves
    while time.time() < deadline:
        # Occasionally use spectral selection to generate a new candidate
        if random.random() < 0.1:
            # Build a pool around current_set
            pool = set(current_set)
            # Add neighbors
            for x in current_set:
                for dx in range(-5, 6):
                    nx = x + dx
                    if 0 <= nx <= 128:
                        pool.add(nx)
            # Add some random elements
            for _ in range(20):
                pool.add(random.randint(0, 128))
            
            pool_list = list(pool)
            if len(pool_list) > 12:
                selected = spectral_select(pool_list, 12)
                if len(set(selected)) == 12 and all(0 <= x <= 128 for x in selected):
                    new_set = selected
                    new_set.sort()
                    new_score = max_sidon_size(new_set)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_set = new_set[:]
                        current_set = new_set[:]
                        current_score = new_score
                        if best_score <= 3:
                            break
                    else:
                        # Accept with probability to escape local minima
                        if random.random() < 0.2:
                            current_set = new_set[:]
                            current_score = new_score
        else:
            # Standard local search: Replace 1 or 2 elements
            new_set = current_set[:]
            num_replaces = random.choice([1, 2])
            indices = random.sample(range(12), num_replaces)
            
            for idx in indices:
                if random.random() < 0.5:
                    # Near existing
                    ref = random.choice(current_set)
                    new_val = ref + random.randint(-3, 3)
                else:
                    new_val = random.randint(0, 128)
                    
                if 0 <= new_val <= 128 and new_val not in new_set:
                    new_set[idx] = new_val
                else:
                    new_val = random.randint(0, 128)
                    if 0 <= new_val <= 128 and new_val not in new_set:
                        new_set[idx] = new_val
                    else:
                        new_set[idx] = (new_set[idx] + 1) % 129
                        if new_set[idx] in new_set:
                            new_set[idx] = (new_set[idx] + 1) % 129
                        
            if len(set(new_set)) != 12:
                continue
                
            new_set.sort()
            new_score = max_sidon_size(new_set)
            
            if new_score < best_score:
                best_score = new_score
                best_set = new_set[:]
                current_set = new_set[:]
                current_score = new_score
                if best_score <= 3:
                    break
            else:
                if random.random() < 0.2:
                    current_set = new_set[:]
                    current_score = new_score

    return best_set
