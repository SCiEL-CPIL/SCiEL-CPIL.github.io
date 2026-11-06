"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _compute_sizes(A):
    """Compute |A+A| and |A*A| for set A."""
    A_list = list(A)
    n = len(A_list)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A_list[i] + A_list[j])
            prods.add(A_list[i] * A_list[j])
    return len(sums), len(prods)

def _score(A):
    s, p = _compute_sizes(A)
    return max(s, p)

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave margin for evaluation overhead
    
    # Candidate pool: include small integers, some structured numbers
    # Small integers create sum collisions
    # Numbers with many factors create product collisions
    candidates = set(range(1, 101))
    # Add some larger numbers that might help
    for i in range(2, 101):
        for j in range(2, 101):
            if i * j <= 10000:
                candidates.add(i * j)
    candidates = list(candidates)
    random.shuffle(candidates)
    
    # Initial set: start with small integers
    A = set(range(1, 21))
    current_score = _score(A)
    
    best_A = set(A)
    best_score = current_score
    
    # Local search: try swapping one element with a candidate
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        A_list = list(A)
        random.shuffle(A_list)
        
        for idx, old_val in enumerate(A_list):
            if (time.time() - start_time) > time_limit:
                break
            # Try replacing old_val with a candidate
            for new_val in candidates:
                if new_val in A or new_val == old_val:
                    continue
                if new_val > 10000:
                    continue
                
                new_A = A.copy()
                new_A.discard(old_val)
                new_A.add(new_val)
                
                new_score = _score(new_A)
                if new_score < current_score:
                    A = new_A
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = set(A)
                    improved = True
                    break  # Restart with new A
                elif new_score == current_score:
                    # Accept equal score to explore
                    if random.random() < 0.3:
                        A = new_A
                        improved = True
                        break
    
    # If time remains, try multiple restarts
    if (time.time() - start_time) < time_limit - 5:
        for restart in range(5):
            if (time.time() - start_time) > time_limit - 2:
                break
            # New random-ish initial set
            A = set()
            # Mix of small and some structured numbers
            small_pool = list(range(1, 51))
            random.shuffle(small_pool)
            for x in small_pool:
                if len(A) >= 20:
                    break
                A.add(x)
            # Fill remaining if needed
            if len(A) < 20:
                for x in range(51, 10001):
                    if len(A) >= 20:
                        break
                    if x not in A:
                        A.add(x)
            
            current_score = _score(A)
            if current_score < best_score:
                best_A = set(A)
                best_score = current_score
            
            # Local search on this restart
            improved = True
            while improved and (time.time() - start_time) < time_limit - 1:
                improved = False
                A_list = list(A)
                random.shuffle(A_list)
                
                for idx, old_val in enumerate(A_list):
                    if (time.time() - start_time) > time_limit - 1:
                        break
                    for new_val in candidates:
                        if new_val in A or new_val == old_val:
                            continue
                        if new_val > 10000:
                            continue
                        
                        new_A = A.copy()
                        new_A.discard(old_val)
                        new_A.add(new_val)
                        
                        new_score = _score(new_A)
                        if new_score < current_score:
                            A = new_A
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_A = set(A)
                            improved = True
                            break
    
    return sorted(best_A)
# EVOLVE-BLOCK-END
