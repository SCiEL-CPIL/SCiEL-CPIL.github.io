import itertools
import math
import time
import numpy as np

def _score_set(U):
    """Compute the score for a set U of nonnegative integers."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))  # ensure uniqueness, keep 0
    U.sort()
    n = len(U)
    if n < 1:
        return 0.0
    
    # Compute sumset |U+U| using numpy for speed
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    P = len(np.unique(sums))
    
    # Compute difference set |U-U|
    diffs = U_arr[:, None] - U_arr[None, :]
    D = len(np.unique(diffs))
    
    max_U = max(U)
    if max_U == 0:
        return 0.0
    
    ratio = D / P
    if ratio <= 1:
        return 0.0
    
    denom = math.log(2 * max_U + 1)
    if denom == 0:
        return 0.0
    
    score = 1 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search for small sets to get a good starting point
    def search_exhaustive(k, max_val):
        nonlocal best_score, best_set
        for combo in itertools.combinations(range(1, max_val + 1), k - 1):
            U = [0] + list(combo)
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_set = U
            
            if time.time() > deadline:
                return True
        
        return False
    
    # Quick exhaustive search for small sets
    search_configs = [
        (2, 20),
        (3, 20),
        (4, 15),
        (5, 12),
    ]
    
    for k, max_val in search_configs:
        if search_exhaustive(k, max_val):
            break
    
    # Phase 2: Simulated Annealing
    def simulated_annealing(U_init, max_val, num_iterations, T0, T_min, alpha):
        nonlocal best_score, best_set
        import random
        
        U = list(U_init)
        current_score = _score_set(U)
        T = T0
        
        for i in range(num_iterations):
            if time.time() > deadline - 2:
                break
            
            # Choose a random move
            move_type = random.randint(0, 4)
            
            if move_type == 0:  # Add an element
                candidates = [x for x in range(1, max_val + 1) if x not in U]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_U = U + [x]
            elif move_type == 1:  # Remove an element
                removable = [x for x in U if x != 0]
                if not removable:
                    continue
                x = random.choice(removable)
                new_U = [y for y in U if y != x]
                if len(new_U) < 2:
                    continue
            elif move_type == 2:  # Replace an element
                removable = [x for x in U if x != 0]
                if not removable:
                    continue
                x = random.choice(removable)
                candidates = [y for y in range(1, max_val + 1) if y not in U]
                if not candidates:
                    continue
                y = random.choice(candidates)
                new_U = [z for z in U if z != x] + [y]
            elif move_type == 3:  # Swap two elements
                removable = [x for x in U if x != 0]
                if len(removable) < 2:
                    continue
                x, y = random.sample(removable, 2)
                candidates = [z for z in range(1, max_val + 1) if z not in U]
                if not candidates:
                    continue
                z = random.choice(candidates)
                new_U = [w for w in U if w != x] + [z]
            else:  # Fine-tune: adjust max element
                max_U = max(U)
                delta = random.randint(-5, 5)
                new_max = max_U + delta
                if new_max <= 0 or new_max in U:
                    continue
                new_U = [x for x in U if x != max_U] + [new_max]
            
            new_score = _score_set(new_U)
            delta_score = new_score - current_score
            
            # Accept if better, or with probability exp(-delta_score/T) if worse
            if delta_score >= 0 or random.random() < math.exp(delta_score / max(T, 1e-10)):
                U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(U)
            
            # Cool down
            T = max(T_min, T * alpha)
        
        return U
    
    # Run SA with multiple restarts from different starting points
    sa_configs = [
        ([0, 1, 3], 100, 5000, 0.1, 0.001, 0.995),
        ([0, 1, 3, 9], 120, 8000, 0.15, 0.001, 0.994),
        ([0, 2, 5, 12], 150, 10000, 0.2, 0.001, 0.993),
        ([0, 1, 4, 10, 25], 200, 12000, 0.25, 0.001, 0.992),
    ]
    
    for U_init, max_val, num_iter, T0, T_min, alpha in sa_configs:
        if time.time() > deadline - 10:
            break
        simulated_annealing(U_init, max_val, num_iter, T0, T_min, alpha)
    
    # Phase 3: Final polish - local search around the best solution
    if time.time() < deadline - 5:
        U = list(best_set)
        U.sort()
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            current_score = _score_set(U)
            
            # Try adding each possible new element
            for x in range(1, 200):
                if x not in U:
                    new_U = U + [x]
                    score = _score_set(new_U)
                    if score > current_score + 1e-9:
                        U = new_U
                        current_score = score
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_set = list(U)
                        break
            
            # Try removing an element if it helps
            if not improved:
                for x in list(U):
                    if x == 0:
                        continue
                    new_U = [y for y in U if y != x]
                    if len(new_U) >= 2:
                        score = _score_set(new_U)
                        if score > current_score + 1e-9:
                            U = new_U
                            current_score = score
                            improved = True
                            if score > best_score:
                                best_score = score
                                best_set = list(U)
                            break
            
            # Try swap: remove one, add another
            if not improved:
                for x in list(U):
                    if x == 0:
                        continue
                    for y in range(1, 200):
                        if y not in U:
                            new_U = [z for z in U if z != x] + [y]
                            score = _score_set(new_U)
                            if score > current_score + 1e-9:
                                U = new_U
                                current_score = score
                                improved = True
                                if score > best_score:
                                    best_score = score
                                    best_set = list(U)
                                break
                    if improved:
                        break
    
    return best_set
