import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin for 120s limit (and 150s wall clock if applicable)
    
    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        if max_u == 0:
            return 1.0
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        arr = np.array(u, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        if r <= 1:
            return 1.0
        return 1 + math.log(r) / math.log(m)

    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # 1. Generate initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
    ]
    
    # 2. Greedy extension of Golomb rulers (fast, deterministic)
    def greedy_extend(base_u, target_n, max_search_depth=500):
        u = list(base_u)
        for _ in range(target_n - len(u)):
            best_pos = None
            best_score = -1.0
            current_max = max(u)
            # Search for the best new mark to add
            # Limit search range to avoid O(n^2) explosion in early steps if base is small
            # For Golomb rulers, optimal gaps grow roughly quadratically
            for new_mark in range(current_max + 1, current_max + max_search_depth + 1):
                candidate = sorted(u + [new_mark])
                if not is_golomb(candidate):
                    continue
                s = compute_score(candidate)
                if s > best_score:
                    best_score = s
                    best_pos = new_mark
            if best_pos is not None:
                u.append(best_pos)
                u.sort()
            else:
                # If no valid Golomb mark found in range, break
                break
        return u

    # Try to extend the best known small ruler
    if time.time() - start_time < DEADLINE - 10:
        ext_u = greedy_extend([0, 1, 4, 10, 12, 17], 12, max_search_depth=100)
        if ext_u and len(ext_u) > 6:
            candidates.append(ext_u)
            
        # Also try extending from scratch to see if we find a different structure
        ext_u2 = greedy_extend([0, 1], 12, max_search_depth=100)
        if ext_u2 and len(ext_u2) > 6:
            candidates.append(ext_u2)

    # 3. Local Search / Hill Climbing with Simulated Annealing
    # This allows us to escape local optima of the "Golomb" constraint
    # by allowing non-Golomb sets if they yield a better score.
    # The score function inherently penalizes sumset collisions.
    
    best_set = max(candidates, key=compute_score)
    best_score = compute_score(best_set)
    
    current_set = list(best_set)
    current_score = best_score
    
    # Initialize temperature
    T = 0.05
    T_min = 0.001
    cooling = 0.995
    
    while time.time() - start_time < DEADLINE:
        # Generate neighbor
        neighbor = list(current_set)
        move = random.choice(['add', 'remove', 'shift'])
        
        max_val = max(neighbor)
        
        if move == 'add':
            # Bias towards adding near the max to keep the set compact,
            # or adding small numbers to fill gaps
            # But sumset size grows with max(U), so we need to be careful.
            # Adding a number x increases max(U) if x > max(U).
            # If we add x <= max(U), max(U) stays same, but sumset/diffset might grow.
            # Usually, adding a new max element increases the denominator log(2*max+1)
            # but also increases numerator log(|U-U|/|U+U|).
            # We want the numerator to grow faster than the denominator.
            
            # Try adding a number slightly larger than max
            if random.random() < 0.5:
                new_val = max_val + random.randint(1, 10)
            else:
                # Try filling a gap
                gaps = [i for i in range(1, max_val + 1) if i not in neighbor]
                if gaps:
                    new_val = random.choice(gaps)
                else:
                    new_val = max_val + 1
            
            if new_val not in neighbor:
                neighbor.append(new_val)
            else:
                continue
                
        elif move == 'remove':
            if len(neighbor) > 1:
                # Don't remove 0
                rem_candidates = [x for x in neighbor if x != 0]
                rem = random.choice(rem_candidates)
                neighbor.remove(rem)
            else:
                continue
        else:
            # Shift a random element
            if len(neighbor) > 1:
                idx = random.choice([i for i, x in enumerate(neighbor) if x != 0])
                old_val = neighbor[idx]
                # Shift by a small amount
                delta = random.randint(-5, 5)
                new_val = old_val + delta
                if new_val < 1:
                    new_val = 1
                if new_val not in neighbor:
                    neighbor[idx] = new_val
                else:
                    continue
            else:
                continue
        
        neighbor.sort()
        
        # Calculate new score
        new_score = compute_score(neighbor)
        
        if new_score > current_score:
            current_set = neighbor
            current_score = new_score
            if current_score > best_score:
                best_set = list(current_set)
                best_score = current_score
        else:
            # Accept worse moves with probability based on temperature
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_set = neighbor
                    current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min

    return best_set
