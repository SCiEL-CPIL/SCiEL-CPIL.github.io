"""Optimized program: Combined exhaustive search, local search, and simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import log, exp

def _compute_score(U):
    """Compute the score for a given set U."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0 or 0 not in U:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Compute U+U and U-U
    sumset = np.unique(U[:, None] + U[None, :])
    diffset = np.unique(U[:, None] - U[None, :])
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _local_search(U, max_iterations=5000, timeout=10.0):
    """Local search: add/remove/swap elements to improve score."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        # Adaptive search bound based on current max
        search_bound = max(50, max_u + 20)
        
        # Generate candidates
        candidates = set()
        # Perturbations - focus on small offsets for ratio optimization
        for x in list(U_set):
            for off in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 11]:
                c = x + off
                if 0 <= c <= search_bound:
                    candidates.add(c)
        # Sums/Diffs of existing elements
        elems = list(U_set)
        for a in elems:
            for b in elems:
                s = a + b
                if s <= search_bound:
                    candidates.add(s)
                d = abs(a - b)
                if 0 < d <= search_bound:
                    candidates.add(d)
        # Random candidates - bias towards smaller values
        for _ in range(20):
            if random.random() < 0.7:
                candidates.add(random.randint(0, max(10, max_u)))
            else:
                candidates.add(random.randint(0, search_bound))
        # Boundary and structured candidates
        if max_u > 0:
            candidates.add(max_u + 1)
            candidates.add(max_u + 2)
            candidates.add(max_u + 3)
            # Powers of 2 can help with difference set structure
            p2 = 1
            while p2 <= search_bound:
                candidates.add(p2)
                p2 *= 2
            # Fibonacci-like numbers
            fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
            for f in fibs:
                if f <= search_bound:
                    candidates.add(f)
            
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        
        improved = False
        for cand in candidate_list:
            if time.time() - start_time > timeout:
                break
            if cand in U_set:
                continue
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-10:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removals - prioritize removing large elements first (they increase denominator)
            removable = [x for x in current if x != 0]
            removable.sort(reverse=True)  # Try removing largest first
            random.shuffle(removable[:len(removable)//2])  # Shuffle first half
            removable = removable[:len(removable)//2] + removable[len(removable)//2:]
            for cand in removable[:20]:
                if time.time() - start_time > timeout:
                    break
                new_set = U_set - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-10:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 10:
            # Swap - try replacing a large element with a smaller one
            if random.random() < 0.5:
                removable = [x for x in current if x != 0]
                if removable:
                    # Preferentially replace larger elements
                    removable.sort(reverse=True)
                    rem = random.choice(removable[:max(1, len(removable)//3)])
                    # Replace with a smaller random value
                    if max_u > 5:
                        new_elem = random.randint(0, max(1, max_u // 2))
                    else:
                        new_elem = random.randint(0, max(1, max_u))
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-10:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
    return best, best_score

def _sa_search(initial_set, max_val, timeout=10.0):
    """Simulated Annealing to find a good set."""
    start_time = time.time()
    
    current = sorted(set(initial_set))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    
    best = list(current)
    best_score = current_score
    
    T = 0.08
    T_min = 1e-6
    cooling = 0.997
    
    while time.time() - start_time < timeout:
        if T < T_min:
            break
            
        # Generate neighbor
        move_type = random.random()
        new_set = list(current)
        
        if move_type < 0.45:
            # Change an element - bias towards reducing max value
            if len(new_set) <= 1:
                continue
            idx = random.randint(0, len(new_set) - 1)
            if idx == 0 and new_set[0] == 0:
                idx = random.randint(1, len(new_set) - 1) if len(new_set) > 1 else 0
            old_val = new_set[idx]
            # Try values biased towards smaller numbers
            if random.random() < 0.6:
                # Replace with a smaller value
                new_val = random.randint(0, max(1, old_val // 2))
            else:
                # Nearby value
                delta = random.randint(-8, 8)
                new_val = old_val + delta
                if new_val < 0:
                    new_val = random.randint(0, max_val)
                elif new_val > max_val:
                    new_val = random.randint(0, max_val)
            if new_val in new_set and new_val != old_val:
                continue
            new_set[idx] = new_val
        elif move_type < 0.70:
            # Add an element - bias towards small values
            if random.random() < 0.7:
                new_elem = random.randint(0, max(5, max_val // 3))
            else:
                new_elem = random.randint(0, max_val)
            if new_elem not in new_set:
                new_set.append(new_elem)
            else:
                continue
        else:
            # Remove an element (not 0) - prefer removing large elements
            removable = [x for x in new_set if x != 0]
            if not removable:
                continue
            # Sort and preferentially remove larger elements
            removable.sort(reverse=True)
            if random.random() < 0.5:
                rem = random.choice(removable[:max(1, len(removable)//2)])
            else:
                rem = random.choice(removable)
            new_set.remove(rem)
            
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            continue
            
        new_score = _compute_score(new_set)
        
        # Acceptance with better temperature scaling
        if new_score >= current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        else:
            # Use a slightly more aggressive acceptance for small T
            effective_T = max(T, 0.01)
            if random.random() < exp((new_score - current_score) / effective_T):
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                
        T *= cooling
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145 # Safety margin for 150s limit

    best_result = [0, 1, 2]
    best_score = _compute_score(best_result)
    
    # 1. Exhaustive search for small sets (n=3 to 6) within a bounded range
    # Focus on smaller max values since they give better denominator
    max_val_exhaustive = 40
    for n in range(3, 7):
        if time.time() > deadline - 30:
            break
        # Use combinations for exact search - prioritize smaller values
        for combo in combinations(range(1, max_val_exhaustive + 1), n - 1):
            U = [0] + list(combo)
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_result = U
    
    # 2. Targeted search for high-ratio structures with small max
    # Try specific structured sets known to have good difference/sum ratios
    if time.time() < deadline - 25:
        structured_sets = [
            [0, 1, 3, 7, 15],          # Mersenne-like
            [0, 1, 2, 4, 8, 16],        # Powers of 2
            [0, 1, 2, 4, 7, 12],        # Mixed
            [0, 1, 3, 6, 10, 15],       # Triangular-ish
            [0, 1, 4, 9, 16, 25],       # Squares
            [0, 1, 3, 9, 27],           # Powers of 3
            [0, 1, 2, 5, 10, 20],       # Geometric-ish
            [0, 1, 3, 4, 8, 12, 16],    # Mixed small
            [0, 1, 2, 3, 5, 8, 13],     # Fibonacci
            [0, 1, 2, 4, 6, 9, 13],     # Sparse
        ]
        for s in structured_sets:
            if 0 not in s:
                s = [0] + s
            score = _compute_score(s)
            if score > best_score:
                best_score = score
                best_result = s
    
    # 3. Local search from the best found
    if time.time() < deadline - 15:
        improved, score = _local_search(best_result, max_iterations=30000, timeout=min(25.0, deadline - time.time() - 3))
        if score > best_score:
            best_score = score
            best_result = improved
            
    # 4. Simulated Annealing from diverse starting points
    if time.time() < deadline - 10:
        # Start from best result with focused max_val
        result, score = _sa_search(best_result, max_val=100, timeout=min(20.0, deadline - time.time() - 3))
        if score > best_score:
            best_score = score
            best_result = result
        
        # Start from structured sets
        for s in structured_sets[:4]:
            if time.time() > deadline - 8:
                break
            if 0 not in s:
                s = [0] + s
            result, score = _sa_search(s, max_val=80, timeout=min(8.0, deadline - time.time() - 2))
            if score > best_score:
                best_score = score
                best_result = result
        
        # Start from random small sets (biased towards small values)
        for _ in range(4):
            if time.time() > deadline - 5:
                break
            n = random.randint(3, 7)
            rand_set = {0}
            while len(rand_set) < n:
                if random.random() < 0.7:
                    rand_set.add(random.randint(1, 30))
                else:
                    rand_set.add(random.randint(1, 60))
            rand_list = sorted(rand_set)
            result, score = _sa_search(rand_list, max_val=80, timeout=min(6.0, deadline - time.time() - 1))
            if score > best_score:
                best_score = score
                best_result = result

    # 5. Final refinement on the best result
    if time.time() < deadline - 2:
        improved, score = _local_search(best_result, max_iterations=15000, timeout=min(8.0, deadline - time.time() - 1))
        if score > best_score:
            best_score = score
            best_result = improved

    # Final check
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 2, 4]
        
    return best_result
# EVOLVE-BLOCK-END
