import time
from math import log, floor
import random
import numpy as np
from itertools import combinations, product

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Faster score computation using numpy for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all sums
    sums = np.add.outer(U_arr, U_arr)
    sums_set = np.unique(sums).size
    
    # Compute all differences (including negatives)
    diffs = np.subtract.outer(U_arr, U_arr)
    diffs_set = np.unique(diffs).size
    
    ratio = diffs_set / sums_set
    if ratio <= 1:
        return -1
        
    max_u = int(U_arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(base, exp_max, offset=0):
    """Generate a geometric-like set: {offset + base^k for k=0..exp_max}"""
    U = []
    for k in range(exp_max + 1):
        U.append(offset + base ** k)
    U = sorted(set(U))
    if 0 not in U:
        U = sorted([0] + U)
    return U

def generate_candidates(deadline):
    """Generate diverse candidate sets."""
    candidates = []
    
    # Geometric sets with various bases
    for base in [2, 3, 4, 5, 7, 10]:
        for exp_max in range(3, 12):
            if time.time() > deadline - 30:
                return candidates
            U = geometric_set(base, exp_max)
            candidates.append(U)
    
    # Mixed geometric sets
    for base1 in [2, 3, 5]:
        for base2 in [3, 4, 7]:
            for exp1 in range(2, 8):
                for exp2 in range(2, 8):
                    if time.time() > deadline - 30:
                        return candidates
                    U1 = geometric_set(base1, exp1)
                    U2 = geometric_set(base2, exp2)
                    U = sorted(set(U1) | set(U2))
                    if 0 not in U:
                        U = [0] + U
                    candidates.append(U)
    
    # Arithmetic progressions with varying step
    for step in [2, 3, 5, 7]:
        for length in range(5, 20):
            if time.time() > deadline - 30:
                return candidates
            U = list(range(0, length * step, step))
            candidates.append(U)
    
    # Quadratic sets
    for coeff in [1, 2, 3]:
        for length in range(5, 25):
            if time.time() > deadline - 30:
                return candidates
            U = [coeff * i * i for i in range(length)]
            if 0 not in U:
                U = [0] + U
            candidates.append(U)
    
    # Random structured sets
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 30:
            return candidates
        n = random.randint(5, 25)
        max_val = random.randint(50, 500)
        U = [0]
        for _ in range(n - 1):
            U.append(random.randint(1, max_val))
        U = sorted(set(U))
        candidates.append(U)
    
    return candidates

def local_search_dynamic(U, deadline, max_iters=20000):
    """
    Dynamic local search that adapts its strategy based on the current set's properties.
    Uses multiple move types with adaptive probabilities.
    """
    U = sorted(set(U))
    current_score = score_set_fast(U)
    best_score_local = current_score
    best_set_local = list(U)
    
    T = 5.0
    cooling = 0.997
    stagnation = 0
    max_stagnation = 500
    
    move_probs = [0.35, 0.35, 0.20, 0.10]  # replace, add, remove, restructure
    
    for i in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Adapt move probabilities based on set size
        n = len(U)
        if n < 8:
            move_probs = [0.25, 0.50, 0.10, 0.15]
        elif n > 25:
            move_probs = [0.45, 0.15, 0.30, 0.10]
        
        # Choose a move
        r = random.random()
        cumulative = 0
        move_type = 0
        for k, p in enumerate(move_probs):
            cumulative += p
            if r <= cumulative:
                move_type = k
                break
        
        new_U = list(U)
        
        if move_type == 0:
            # Replace an element
            if len(new_U) < 2:
                continue
            idx = random.randint(1, len(new_U) - 1)
            old_val = new_U[idx]
            
            # Try several replacement strategies
            if random.random() < 0.5:
                # Local perturbation
                delta = random.randint(-max(1, old_val // 10), max(1, old_val // 10))
                new_val = old_val + delta
            else:
                # Global random
                new_val = random.randint(1, max(10, int(max(new_U) * 1.5)))
            
            if new_val <= 0 or new_val in U:
                continue
            new_U[idx] = new_val
            new_U = sorted(set(new_U))
            
        elif move_type == 1:
            # Add an element
            if len(new_U) < 35:
                max_val = max(new_U)
                if random.random() < 0.4:
                    # Near existing elements
                    base = random.choice(new_U[1:])
                    new_val = base + random.randint(-3, 3)
                elif random.random() < 0.5:
                    # Near max
                    new_val = random.randint(max(1, max_val - 5), max_val + 20)
                else:
                    new_val = random.randint(1, max(10, max_val))
                
                if new_val <= 0 or new_val in U:
                    continue
                new_U.append(new_val)
                new_U = sorted(set(new_U))
                
        elif move_type == 2:
            # Remove an element
            if len(new_U) > 3:
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
                new_U = sorted(set(new_U))
                
        else:
            # Restructure: swap two elements
            if len(new_U) >= 3:
                i, j = random.sample(range(1, len(new_U)), 2)
                new_U[i], new_U[j] = new_U[j], new_U[i]
                new_U = sorted(set(new_U))
        
        if len(new_U) < 2:
            continue
            
        new_score = score_set_fast(new_U)
        
        # Simulated annealing acceptance
        if new_score > current_score:
            U = new_U
            current_score = new_score
            stagnation = 0
            if current_score > best_score_local:
                best_score_local = current_score
                best_set_local = list(U)
        else:
            delta = new_score - current_score
            if delta > -0.5 and random.random() < np.exp(delta / T):
                U = new_U
                current_score = new_score
                stagnation += 1
            else:
                stagnation += 1
        
        T *= cooling
        
        # Restart if stuck
        if stagnation > max_stagnation:
            T = max(T, 2.0)
            stagnation = 0
            
    return best_score_local, best_set_local

def final_polish(U, deadline):
    """High-precision polish pass with tighter tolerance."""
    U = sorted(set(U))
    current_score = score_set_fast(U)
    n = len(U)
    
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        
        # Try replacing each element with nearby values (tighter range)
        for i in range(1, n):
            if time.time() > deadline - 1:
                break
            old_val = U[i]
            for delta in range(-8, 9):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0 or new_val in U:
                    continue
                new_set = [U[j] if j != i else new_val for j in range(n)]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = score_set_fast(new_set)
                if new_score > current_score + 1e-12:
                    U = new_set
                    current_score = new_score
                    n = len(U)
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved and n < 35:
            max_val = max(U)
            for new_val in range(max(1, max_val - 8), max_val + 12):
                if new_val in U:
                    continue
                new_set = sorted(U + [new_val])
                new_score = score_set_fast(new_set)
                if new_score > current_score + 1e-12:
                    U = new_set
                    current_score = new_score
                    n = len(U)
                    improved = True
                    break
        
        # Try removing an element
        if not improved and n > 4:
            for i in range(1, n):
                new_set = [U[j] for j in range(n) if j != i]
                new_score = score_set_fast(new_set)
                if new_score > current_score + 1e-12:
                    U = new_set
                    current_score = new_score
                    n = len(U)
                    improved = True
                    break
        
        # Try swapping pairs
        if not improved and n >= 4:
            for i in range(1, n):
                for j in range(i + 1, n):
                    new_set = list(U)
                    new_set[i], new_set[j] = new_set[j], new_set[i]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set_fast(new_set)
                    if new_score > current_score + 1e-12:
                        U = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    best_set = [0, 1]
    best_score = score_set_fast(best_set)

    # Generate diverse candidates
    candidates = generate_candidates(deadline)
    
    for cand in candidates:
        if time.time() > deadline - 40:
            break
        s = score_set_fast(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Run local search from best and top candidates
    scored_candidates = [(score_set_fast(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: -x[0])
    
    search_seeds = [best_set]
    for s, c in scored_candidates[:10]:
        if c != best_set and s > best_score * 0.85:
            search_seeds.append(c)
    
    random.shuffle(search_seeds)
    
    for seed in search_seeds:
        if time.time() > deadline - 20:
            break
        local_score, local_set = local_search_dynamic(seed, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final high-precision polish
    if time.time() < deadline - 2:
        polished_set, polished_score = final_polish(best_set, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished_set

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
