import time
from math import log, floor
import random
from itertools import combinations

def score_set(U):
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

def fast_score_set(U):
    """
    Slightly optimized scoring for frequent calls during search.
    Assumes U is a sorted list of unique non-negative ints containing 0.
    """
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
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def exhaustive_search_small(n, max_val, deadline):
    """
    Exhaustive search for sets of size n containing 0, with max element <= max_val.
    Returns the best (score, set) found within time budget.
    """
    best_score = -1
    best_set = None
    
    if n - 1 > max_val:
        return best_score, best_set
    
    candidates = list(range(1, max_val + 1))
    
    for combo in combinations(candidates, n - 1):
        if time.time() > deadline:
            break
        U = [0] + list(combo)
        s = fast_score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    return best_score, best_set

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def greedy_construction(target_size, max_val, deadline):
    """
    Greedily adds elements to maximize the ratio |U-U|/|U+U| at each step.
    """
    U = [0]
    best_score = -1
    best_set = [0]
    
    candidates = list(range(1, max_val + 1))
    
    while len(U) < target_size and time.time() < deadline:
        best_candidate = None
        best_candidate_score = -1
        
        if len(candidates) > 50:
            sample = random.sample(candidates, 50)
        else:
            sample = candidates
            
        for c in sample:
            new_U = U + [c]
            new_U.sort()
            s = fast_score_set(new_U)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = c
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
            if best_candidate in candidates:
                candidates.remove(best_candidate)
            
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_set = list(U)
        else:
            break
            
    return best_score, best_set

def ising_search(U_init, deadline, temp_start=1.0, cool_rate=0.98):
    """
    Simulated Annealing with Ising-like moves.
    Moves: Add, Remove, Replace.
    Uses a dynamic candidate pool generated from sums and differences.
    """
    U = list(U_init)
    U.sort()
    current_score = fast_score_set(U)
    best_score = current_score
    best_set = list(U)
    
    temp = temp_start
    max_val_limit = max(U) * 2 + 100 if U else 100
    
    # Generate initial candidate pool
    def generate_candidates(U_set, limit_val):
        candidates = set()
        for a in U_set:
            for b in U_set:
                s = a + b
                d = abs(a - b)
                if 0 < s <= limit_val and s not in U_set:
                    candidates.add(s)
                if 0 < d <= limit_val and d not in U_set:
                    candidates.add(d)
        return list(candidates)
    
    U_set = set(U)
    candidates = generate_candidates(U_set, max_val_limit)
    
    while time.time() < deadline and temp > 1e-4:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            if len(U) < 30 and candidates:
                # Pick a random candidate
                random.shuffle(candidates)
                new_val = candidates[0]
                if new_val not in U:
                    new_U = U + [new_val]
                    new_U.sort()
                    new_score = fast_score_set(new_U)
                    if new_score > current_score or random.random() < pow(2.718, (new_score - current_score) / temp):
                        U = new_U
                        U_set.add(new_val)
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(U)
                        # Regenerate candidates
                        max_val_limit = max(max_val_limit, max(U) * 2)
                        candidates = generate_candidates(U_set, max_val_limit)
        elif move_type < 0.7:
            # Remove
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                new_U = U[:idx] + U[idx+1:]
                new_score = fast_score_set(new_U)
                if new_score > current_score or random.random() < pow(2.718, (new_score - current_score) / temp):
                    U = new_U
                    U_set.remove(old_val)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(U)
                    candidates = generate_candidates(U_set, max_val_limit)
        else:
            # Replace
            if len(U) > 2 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                new_val = candidates[0]
                if new_val != old_val and new_val not in U:
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = fast_score_set(new_U)
                    if new_score > current_score or random.random() < pow(2.718, (new_score - current_score) / temp):
                        U = new_U
                        U_set.remove(old_val)
                        U_set.add(new_val)
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(U)
                        candidates = generate_candidates(U_set, max_val_limit)
        
        temp *= cool_rate
        
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 0. Exhaustive search for small sets to find strong seeds
    if time.time() < deadline - 90:
        ex_score, ex_set = exhaustive_search_small(3, 30, deadline - 90)
        if ex_set and ex_score > best_score:
            best_score = ex_score
            best_set = ex_set
        
        if time.time() < deadline - 70:
            ex_score, ex_set = exhaustive_search_small(4, 20, deadline - 70)
            if ex_set and ex_score > best_score:
                best_score = ex_score
                best_set = ex_set
        
        if time.time() < deadline - 50:
            ex_score, ex_set = exhaustive_search_small(5, 15, deadline - 50)
            if ex_set and ex_score > best_score:
                best_score = ex_score
                best_set = ex_set

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    for cand in structured_candidates:
        s = fast_score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Greedy Construction
    greedy_seeds = [
        [0, 1],
        [0, 1, 4],
        [0, 1, 3, 9],
        best_set
    ]
    
    for seed in greedy_seeds:
        if time.time() > deadline - 30:
            break
        s, U = greedy_construction(20, 200, deadline - 30)
        if s > best_score:
            best_score = s
            best_set = U

    # 3. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 30:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = fast_score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 4. Ising Search (Simulated Annealing with dynamic candidates)
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        sa_score, sa_set = ising_search(U, deadline - 15, temp_start=0.5, cool_rate=0.99)
        
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set
            if len(seeds) < 5:
                seeds.append(sa_set)
            else:
                seeds[-1] = sa_set

    # 5. Final Polish: Fine-grained local search on the best set
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 500:
            improved = False
            iteration += 1
            n = len(current)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = fast_score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved and n < 25 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 10, max_val + 11):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = fast_score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = fast_score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
