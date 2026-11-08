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
    This is different from local search which moves around a fixed size.
    It builds up the set from scratch.
    """
    U = [0]
    best_score = -1
    best_set = [0]
    
    # Pre-compute potential candidates
    candidates = list(range(1, max_val + 1))
    
    while len(U) < target_size and time.time() < deadline:
        best_candidate = None
        best_candidate_score = -1
        
        # To save time, don't check all candidates if we have many.
        # But for small max_val, checking all is fine.
        # We prioritize candidates that create new differences without creating new sums?
        # Actually, just score the new set.
        
        # Optimization: Sample if candidates are too many
        if len(candidates) > 50:
            sample = random.sample(candidates, 50)
        else:
            sample = candidates
            
        for c in sample:
            new_U = U + [c]
            new_U.sort()
            # Quick check: if c is too close to existing, it might not add much diff
            s = fast_score_set(new_U)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = c
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
            # Remove used candidate
            if best_candidate in candidates:
                candidates.remove(best_candidate)
            
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_set = list(U)
        else:
            break
            
    return best_score, best_set

def simulated_annealing_search(U_init, deadline, temp_start=1.0, cool_rate=0.95):
    """
    Simulated Annealing to escape local optima.
    Moves: Add, Remove, Replace.
    """
    U = list(U_init)
    U.sort()
    current_score = fast_score_set(U)
    best_score = current_score
    best_set = list(U)
    
    temp = temp_start
    max_val_limit = max(U) * 2 + 100 if U else 100
    
    while time.time() < deadline and temp > 1e-4:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            if len(U) < 25:
                # Pick a random new element
                # Bias towards larger numbers to increase D? Or smaller to increase ratio?
                # Large D reduces the log term denominator, which might hurt if ratio doesn't grow fast enough.
                # But large D also allows larger ratios.
                # Let's try a mix.
                if random.random() < 0.5:
                    new_val = random.randint(1, max_val_limit)
                else:
                    new_val = random.randint(1, max(U) + 10) if U else 10
                if new_val not in U:
                    new_U = U + [new_val]
                    new_U.sort()
                    new_score = fast_score_set(new_U)
                    if new_score > current_score or random.random() < pow(2.718, (new_score - current_score) / temp):
                        U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(U)
        elif move_type < 0.7:
            # Remove
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                new_U = U[:idx] + U[idx+1:]
                new_score = fast_score_set(new_U)
                if new_score > current_score or random.random() < pow(2.718, (new_score - current_score) / temp):
                    U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(U)
        else:
            # Replace
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                new_val = random.randint(1, max_val_limit)
                if new_val != old_val and new_val not in U:
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = fast_score_set(new_U)
                    if new_score > current_score or random.random() < pow(2.718, (new_score - current_score) / temp):
                        U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(U)
        
        temp *= cool_rate
        
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 0. Exhaustive search for small sets to find strong seeds
    # For n=3, max_val=30: C(30,2)=435 combinations
    # For n=4, max_val=20: C(20,3)=1140 combinations
    # For n=5, max_val=15: C(15,4)=1365 combinations
    if time.time() < deadline - 80:
        ex_score, ex_set = exhaustive_search_small(3, 30, deadline - 80)
        if ex_set and ex_score > best_score:
            best_score = ex_score
            best_set = ex_set
        
        if time.time() < deadline - 60:
            ex_score, ex_set = exhaustive_search_small(4, 20, deadline - 60)
            if ex_set and ex_score > best_score:
                best_score = ex_score
                best_set = ex_set
        
        if time.time() < deadline - 40:
            ex_score, ex_set = exhaustive_search_small(5, 15, deadline - 40)
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

    # 2. Greedy Construction (from Inspiration)
    # Use greedy construction to build up from strong seeds
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
    # Quick coarse grid to find a good geometric seed
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

    # 4. Simulated Annealing (from Inspiration)
    # Use the best found so far as the primary seed, and structured ones as secondary seeds
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run simulated annealing
        sa_score, sa_set = simulated_annealing_search(U, deadline - 15, temp_start=0.5, cool_rate=0.99)
        
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set
            # Add the new best to seeds for future restarts? 
            # To keep it simple, we just update the primary seed list
            if len(seeds) < 5:
                seeds.append(sa_set)
            else:
                seeds[-1] = sa_set

    # 5. Final Polish: Fine-grained local search on the best set
    # This addresses the "refinement machinery" requirement.
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 500:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
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
            
            # Try adding an element near the max
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

            # Try removing an element
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

# Ensure 0 is in the set and it's a list of nonnegative integers
# The function construct returns a list.
# The evaluator calls construct().

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
