import time
import random
from math import log
from itertools import combinations, product

def score_set(U):
    """
    Computes the score for a set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
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
    
    # Using sets directly is often faster than lists for small n
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
        best_candidate_ratio = -1
        
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
                best_candidate_ratio = s
        
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
    deadline = start_time + 140  # 10s safety margin
    
    best_score = -1
    best_set = [0, 1]
    
    # Phase 1: Greedy Construction from different seeds
    # Try different starting sizes and max values
    seeds = [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 4],
        [0, 1, 3, 9],
    ]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        s, U = greedy_construction(20, 200, deadline - 20)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Phase 2: Simulated Annealing on the best found and some random restarts
    # Use the best set from Phase 1
    if time.time() < deadline - 10:
        s, U = simulated_annealing_search(best_set, deadline - 10, temp_start=0.5, cool_rate=0.99)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Phase 3: More SA restarts with different initializations
    # Random geometric seeds
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        # Generate a random geometric-ish set
        b = random.uniform(1.2, 2.5)
        m = random.randint(5, 15)
        U_seed = [0]
        seen = {0}
        for k in range(1, m+1):
            val = int(b ** k)
            if val not in seen and val > 0:
                U_seed.append(val)
                seen.add(val)
        U_seed.sort()
        
        s, U = simulated_annealing_search(U_seed, deadline - 5, temp_start=0.5, cool_rate=0.99)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Phase 4: Local Polish (Small moves) on the global best
    # This helps fine-tune the discrete values
    if time.time() < deadline - 1:
        current = list(best_set)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            n = len(current)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try small perturbations
                for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = current.copy()
                    new_set[i] = new_val
                    new_set.sort()
                    # Check for duplicates
                    if len(set(new_set)) != n:
                        continue
                    new_score = fast_score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        if current_score > best_score:
            best_score = current_score
            best_set = current

    # Ensure 0 is in the set and it's a list of nonnegative integers
    best_set = sorted(list(set(best_set)))
    if 0 not in best_set:
        best_set.insert(0, 0)
        
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
