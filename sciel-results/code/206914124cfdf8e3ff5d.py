"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    def compute_score(U):
        U = sorted(set(int(x) for x in U if x >= 0))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        # Compute sumset
        sumset = set()
        diffset = set()
        n = len(U)
        for i in range(n):
            for j in range(n):
                sumset.add(U[i] + U[j])
                diffset.add(U[i] - U[j])
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 1.0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def is_golomb(U):
        diffs = set()
        n = len(U)
        for i in range(n):
            for j in range(i + 1, n):
                d = U[j] - U[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Known optimal Golomb rulers
    optimal_rulers = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
    }

    best_score = 0
    best_set = [0, 1, 3]

    # Phase 1: Evaluate known Golomb rulers and some structured sets
    candidates = []
    for n, ruler in optimal_rulers.items():
        candidates.append(list(ruler))
    
    # Add some structured non-Golomb candidates
    for max_u in [10, 20, 50, 100]:
        geo = [0, 1]
        while geo[-1] * 2 <= max_u:
            geo.append(geo[-1] * 2)
        if max(geo) <= max_u:
            candidates.append(geo)
        
        fib = [0, 1]
        while fib[-1] + fib[-2] <= max_u:
            fib.append(fib[-1] + fib[-2])
        if max(fib) <= max_u:
            candidates.append(fib)

    # Additional hand-crafted candidates
    extra_sets = [
        [0, 1, 3, 7],
        [0, 1, 4, 6, 10],
        [0, 2, 5, 11],
        [0, 1, 5, 11],
        [0, 3, 7, 12],
        [0, 1, 4, 9, 11, 18],
        [0, 2, 7, 12, 18],
        [0, 1, 5, 10, 16],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 8, 13],
        [0, 2, 5, 11, 17],
        [0, 1, 5, 11, 17],
    ]
    candidates.extend(extra_sets)

    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(sorted(c))

    # Score all initial candidates
    for c in unique_candidates:
        s = compute_score(c)
        if s > best_score:
            best_score = s
            best_set = list(c)

    rng = random.Random(42)

    # Phase 2: Multi-restart local search with diverse move operators
    max_restarts = 8
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        # Choose starting point
        if restart == 0:
            current = list(best_set)
        elif restart < len(optimal_rulers):
            n = list(optimal_rulers.keys())[restart]
            current = list(optimal_rulers[n])
        else:
            # Random perturbation of best set
            current = list(best_set)
            if len(current) > 2:
                idx = rng.randint(1, len(current) - 1)
                current[idx] = max(1, current[idx] + rng.randint(-5, 5))
                current = sorted(set(current))
                if 0 not in current:
                    current = list(best_set)
        
        current_score = compute_score(current)
        
        # Local search phase
        no_improve = 0
        max_no_improve = 300
        
        while time.time() < deadline - 15 and no_improve < max_no_improve:
            improved = False
            n = len(current)
            
            # Choose a random move type
            move_type = rng.randint(0, 3)
            
            if move_type == 0 and n > 2:
                # Remove a random element (not 0)
                idx = rng.randint(1, n - 1)
                new_U = list(current)
                new_U.pop(idx)
                new_U = sorted(new_U)
                if 0 in new_U and len(new_U) >= 2:
                    s = compute_score(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_set = list(new_U)
                        current = list(new_U)
                        current_score = s
                        no_improve = 0
                        improved = True
                    elif s > current_score - 0.005 and rng.random() < 0.1:
                        current = list(new_U)
                        current_score = s
                        no_improve = 0
                        improved = True

            elif move_type == 1:
                # Add a random element
                if n < 14:
                    max_val = max(current)
                    upper = max(2 * max_val + 10, 20)
                    new_val = rng.randint(1, upper)
                    if new_val not in current:
                        new_U = sorted(current + [new_val])
                        s = compute_score(new_U)
                        if s > best_score + 1e-10:
                            best_score = s
                            best_set = list(new_U)
                            current = list(new_U)
                            current_score = s
                            no_improve = 0
                            improved = True
                        elif s > current_score - 0.005 and rng.random() < 0.05:
                            current = list(new_U)
                            current_score = s
                            no_improve = 0
                            improved = True

            elif move_type == 2 and n > 1:
                # Modify a random element to a nearby value
                idx = rng.randint(1, n - 1)
                orig = current[idx]
                # Try a range of deltas
                deltas = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -8, 8, -12, 12]
                rng.shuffle(deltas)
                for delta in deltas:
                    new_val = orig + delta
                    if new_val < 1 or new_val in current:
                        continue
                    new_U = list(current)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    s = compute_score(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_set = list(new_U)
                        current = list(new_U)
                        current_score = s
                        no_improve = 0
                        improved = True
                        break
                    elif s > current_score - 0.003 and rng.random() < 0.08:
                        current = list(new_U)
                        current_score = s
                        no_improve = 0
                        improved = True
                        break

            else:
                # Swap: remove one element and add another
                if n > 2:
                    idx = rng.randint(1, n - 1)
                    removed_val = current[idx]
                    new_U = list(current)
                    new_U.pop(idx)
                    max_val = max(new_U) if new_U else 0
                    upper = max(2 * max_val + 10, 20)
                    new_val = rng.randint(1, upper)
                    if new_val not in new_U:
                        new_U.append(new_val)
                        new_U = sorted(new_U)
                        s = compute_score(new_U)
                        if s > best_score + 1e-10:
                            best_score = s
                            best_set = list(new_U)
                            current = list(new_U)
                            current_score = s
                            no_improve = 0
                            improved = True
                        elif s > current_score - 0.005 and rng.random() < 0.05:
                            current = list(new_U)
                            current_score = s
                            no_improve = 0
                            improved = True
            
            if not improved:
                no_improve += 1
                # Random perturbation if stuck
                if no_improve % 50 == 0 and n > 3:
                    idx = rng.randint(1, n - 1)
                    new_val = max(1, current[idx] + rng.randint(-10, 10))
                    if new_val not in current:
                        new_U = list(current)
                        new_U[idx] = new_val
                        new_U = sorted(set(new_U))
                        if 0 in new_U:
                            s = compute_score(new_U)
                            if s > best_score:
                                best_score = s
                                best_set = list(new_U)
                                current = list(new_U)
                                current_score = s
                                no_improve = 0

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = compute_score(current)
        
        # Fine-grained local search with small deltas
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            n = len(current)
            
            if n < 2:
                break
            
            indices = list(range(1, n))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5:
                    break
                orig = current[i]
                
                # Try small deltas
                for delta in [-2, -1, 1, 2]:
                    new_val = orig + delta
                    if new_val < 1 or new_val in current:
                        continue
                    
                    new_U = list(current)
                    new_U[i] = new_val
                    new_U = sorted(new_U)
                    
                    s = compute_score(new_U)
                    
                    if s > best_score + 1e-10:
                        best_score = s
                        best_set = list(new_U)
                        current = list(new_U)
                        current_score = s
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try adding/removing
                if n < 13:
                    max_val = max(current)
                    for new_val in range(1, max_val + 5):
                        if new_val in current:
                            continue
                        new_U = sorted(current + [new_val])
                        s = compute_score(new_U)
                        if s > best_score + 1e-10:
                            best_score = s
                            best_set = list(new_U)
                            current = list(new_U)
                            current_score = s
                            improved = True
                            break
                
                if not improved and n > 3:
                    for i in range(1, n):
                        new_U = [x for j, x in enumerate(current) if j != i]
                        s = compute_score(new_U)
                        if s > best_score + 1e-10:
                            best_score = s
                            best_set = list(new_U)
                            current = list(new_U)
                            current_score = s
                            improved = True
                            break

    return list(best_set)
# EVOLVE-BLOCK-END
