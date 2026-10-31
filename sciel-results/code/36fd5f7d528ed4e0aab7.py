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
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        # Compute sumset
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        # Compute difference set
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
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
    
    # Add some structured non-Golomb candidates (e.g. powers of 2, Fibonacci)
    for max_u in [10, 20, 50, 100]:
        # Powers of 2
        geo = [0, 1]
        while geo[-1] * 2 <= max_u:
            geo.append(geo[-1] * 2)
        if max(geo) <= max_u:
            candidates.append(geo)
        
        # Fibonacci
        fib = [0, 1]
        while fib[-1] + fib[-2] <= max_u:
            fib.append(fib[-1] + fib[-2])
        if max(fib) <= max_u:
            candidates.append(fib)

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

    # Phase 2: Maximum Weight Independent Set on Conflict Graph
    # Vertices: candidate integers (0..max_candidate)
    # Edges: pairs that create "sum collisions" (a+b=c+d with {a,b} != {c,d})
    # We want to select a subset U containing 0 that minimizes |U+U| while keeping |U-U| large
    
    rng = random.Random(42)
    
    # Try different max_candidate values
    for max_candidate in [50, 100, 200]:
        if time.time() > deadline - 30:
            break
        
        # Build conflict graph: edge between x,y if adding both creates a sum collision
        # A sum collision occurs if there exist a,b,c,d in U with a+b=c+d and {a,b} != {c,d}
        # For efficiency, we'll use a greedy approach with conflict detection
        
        # Start with 0 and build up greedily
        U = [0]
        
        # Candidate pool: all integers from 1 to max_candidate
        candidates_pool = list(range(1, max_candidate + 1))
        rng.shuffle(candidates_pool)
        
        # Greedy construction: add elements that don't create too many sum collisions
        for candidate in candidates_pool:
            if time.time() > deadline - 30:
                break
            
            # Check if adding this candidate improves the score
            test_U = sorted(U + [candidate])
            if len(test_U) > 15:  # Limit set size for computational tractability
                continue
            
            s = compute_score(test_U)
            
            # Accept if it improves score significantly or maintains Golomb property
            if s > best_score + 1e-10:
                best_score = s
                best_set = list(test_U)
                U = list(test_U)
            elif is_golomb(test_U) and s >= best_score - 0.01:
                # Allow non-improving moves if we maintain Golomb (structural sparsity)
                U = list(test_U)
        
        # Local search refinement on the constructed set
        current = list(best_set) if best_set else [0]
        current_is_golomb = is_golomb(current)
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline - 20 and no_improve < max_no_improve:
            improved = False
            n = len(current)
            
            # Try moving each element by small deltas
            indices = list(range(1, n))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 20:
                    break
                orig = current[i]
                
                # Try a range of deltas
                for delta in [-3, -2, -1, 1, 2, 3, -5, 5]:
                    new_val = orig + delta
                    if new_val < 0 or new_val in current:
                        continue
                    
                    new_U = list(current)
                    new_U[i] = new_val
                    new_U = sorted(new_U)
                    
                    if 0 not in new_U:
                        continue
                    
                    s = compute_score(new_U)
                    
                    if current_is_golomb:
                        if is_golomb(new_U):
                            if s > best_score + 1e-10:
                                best_score = s
                                best_set = list(new_U)
                                current = list(new_U)
                                improved = True
                                break
                        else:
                            if s > best_score + 0.005:
                                best_score = s
                                best_set = list(new_U)
                                current = list(new_U)
                                current_is_golomb = False
                                improved = True
                                break
                    else:
                        if s > best_score + 1e-10:
                            best_score = s
                            best_set = list(new_U)
                            current = list(new_U)
                            current_is_golomb = is_golomb(new_U)
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                # Random perturbation if stuck
                if rng.random() < 0.15 and n > 3:
                    idx = rng.randint(1, n - 1)
                    new_val = max(1, current[idx] + rng.randint(-8, 8))
                    if new_val not in current:
                        new_U = list(current)
                        new_U[idx] = new_val
                        new_U = sorted(new_U)
                        if 0 in new_U:
                            s = compute_score(new_U)
                            if s > best_score:
                                best_score = s
                                best_set = list(new_U)
                                current = list(new_U)
                                current_is_golomb = is_golomb(new_U)
                                no_improve = 0

    # Phase 3: Final add/remove refinement
    if time.time() < deadline - 10:
        current = list(best_set)
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            improved = False
            
            # Try adding an element
            if len(current) < 14:
                max_cur = max(current)
                candidates_to_add = list(range(1, max_cur + 15))
                rng.shuffle(candidates_to_add)
                for new_el in candidates_to_add[:15]:
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            # Try removing an element
            if len(current) > 3:
                for i in range(1, len(current)):
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            if not improved:
                # Random replacement
                if len(current) >= 3:
                    idx = rng.randint(1, len(current) - 1)
                    new_el = rng.randint(1, max(current) + 10)
                    if new_el not in current:
                        cand = list(current)
                        cand[idx] = new_el
                        cand = sorted(set(cand))
                        if 0 in cand:
                            s = compute_score(cand)
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                                current = list(cand)

    # Final check
    return list(best_set)
# EVOLVE-BLOCK-END
