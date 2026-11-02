import itertools
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = 1.0281
    
    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        n = len(U)
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(n):
                sums.add(U[i] + U[j])
                diffs.add(U[i] - U[j])
        
        size_s = len(sums)
        size_d = len(diffs)
        
        if size_s == 0:
            return 0.0
            
        ratio = size_d / size_s
        if ratio <= 1.0:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    def local_search(U, max_iter=500):
        """Refine a set by trying to add, remove, or swap elements"""
        U = sorted(set(U))
        if not U or 0 not in U:
            return U, 1.0
        current_score = calc_score(U)
        max_u = max(U)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding a new element
            for new_val in range(max_u + 1, max_u + 30):
                candidate = sorted(U + [new_val])
                s = calc_score(candidate)
                if s > current_score:
                    U = candidate
                    current_score = s
                    max_u = new_val
                    improved = True
                    break
            if improved:
                continue
            
            # Try removing an element (not 0)
            for i in range(1, len(U)):
                candidate = U[:i] + U[i+1:]
                if len(candidate) < 3:
                    continue
                s = calc_score(candidate)
                if s > current_score:
                    U = candidate
                    current_score = s
                    max_u = max(U) if U else 0
                    improved = True
                    break
            if improved:
                continue
            
            # Try swapping an element
            for i in range(1, len(U)):
                old_val = U[i]
                for new_val in range(1, max_u + 5):
                    if new_val == old_val or new_val in U:
                        continue
                    candidate = sorted(U[:i] + [new_val] + U[i+1:])
                    s = calc_score(candidate)
                    if s > current_score:
                        U = candidate
                        current_score = s
                        max_u = max(U)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return U, current_score

    # Try small sets
    # We can try all subsets of {0, 1, ..., K} of size N
    # But N and K are unknown.
    # Let's try a few heuristics.
    
    # Heuristic 1: Geometric Progressions (without duplicate 0)
    for r in range(2, 12):
        for n in range(3, 10):
            U = [r**i for i in range(1, n+1)]  # Start from r^1 to avoid 0
            U = sorted([0] + U)
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            if time.time() > deadline:
                return best_set
    
    # Heuristic 2: Random sets with small max
    random.seed(42)
    
    for n in range(3, 14):
        for max_val in range(n, 60):
            if time.time() > deadline:
                return best_set
            for _ in range(500):
                if time.time() > deadline:
                    return best_set
                # Generate n-1 random numbers from 1..max_val
                candidates = list(range(1, max_val + 1))
                if len(candidates) < n - 1:
                    continue
                sample = random.sample(candidates, n - 1)
                U = sorted([0] + sample)
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Heuristic 3: Exhaustive search for very small sets
    # n=3, 4, 5 with small max
    for n in range(3, 7):
        for max_val in range(n, 25):
            if time.time() > deadline:
                return best_set
            # Generate all combinations of n-1 elements from 1..max_val
            range_vals = list(range(1, max_val + 1))
            if len(range_vals) < n - 1:
                continue
            for comb in itertools.combinations(range_vals, n - 1):
                if time.time() > deadline:
                    return best_set
                U = sorted([0] + list(comb))
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Heuristic 4: Local search refinement on best found
    if time.time() < deadline - 20:
        refined_set, refined_score = local_search(best_set, max_iter=1000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    # Heuristic 5: Try structured sets based on known good patterns
    # Sets with arithmetic progressions or mixed structures
    for n in range(4, 12):
        if time.time() > deadline:
            return best_set
        # Try sets with small gaps followed by larger gaps
        for base in range(1, 5):
            for spread in range(2, 15):
                U = [0]
                for i in range(1, n):
                    if i <= base:
                        U.append(i)
                    else:
                        U.append(base + (i - base) * spread)
                U = sorted(set(U))
                if len(U) != n:
                    continue
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
                if time.time() > deadline:
                    return best_set
    
    # Final local search polish
    if time.time() < deadline - 10:
        refined_set, refined_score = local_search(best_set, max_iter=2000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    return best_set
