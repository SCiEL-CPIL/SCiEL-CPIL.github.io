"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Start with Sidon-like structures (maximize distinct differences)
    - Use local search to minimize |U+U| while preserving |U-U|
    - Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom

    def local_search(U, max_iters=200, max_val=100):
        """Local search to minimize max(U) and |U+U| while keeping |U-U| high."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Try swapping each element (except 0) with a smaller value to reduce max(U)
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                # Try replacing with any value not in U, smaller than current
                # to reduce max(U) if possible, or just to improve score
                candidates = list(range(1, elem))
                random.shuffle(candidates)
                for new_val in candidates:
                    if new_val in U_set:
                        continue
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
            
            # Try small perturbations
            if not improved:
                for i, elem in enumerate(best_U):
                    if elem == 0:
                        continue
                    for delta in [-1, 1]:
                        new_val = elem + delta
                        if new_val < 0 or new_val in U_set:
                            continue
                        if new_val > max_val:
                            continue
                        new_U = list(best_U)
                        new_U[i] = new_val
                        new_U = sorted(set(new_U))
                        if 0 not in new_U or len(new_U) < 2:
                            continue
                        new_score = compute_score(new_U)
                        if new_score > best_score + 1e-10:
                            best_score = new_score
                            best_U = new_U
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_U, best_score

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Known Sidon-like sequences (Mian-Chowla style)
    # These are optimal for maximizing |U-U| for small max U
    candidates.append([0, 1])
    candidates.append([0, 1, 3])
    candidates.append([0, 1, 3, 7])
    candidates.append([0, 1, 3, 7, 15])
    candidates.append([0, 1, 3, 7, 15, 31])
    candidates.append([0, 1, 3, 7, 15, 31, 63])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40])
    candidates.append([0, 1, 4, 10, 18, 23, 31])
    
    # 2. Other small Sidon sets
    candidates.append([0, 1, 4, 10])
    candidates.append([0, 1, 4, 10, 18])
    candidates.append([0, 1, 4, 10, 18, 23])
    
    # 3. Random constructions with Sidon bias
    random.seed(42)
    for trial in range(50):
        n = random.randint(3, 8)
        max_val = random.randint(20, 50)
        U = [0]
        U_set = {0}
        for _ in range(n - 1):
            best_candidate = None
            best_score_candidate = float('-inf')
            for _ in range(50):
                candidate = random.randint(1, max_val)
                if candidate in U_set:
                    continue
                # Check Sidon property roughly
                valid = True
                for a in U_set:
                    for b in U_set:
                        if a < b:
                            if candidate + a == b: # Sum collision
                                valid = False
                                break
                        # Diff collision
                        if candidate - a == b - a: # candidate == b
                            pass
                # Just compute score for quick filter
                new_U = sorted(U + [candidate])
                s = compute_score(new_U)
                if s > best_score_candidate:
                    best_score_candidate = s
                    best_candidate = candidate
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
        U = sorted(U)
        candidates.append(U)
    
    # Run local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    
    for U in candidates:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        searched_U, searched_score = local_search(U, max_iters=100, max_val=50)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
