"""Dual-view construction: optimize |U-U|/|U+U| ratio by finding sets with spread-out differences but compact sums."""
import numpy as np
import math
import random

def construct():
    """
    Construct U to maximize 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    
    Key insight: We want |U-U| > |U+U|. This happens when the set is "spread out"
    (many unique differences) but "additively compact" (few unique sums).
    
    Strategy:
    1. Generate candidate sets with various structures
    2. Use local search that directly optimizes the score
    3. Focus on moves that increase |U-U|/|U+U| ratio
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
    
    def local_search(U, max_iters=500, max_val=400):
        """Local search optimizing the score directly."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Try moving each element
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                
                best_new_val = None
                best_new_score = best_score
                
                # Search a range of new values
                for new_val in range(max(1, elem - 25), min(max_val + 1, elem + 26)):
                    if new_val == elem or new_val in U_set:
                        continue
                    
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_new_score + 1e-10:
                        best_new_score = new_score
                        best_new_val = new_val
                
                if best_new_val is not None:
                    best_U[i] = best_new_val
                    best_U = sorted(set(best_U))
                    best_score = best_new_score
                    improved = True
                    break
            
            # Try adding a new element
            if not improved:
                U_set = set(best_U)
                best_candidate = None
                best_candidate_score = best_score
                
                # Sample candidates from a wider range
                candidates = list(range(1, min(max_val + 1, 350)))
                random.shuffle(candidates)
                
                for candidate in candidates[:120]:
                    if candidate in U_set:
                        continue
                    
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_candidate_score + 1e-10:
                        best_candidate_score = new_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    best_U = sorted(best_U + [best_candidate])
                    best_score = best_candidate_score
                    improved = True
            
            # Try removing an element
            if not improved:
                for i, elem in enumerate(best_U):
                    if elem == 0:
                        continue
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            # Try 2-element swap
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        e1, e2 = best_U[i], best_U[j]
                        for d1 in range(-12, 13):
                            for d2 in range(-12, 13):
                                new_v1 = e1 + d1
                                new_v2 = e2 + d2
                                if new_v1 < 0 or new_v2 < 0:
                                    continue
                                if new_v1 > max_val or new_v2 > max_val:
                                    continue
                                U_set_check = set(best_U)
                                if new_v1 in U_set_check or new_v2 in U_set_check:
                                    continue
                                if new_v1 == new_v2:
                                    continue
                                new_U = list(best_U)
                                new_U[i] = new_v1
                                new_U[j] = new_v2
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
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_U, best_score
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Small sets with known good ratio
    candidates.append([0, 1, 3])
    candidates.append([0, 1, 4, 10])
    candidates.append([0, 1, 3, 7])
    candidates.append([0, 1, 4, 10, 18])
    candidates.append([0, 2, 5, 12])
    candidates.append([0, 1, 5, 12, 24])
    candidates.append([0, 1, 3, 6, 10])
    candidates.append([0, 2, 5, 9, 14])
    
    # 2. Perturbed APs
    random.seed(42)
    for base in [2, 3, 4, 5, 6, 7, 8]:
        for n in range(3, 12):
            U = [i * base for i in range(n)]
            # Perturb to break AP structure
            for i in range(1, len(U)):
                U[i] += random.randint(-2, 2)
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 2:
                candidates.append(U)
    
    # 3. Random sets with bias toward spread-out elements
    for trial in range(40):
        n = random.randint(3, 12)
        max_val = random.randint(50, 250)
        U = [0]
        U_set = {0}
        
        for _ in range(n - 1):
            # Prefer larger gaps to increase |U-U|
            candidate = random.randint(1, max_val)
            if candidate not in U_set:
                U.append(candidate)
                U_set.add(candidate)
        
        U = sorted(U)
        candidates.append(U)
    
    # 4. Sets based on quadratic growth
    for a in [1, 2, 3]:
        for b in [0, 1, 2]:
            U = [a*i*i + b*i for i in range(10)]
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 2:
                candidates.append(U)
    
    # 5. Hybrid: small core + sparse large elements
    for base in [3, 4, 5]:
        for n_core in range(3, 7):
            U = [i * base for i in range(n_core)]
            for _ in range(random.randint(1, 3)):
                U.append(random.randint(n_core * base + 5, n_core * base + 80))
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 2:
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
        searched_U, searched_score = local_search(U, max_iters=400, max_val=350)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
