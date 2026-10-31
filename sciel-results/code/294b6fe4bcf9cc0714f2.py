"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Build U greedily by selecting elements that maximize new differences while minimizing new sums
    - Use efficient incremental computation of |U-U| and |U+U|
    - Apply local search to refine the solution
    - Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    
    def compute_score(U):
        """Compute the score for a given set U."""
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
    
    def greedy_construct(target_size, max_val, seed=42):
        """
        Greedy construction using dual-view:
        - Track difference set D = U - U
        - Track sum set S = U + U
        - At each step, choose candidate that maximizes new_diffs - lambda * new_sums
        """
        random.seed(seed)
        U = [0]
        U_set = {0}
        diff_set = {0}
        sum_set = {0}
        
        for _ in range(target_size - 1):
            best_candidate = None
            best_score = float('-inf')
            
            # Try all candidates in range
            for candidate in range(1, max_val + 1):
                if candidate in U_set:
                    continue
                
                # Compute new differences and sums
                new_diffs = 0
                new_sums = 0
                for a in U_set:
                    d1 = candidate - a
                    d2 = a - candidate
                    s = candidate + a
                    if d1 not in diff_set:
                        new_diffs += 1
                    if d2 not in diff_set:
                        new_diffs += 1
                    if s not in sum_set:
                        new_sums += 1
                
                # Score: prefer more new differences, fewer new sums
                # Also penalize large candidates slightly to keep max_u small
                candidate_score = new_diffs * 2.0 - new_sums * 1.0 - candidate * 0.05
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_candidate = candidate
            
            if best_candidate is None:
                break
            
            U.append(best_candidate)
            U_set.add(best_candidate)
            
            # Update diff_set and sum_set
            for a in U_set:
                diff_set.add(best_candidate - a)
                diff_set.add(a - best_candidate)
                sum_set.add(best_candidate + a)
        
        return sorted(U)
    
    def local_search(U, max_iters=300, max_val=600):
        """
        Local search with efficient score computation.
        Try: move, swap, add, remove operations.
        """
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Try moving each element (except 0)
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                for delta in range(-15, 16):
                    new_val = elem + delta
                    if new_val < 0 or new_val in U_set or new_val > max_val:
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
            
            # Try 2-move: change two elements simultaneously
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        e1, e2 = best_U[i], best_U[j]
                        for d1 in range(-8, 9):
                            for d2 in range(-8, 9):
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
            
            # Try adding a new element
            if not improved:
                U_set = set(best_U)
                # Sample candidates for speed
                candidates_to_try = list(range(1, min(max_val + 1, 300)))
                random.shuffle(candidates_to_try)
                for candidate in candidates_to_try[:80]:
                    if candidate in U_set:
                        continue
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            # Try removing an element (except 0)
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
            
            if not improved:
                break
        
        return best_U, best_score
    
    # Generate diverse initial candidates using greedy construction with different parameters
    candidates = []
    
    # Greedy constructions with varying parameters
    for target_size in [8, 10, 12, 14, 16]:
        for max_val in [150, 250, 400, 600]:
            for seed in [42, 123, 456, 789]:
                U = greedy_construct(target_size, max_val, seed)
                if len(U) >= 2:
                    candidates.append(U)
    
    # Also include some known good structures
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100, 115])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61])
    candidates.append([0, 1, 3, 7, 12, 20, 30, 44, 60, 79, 101])
    
    # Run local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    
    for U in candidates:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        searched_U, searched_score = local_search(U, max_iters=250, max_val=500)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
