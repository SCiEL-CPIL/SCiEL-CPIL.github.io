"""Dual-view construction: define U via difference-set structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Define U implicitly through its difference structure
    - Minimize |U+U| while maximizing |U-U|
    - Keep max(U) small to minimize the log denominator
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
    
    def local_search(U, max_iters=500, max_val=200):
        """Local search to optimize score."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Precompute
            cur_sum_set = set()
            cur_diff_set = set()
            for a in U_set:
                for b in U_set:
                    cur_sum_set.add(a + b)
                    cur_diff_set.add(a - b)
            
            # Try moving each element
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                
                best_new_val = None
                best_new_score = best_score
                
                for new_val in range(max(1, elem - 10), min(max_val + 1, elem + 11)):
                    if new_val == elem or new_val in U_set:
                        continue
                    
                    # Quick estimate
                    new_diff_count = 0
                    new_sum_count = 0
                    for a in U_set:
                        if a == elem:
                            continue
                        d1 = new_val - a
                        d2 = a - new_val
                        if d1 not in cur_diff_set:
                            new_diff_count += 1
                        if d2 not in cur_diff_set:
                            new_diff_count += 1
                        s = new_val + a
                        if s not in cur_sum_set:
                            new_sum_count += 1
                    
                    net_gain = new_diff_count - new_sum_count * 0.9
                    if net_gain > 0:
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
            
            # Try adding
            if not improved:
                best_candidate = None
                best_candidate_score = best_score
                
                for candidate in range(1, min(max_val + 1, 150)):
                    if candidate in U_set:
                        continue
                    
                    new_diff_count = 0
                    new_sum_count = 0
                    for a in U_set:
                        d1 = candidate - a
                        d2 = a - candidate
                        if d1 not in cur_diff_set:
                            new_diff_count += 1
                        if d2 not in cur_diff_set:
                            new_diff_count += 1
                        s = candidate + a
                        if s not in cur_sum_set:
                            new_sum_count += 1
                    
                    net_gain = new_diff_count - new_sum_count * 0.8
                    if net_gain > 0:
                        new_U = sorted(best_U + [candidate])
                        new_score = compute_score(new_U)
                        if new_score > best_candidate_score + 1e-10:
                            best_candidate_score = new_score
                            best_candidate = candidate
                
                if best_candidate is not None:
                    best_U = sorted(best_U + [best_candidate])
                    best_score = best_candidate_score
                    improved = True
            
            # Try removing
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
    
    random.seed(42)
    
    best_U = [0, 1]
    best_score = 0.0
    
    # Strategy: Start with near-AP sets and perturb to increase |U-U| while keeping |U+U| small
    # Key insight: small perturbations of an AP can make differences unique while sums remain clustered
    
    for trial in range(30):
        n = random.randint(8, 20)
        base = random.randint(3, 12)
        
        # Start with AP
        U = [i * base for i in range(n)]
        
        # Apply small random perturbations to break AP structure
        # This increases |U-U| while keeping |U+U| relatively small
        for i in range(1, len(U)):
            if random.random() < 0.7:  # Perturb most elements
                U[i] += random.randint(-2, 2)
        
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        U = [x for x in U if x >= 0]
        U = sorted(U)
        
        if len(U) >= 2:
            searched_U, searched_score = local_search(U, max_iters=400, max_val=180)
            if searched_score > best_score:
                best_score = searched_score
                best_U = searched_U
    
    # Also try structured constructions: "twisted" APs
    for base in [4, 5, 6, 7, 8]:
        for n in range(8, 18):
            U = []
            for i in range(n):
                # Twisted: add a small quadratic term
                val = i * base + (i * i) % base
                U.append(val)
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            U = [x for x in U if x >= 0]
            U = sorted(U)
            
            if len(U) >= 2:
                searched_U, searched_score = local_search(U, max_iters=400, max_val=180)
                if searched_score > best_score:
                    best_score = searched_score
                    best_U = searched_U
    
    # Try Mian-Chowla-like sequences (Sidon sets)
    for seed_val in [1, 2, 3]:
        random.seed(seed_val)
        U = [0, 1]
        U_set = {0, 1}
        diffs = {1, -1}
        
        for _ in range(25):
            best_candidate = None
            best_metric = -float('inf')
            
            for candidate in range(2, 100):
                if candidate in U_set:
                    continue
                
                # Check if adding candidate keeps Sidon property (all diffs unique)
                is_sidon = True
                for a in U_set:
                    d1 = candidate - a
                    d2 = a - candidate
                    if d1 in diffs or d2 in diffs:
                        is_sidon = False
                        break
                
                if not is_sidon:
                    continue
                
                # Count new sums
                new_sums = 0
                for a in U_set:
                    s = candidate + a
                    # Check if this sum already exists
                    exists = False
                    for b in U_set:
                        for c in U_set:
                            if b + c == s:
                                exists = True
                                break
                        if exists:
                            break
                    if not exists:
                        new_sums += 1
                
                # We want few new sums
                metric = -new_sums
                if metric > best_metric:
                    best_metric = metric
                    best_candidate = candidate
            
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
                for a in U_set:
                    diffs.add(best_candidate - a)
                    diffs.add(a - best_candidate)
        
        U = sorted(U)
        searched_U, searched_score = local_search(U, max_iters=400, max_val=150)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
