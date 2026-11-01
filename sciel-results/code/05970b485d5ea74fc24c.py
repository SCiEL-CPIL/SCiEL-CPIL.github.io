"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Start with structured sets (APs) that minimize |U+U|
    - Use difference-set-guided perturbations to increase |U-U|
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
    
    def local_search(U, max_iters=1000, max_val_limit=50):
        """
        Efficient local search optimizing the score directly.
        Focuses on small max_val to keep denominator small.
        """
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            return U, 0.0
            
        best_U = list(U)
        best_score = compute_score(U)
        
        # Adaptive max_val: start small, allow growth if score improves significantly
        current_max_val = max(best_U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # 1. Try moving elements
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                
                # Search neighborhood
                # Bias towards smaller values to reduce denominator
                start = max(1, elem - 5)
                end = min(current_max_val + 10, elem + 5)
                
                best_move_val = None
                best_move_score = best_score
                
                for new_val in range(start, end + 1):
                    if new_val == elem or new_val in U_set:
                        continue
                    # Avoid creating negative numbers or exceeding limit
                    if new_val < 0 or new_val > current_max_val + 10:
                        continue
                        
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U:
                        continue
                        
                    new_score = compute_score(new_U)
                    if new_score > best_move_score + 1e-9:
                        best_move_score = new_score
                        best_move_val = new_val
                
                if best_move_val is not None:
                    best_U = list(best_U)
                    best_U[i] = best_move_val
                    best_U = sorted(set(best_U))
                    best_score = best_move_score
                    current_max_val = max(best_U)
                    improved = True
                    break
            
            # 2. Try adding an element
            if not improved:
                best_add_val = None
                best_add_score = best_score
                
                # Candidates: small numbers (to keep max low) or gaps
                candidates = list(range(1, current_max_val + 5))
                random.shuffle(candidates)
                
                for candidate in candidates[:50]:
                    if candidate in U_set:
                        continue
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score(new_U)
                    if new_score > best_add_score + 1e-9:
                        best_add_score = new_score
                        best_add_val = candidate
                
                if best_add_val is not None:
                    best_U = sorted(best_U + [best_add_val])
                    best_score = best_add_score
                    current_max_val = max(best_U)
                    improved = True
            
            # 3. Try removing an element
            if not improved:
                for i, elem in enumerate(best_U):
                    if elem == 0:
                        continue
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    new_score = compute_score(new_U)
                    if new_score > best_score + 1e-9:
                        best_U = new_U
                        best_score = new_score
                        current_max_val = max(best_U)
                        improved = True
                        break
                        
            if not improved:
                # Stagnation: try random restart with a new seed or slightly different start
                # To avoid getting stuck, we can break and let the outer loop try another start
                break
                
        return best_U, best_score

    best_U = [0, 1]
    best_score = 0.0
    
    # Generate diverse initial candidates
    # Strategy: Start with small sets and grow them, or start with specific structures.
    
    # 1. Small APs
    for n in range(3, 15):
        U = list(range(n))
        U, score = local_search(U, max_iters=200, max_val_limit=50)
        if score > best_score:
            best_score = score
            best_U = U
            
    # 2. Small Sidon-like / Geometric
    bases = [2, 3, 4, 5]
    for base in bases:
        U = [0]
        for i in range(1, 10):
            val = base ** i
            if val > 100: break
            U.append(val)
        U, score = local_search(U, max_iters=200, max_val_limit=100)
        if score > best_score:
            best_score = score
            best_U = U
            
    # 3. Random sets with small max
    random.seed(42)
    for _ in range(50):
        max_val = random.randint(10, 40)
        n = random.randint(4, 12)
        U = [0]
        while len(U) < n:
            v = random.randint(1, max_val)
            if v not in U:
                U.append(v)
        U = sorted(U)
        U, score = local_search(U, max_iters=200, max_val_limit=60)
        if score > best_score:
            best_score = score
            best_U = U
            
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
