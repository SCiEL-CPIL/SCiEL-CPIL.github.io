"""Dual-view construction: define U via difference structure, minimize additive energy for given |U-U|."""
import numpy as np
import math
import random

def construct():
    """
    Dual-view approach: 
    - View U through its difference set D = U-U
    - Optimize by finding U that maximizes |D|/|U+U| ratio
    - Key insight: for fixed |D|, minimize |U+U| by creating sum collisions
    - Use structured initial candidates + efficient local search
    """
    
    def compute_score(U):
        """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
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
    
    def compute_score_fast(U):
        """Faster score computation using numpy for larger sets"""
        if not U or 0 not in U:
            return 0.0
        U_arr = np.array(sorted(set(U)))
        max_u = U_arr[-1]
        if max_u == 0:
            return 0.0
        # Compute all sums and differences
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def local_search(U, max_iters=300, max_val=400):
        """Local search with multiple move types, optimized for speed."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score_fast(U)
        U_set = set(best_U)
        n = len(best_U)
        
        for iteration in range(max_iters):
            improved = False
            
            # Move type 1: Single element relocation
            for i in range(1, n):
                elem = best_U[i]
                # Try nearby values
                for delta in range(-10, 11):
                    new_val = elem + delta
                    if new_val < 0 or new_val in U_set or new_val > max_val:
                        continue
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    new_score = compute_score_fast(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        U_set = set(best_U)
                        n = len(best_U)
                        improved = True
                        break
                if improved:
                    break
            
            # Move type 2: Two-element swap
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        e1, e2 = best_U[i], best_U[j]
                        for d1 in range(-4, 5):
                            for d2 in range(-4, 5):
                                new_v1 = e1 + d1
                                new_v2 = e2 + d2
                                if new_v1 < 0 or new_v2 < 0 or new_v1 > max_val or new_v2 > max_val:
                                    continue
                                if new_v1 in U_set or new_v2 in U_set or new_v1 == new_v2:
                                    continue
                                new_U = list(best_U)
                                new_U[i] = new_v1
                                new_U[j] = new_v2
                                new_U = sorted(set(new_U))
                                if 0 not in new_U or len(new_U) < 2:
                                    continue
                                new_score = compute_score_fast(new_U)
                                if new_score > best_score + 1e-10:
                                    best_score = new_score
                                    best_U = new_U
                                    U_set = set(best_U)
                                    n = len(best_U)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Move type 3: Add element (targeted)
            if not improved:
                U_set = set(best_U)
                # Try adding elements that create sum collisions
                for candidate in range(1, min(max_val + 1, 150)):
                    if candidate in U_set:
                        continue
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score_fast(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        U_set = set(best_U)
                        n = len(best_U)
                        improved = True
                        break
            
            # Move type 4: Remove element
            if not improved:
                for i in range(1, n):
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    new_score = compute_score_fast(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        U_set = set(best_U)
                        n = len(best_U)
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_U, best_score
    
    # Generate initial candidates using dual-view insight
    candidates = []
    
    # Strategy: Create sets where differences are spread out (large |U-U|) 
    # but sums collide (small |U+U|)
    
    # 1. Mixed base constructions: combine arithmetic and geometric
    for base_ar in [3, 4, 5, 6, 7, 8]:
        for base_geo in [2, 3]:
            U = [0]
            current = 0
            for i in range(8):
                if i % 2 == 0:
                    current += base_ar
                else:
                    current += base_geo ** (i // 2 + 1)
                U.append(current)
            U = sorted(set(U))
            if len(U) >= 3:
                candidates.append(U)
    
    # 2. Perturbed Sidon sets: start with Mian-Chowla, then perturb to create sum collisions
    random.seed(123)
    for trial in range(15):
        # Start with a Sidon-like sequence
        U = [0, 1]
        for i in range(2, 12):
            # Find next element that maintains Sidon property
            best_elem = None
            for candidate in range(U[-1] + 1, U[-1] + 20):
                # Check if adding candidate preserves distinct differences
                valid = True
                for a in U:
                    for b in U:
                        if a != b:
                            if candidate - a == b - a or candidate - a == a - b:
                                valid = False
                                break
                            if candidate - b == a - b or candidate - b == b - a:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    best_elem = candidate
                    break
            if best_elem is None:
                best_elem = U[-1] + random.randint(1, 5)
            U.append(best_elem)
        
        # Now perturb to create sum collisions
        for _ in range(5):
            idx = random.randint(1, len(U) - 1)
            delta = random.choice([-2, -1, 1, 2, 3])
            new_val = U[idx] + delta
            if new_val > 0 and new_val not in U:
                U[idx] = new_val
                U = sorted(set(U))
                if 0 not in U:
                    U = [0] + U
        
        if len(U) >= 3:
            candidates.append(U)
    
    # 3. Structured sets with controlled collision patterns
    # Use the fact that if U = {a_i * k + r_i}, sums collide when k is large
    for k in [5, 7, 10, 13, 17]:
        for n in range(5, 11):
            U = [0]
            for i in range(1, n):
                # Create elements that are close to multiples of k
                offset = random.randint(0, k - 1)
                U.append(i * k + offset)
            U = sorted(set(U))
            if len(U) >= 3:
                candidates.append(U)
    
    # 4. Random greedy with dual-view scoring
    random.seed(456)
    for trial in range(12):
        max_val = random.randint(150, 350)
        U = [0]
        U_set = {0}
        # Track difference set to maximize |U-U|
        diff_set = {0}
        
        for _ in range(random.randint(8, 14)):
            best_candidate = None
            best_score = float('-inf')
            
            # Sample candidates
            for _ in range(80):
                candidate = random.randint(1, max_val)
                if candidate in U_set:
                    continue
                
                new_U = sorted(U + [candidate])
                new_score = compute_score_fast(new_U)
                
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = candidate
            
            if best_candidate is not None and best_score > float('-inf'):
                U.append(best_candidate)
                U_set.add(best_candidate)
        
        if len(U) >= 3:
            candidates.append(U)
    
    # 5. Hybrid: combine known good structures
    # Take elements from different types of sets
    random.seed(789)
    for trial in range(10):
        U = {0}
        # Add some arithmetic progression elements
        base = random.randint(3, 8)
        for i in range(random.randint(3, 6)):
            U.add(i * base)
        # Add some geometric elements
        for i in range(random.randint(2, 4)):
            U.add(2 ** (i + 1))
        # Add some random elements
        for _ in range(random.randint(2, 5)):
            U.add(random.randint(1, 100))
        
        U = sorted(U)
        if len(U) >= 4:
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
        
        searched_U, searched_score = local_search(U, max_iters=250, max_val=350)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
