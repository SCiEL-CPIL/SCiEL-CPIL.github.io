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

    def local_search_fast(U, max_iters=500, max_val=400):
        """
        Optimized local search using incremental set updates.
        """
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score(U)
        
        # Precompute sets for current best
        U_set = set(best_U)
        cur_sum_set = set()
        cur_diff_set = set()
        for a in U_set:
            for b in U_set:
                cur_sum_set.add(a + b)
                cur_diff_set.add(a - b)
        
        n_diff = len(cur_diff_set)
        n_sum = len(cur_sum_set)
        max_u = max(U_set)
        
        for iteration in range(max_iters):
            improved = False
            # Randomize the order of elements to avoid bias
            indices = list(range(len(best_U)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx == 0 or best_U[idx] == 0:
                    continue
                
                elem = best_U[idx]
                # Try moving the element
                # We check a window around the current element and some random jumps
                candidates = set(range(max(1, elem - 20), min(max_val + 1, elem + 21)))
                # Add some random candidates to escape local minima
                for _ in range(10):
                    candidates.add(random.randint(1, max_val))
                
                best_new_val = None
                best_new_score = best_score
                
                for new_val in candidates:
                    if new_val == elem or new_val in U_set:
                        continue
                    
                    # Incremental update logic
                    # 1. Identify sums/diffs that ONLY involve 'elem' and will be removed
                    # 2. Identify new sums/diffs that will be added with 'new_val'
                    
                    # Sums/Diffs to remove: x + elem or x - elem / elem - x for x in U_set \ {elem}
                    # Sums/Diffs to add: x + new_val or x - new_val / new_val - x for x in U_set \ {elem}
                    
                    # To be safe and simple, we re-calculate the sets if the heuristic suggests improvement,
                    # but we use a heuristic to filter candidates first.
                    
                    # Heuristic: Estimate change in |U-U| and |U+U|
                    # This is an approximation.
                    # Let's just do a full recalculation for the top few candidates to keep code simple but effective.
                    # Given N is small (< 30), O(N^2) is fast enough.
                    
                    # Optimization: Only do full calc if new_val is "promising".
                    # A move is promising if it likely increases diff/sum ratio.
                    # We can't easily know without calculation, so we just try all candidates in the window.
                    
                    new_U = list(best_U)
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    
                    # Fast score calculation
                    new_U_set = set(new_U)
                    new_max = max(new_U_set)
                    new_sum_set = set()
                    new_diff_set = set()
                    for a in new_U_set:
                        for b in new_U_set:
                            new_sum_set.add(a + b)
                            new_diff_set.add(a - b)
                    
                    if not new_sum_set:
                        continue
                        
                    new_n_diff = len(new_diff_set)
                    new_n_sum = len(new_sum_set)
                    
                    log_denom = math.log(2 * new_max + 1)
                    if log_denom <= 0:
                        continue
                        
                    new_score = 1.0 + math.log(new_n_diff / new_n_sum) / log_denom
                    
                    if new_score > best_new_score + 1e-10:
                        best_new_score = new_score
                        best_new_val = new_val
                
                if best_new_val is not None:
                    best_U[idx] = best_new_val
                    best_U = sorted(set(best_U))
                    best_score = best_new_score
                    improved = True
                    
                    # Update precomputed sets for next iteration
                    U_set = set(best_U)
                    cur_sum_set = set()
                    cur_diff_set = set()
                    for a in U_set:
                        for b in U_set:
                            cur_sum_set.add(a + b)
                            cur_diff_set.add(a - b)
                    n_diff = len(cur_diff_set)
                    n_sum = len(cur_sum_set)
                    max_u = max(U_set)
                    break # Restart loop with new state
            
            # If no move improved, try adding an element
            if not improved:
                # Try adding a new element
                best_add_val = None
                best_add_score = best_score
                
                # Sample candidates for addition
                add_candidates = set(range(1, min(max_val + 1, 100)))
                for _ in range(50):
                    add_candidates.add(random.randint(1, max_val))
                
                for cand in add_candidates:
                    if cand in U_set:
                        continue
                    
                    new_U = sorted(best_U + [cand])
                    new_U_set = set(new_U)
                    new_max = max(new_U_set)
                    new_sum_set = set()
                    new_diff_set = set()
                    for a in new_U_set:
                        for b in new_U_set:
                            new_sum_set.add(a + b)
                            new_diff_set.add(a - b)
                    
                    if not new_sum_set:
                        continue
                        
                    new_n_diff = len(new_diff_set)
                    new_n_sum = len(new_sum_set)
                    
                    log_denom = math.log(2 * new_max + 1)
                    if log_denom <= 0:
                        continue
                        
                    new_score = 1.0 + math.log(new_n_diff / new_n_sum) / log_denom
                    
                    if new_score > best_add_score + 1e-10:
                        best_add_score = new_score
                        best_add_val = cand
                
                if best_add_val is not None:
                    best_U = sorted(best_U + [best_add_val])
                    best_score = best_add_score
                    improved = True
                    
                    U_set = set(best_U)
                    cur_sum_set = set()
                    cur_diff_set = set()
                    for a in U_set:
                        for b in U_set:
                            cur_sum_set.add(a + b)
                            cur_diff_set.add(a - b)
                    n_diff = len(cur_diff_set)
                    n_sum = len(cur_sum_set)
                    max_u = max(U_set)
            
            # If no move improved, try removing an element
            if not improved:
                for i in range(1, len(best_U)):
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    new_U_set = set(new_U)
                    new_max = max(new_U_set)
                    new_sum_set = set()
                    new_diff_set = set()
                    for a in new_U_set:
                        for b in new_U_set:
                            new_sum_set.add(a + b)
                            new_diff_set.add(a - b)
                    
                    if not new_sum_set:
                        continue
                        
                    new_n_diff = len(new_diff_set)
                    new_n_sum = len(new_sum_set)
                    
                    log_denom = math.log(2 * new_max + 1)
                    if log_denom <= 0:
                        continue
                        
                    new_score = 1.0 + math.log(new_n_diff / new_n_sum) / log_denom
                    
                    if new_score > best_score + 1e-10:
                        best_U = new_U
                        best_score = new_score
                        improved = True
                        
                        U_set = set(best_U)
                        cur_sum_set = set()
                        cur_diff_set = set()
                        for a in U_set:
                            for b in U_set:
                                cur_sum_set.add(a + b)
                                cur_diff_set.add(a - b)
                        n_diff = len(cur_diff_set)
                        n_sum = len(cur_sum_set)
                        max_u = max(U_set)
                        break
            
            if not improved:
                break
        
        return best_U, best_score
    
    # Generate diverse initial candidates focused on structured sets
    candidates = []
    
    # 1. Arithmetic progressions with perturbations (start structured, then diversify)
    for base in [4, 5, 6, 7, 8, 9, 10, 12, 14]:
        for n in range(5, 18):
            U = [i * base for i in range(n)]
            # Light perturbation to break perfect AP structure
            perturbation_strength = random.randint(1, 3)
            for i in range(1, len(U)):
                U[i] += random.randint(-perturbation_strength, perturbation_strength)
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 2:
                candidates.append(U)
    
    # 2. Geometric-like sequences (good for large max, controlled sums)
    candidates.append([0, 1, 3, 9, 27, 81, 243])
    candidates.append([0, 1, 3, 7, 15, 31, 63, 127])
    candidates.append([0, 1, 4, 16, 64, 256])
    candidates.append([0, 2, 5, 12, 29, 66, 149])
    
    # 3. Mian-Chowla / Sidon sequences
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100])
    candidates.append([0, 1, 3, 7, 12, 20, 30, 44, 60, 79])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61])
    
    # 4. Hybrid: start with AP, then add sparse large elements
    for base in [5, 7, 9]:
        for n in range(4, 10):
            U = [i * base for i in range(n)]
            # Add a few sparse large elements
            for _ in range(random.randint(1, 3)):
                U.append(random.randint(n * base + 10, n * base + 100))
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 2:
                candidates.append(U)
    
    # 5. Random constructions with structure bias
    random.seed(123)
    for trial in range(20):
        n = random.randint(6, 15)
        max_val = random.randint(150, 350)
        U = [0]
        U_set = {0}
        
        # Start with a small structured core
        core_size = random.randint(3, 5)
        base = random.randint(3, 8)
        for i in range(core_size):
            val = i * base + random.randint(0, 2)
            if val not in U_set and val > 0:
                U.append(val)
                U_set.add(val)
        
        # Add more elements with bias toward creating new differences
        for _ in range(n - len(U)):
            best_candidate = None
            best_net_gain = -float('inf')
            
            for _ in range(50):
                candidate = random.randint(1, max_val)
                if candidate in U_set:
                    continue
                
                # Estimate new diffs vs sums
                new_diffs = 0
                new_sums = 0
                for a in U_set:
                    d1 = candidate - a
                    d2 = a - candidate
                    # Check if these diffs already exist
                    diff_exists = False
                    sum_exists = False
                    for b in U_set:
                        if candidate - a == b - a or a - candidate == b - a:
                            diff_exists = True
                            break
                    for b in U_set:
                        if candidate + a == b + a:
                            sum_exists = True
                            break
                    if not diff_exists:
                        new_diffs += 1
                    if not sum_exists:
                        new_sums += 1
                
                net_gain = new_diffs - new_sums * 0.8
                if net_gain > best_net_gain:
                    best_net_gain = net_gain
                    best_candidate = candidate
            
            if best_candidate is not None and best_net_gain > 0:
                U.append(best_candidate)
                U_set.add(best_candidate)
        
        U = sorted(U)
        candidates.append(U)
        
    # 6. Modular constructions
    for mod in [7, 11, 13]:
        for n in range(5, 12):
            U = [0]
            U_set = {0}
            for i in range(1, n):
                # Pick values that are distinct mod mod
                val = i * mod + random.randint(0, 2)
                if val not in U_set:
                    U.append(val)
                    U_set.add(val)
            candidates.append(sorted(U))

    # Run fast local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    
    for U in candidates:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        searched_U, searched_score = local_search_fast(U, max_iters=300, max_val=400)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
