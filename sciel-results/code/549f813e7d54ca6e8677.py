"""Dual-view construction: define U by difference set D, minimize |U+U| while preserving |U-U|."""
import numpy as np
import math
import random
import time

def construct():
    """
    Dual-view approach:
    1. Generate Sidon-like sets (all pairwise differences distinct) to maximize |U-U|
    2. Optimize by finding elements that minimize |U+U| while preserving difference structure
    3. Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
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
    
    def get_diff_sum_sets(U):
        """Return (diff_set, sum_set) for U."""
        U_list = list(U)
        sum_set = set()
        diff_set = set()
        for a in U_list:
            for b in U_list:
                sum_set.add(a + b)
                diff_set.add(a - b)
        return diff_set, sum_set
    
    def generate_sidon_set(n, max_val, seed=None):
        """Generate a Sidon set of size n (all pairwise differences distinct)."""
        if seed is not None:
            random.seed(seed)
        
        U = [0]
        U_set = {0}
        
        for _ in range(n - 1):
            best_candidate = None
            best_score = -float('inf')
            
            # Try many random candidates
            for _ in range(200):
                candidate = random.randint(1, max_val)
                if candidate in U_set:
                    continue
                
                # Check if adding candidate preserves Sidon property
                # All differences candidate - u and u - candidate for u in U must be new
                valid = True
                new_diffs = set()
                for u in U_set:
                    d1 = candidate - u
                    d2 = u - candidate
                    new_diffs.add(d1)
                    new_diffs.add(d2)
                
                # Check against existing differences
                existing_diffs = set()
                for a in U_set:
                    for b in U_set:
                        if a != b:
                            existing_diffs.add(a - b)
                
                # If any new diff already exists, not Sidon
                if new_diffs & existing_diffs:
                    continue
                
                # Score: prefer candidates that create "spread" differences
                # (larger absolute differences give more room)
                score = sum(abs(d) for d in new_diffs) / len(new_diffs)
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
            else:
                break
        
        return sorted(U)
    
    def minimize_sumset(U, max_iters=500, max_val=500):
        """
        Local search to minimize |U+U| while preserving |U-U|.
        
        Key idea: for each element, try moving it to a position where:
        1. The difference set doesn't shrink (ideally stays same or grows)
        2. The sum set shrinks
        
        This is the dual-view optimization: D is roughly fixed, minimize S.
        """
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_diff_set, best_sum_set = get_diff_sum_sets(best_U)
        best_n_diff = len(best_diff_set)
        best_n_sum = len(best_sum_set)
        best_score = compute_score(best_U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            cur_diff_set, cur_sum_set = get_diff_sum_sets(best_U)
            cur_n_diff = len(cur_diff_set)
            cur_n_sum = len(cur_sum_set)
            
            # Try moving each element (except 0)
            for i, elem in enumerate(best_U):
                if elem == 0:
                    continue
                
                best_new_val = None
                best_new_score = best_score
                best_new_diff_count = cur_n_diff
                best_new_sum_count = cur_n_sum
                
                # Search range around current value
                search_radius = 20
                for new_val in range(max(1, elem - search_radius), min(max_val + 1, elem + search_radius + 1)):
                    if new_val == elem or new_val in U_set:
                        continue
                    
                    # Compute new diff and sum sets
                    new_diff_count = 0
                    new_sum_count = 0
                    
                    # Remove contributions from old elem
                    # Add contributions from new_val
                    
                    # New differences: new_val - a and a - new_val for a in U_set - {elem}
                    # Also need to remove elem - a and a - elem for a in U_set - {elem}
                    
                    # Efficient computation:
                    # New diff set = (old diff set - {elem-a, a-elem for a in U\{elem}}) ∪ {new_val-a, a-new_val for a in U\{elem}}
                    # This is complex, so just compute directly for candidate moves
                    
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    
                    new_diff_set, new_sum_set = get_diff_sum_sets(new_U)
                    new_n_diff = len(new_diff_set)
                    new_n_sum = len(new_sum_set)
                    
                    # Accept if:
                    # 1. Difference set doesn't shrink significantly
                    # 2. Sum set shrinks
                    # 3. Overall score improves
                    
                    if new_n_diff < cur_n_diff - 2:
                        continue  # Too much loss in differences
                    
                    # Prefer moves that reduce sum set
                    if new_n_sum >= cur_n_sum and new_n_diff <= cur_n_diff:
                        continue  # No improvement
                    
                    new_score = compute_score(new_U)
                    if new_score > best_new_score + 1e-10:
                        best_new_score = new_score
                        best_new_val = new_val
                        best_new_diff_count = new_n_diff
                        best_new_sum_count = new_n_sum
                
                if best_new_val is not None:
                    best_U[i] = best_new_val
                    best_U = sorted(set(best_U))
                    best_score = best_new_score
                    best_diff_set, best_sum_set = get_diff_sum_sets(best_U)
                    best_n_diff = len(best_diff_set)
                    best_n_sum = len(best_sum_set)
                    improved = True
                    break
            
            # Try adding a new element that preserves Sidon-like property
            if not improved:
                best_candidate = None
                best_candidate_score = best_score
                
                # Sample candidates from a range
                for candidate in range(1, min(max_val + 1, 300)):
                    if candidate in U_set:
                        continue
                    
                    new_U = sorted(best_U + [candidate])
                    
                    # Quick check: does this preserve good difference structure?
                    new_diff_set, new_sum_set = get_diff_sum_sets(new_U)
                    new_n_diff = len(new_diff_set)
                    new_n_sum = len(new_sum_set)
                    
                    # We want new differences to be "new" (not already in diff set)
                    # and new sums to be "old" (already in sum set)
                    
                    # Count how many new diffs and new sums
                    old_diff_set = best_diff_set
                    old_sum_set = best_sum_set
                    
                    truly_new_diffs = 0
                    truly_new_sums = 0
                    for d in new_diff_set:
                        if d not in old_diff_set:
                            truly_new_diffs += 1
                    for s in new_sum_set:
                        if s not in old_sum_set:
                            truly_new_sums += 1
                    
                    # Heuristic: prefer candidates that add few new sums relative to new diffs
                    if truly_new_sums > truly_new_diffs:
                        continue
                    
                    new_score = compute_score(new_U)
                    if new_score > best_candidate_score + 1e-10:
                        best_candidate_score = new_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    best_U = sorted(best_U + [best_candidate])
                    best_score = best_candidate_score
                    best_diff_set, best_sum_set = get_diff_sum_sets(best_U)
                    best_n_diff = len(best_diff_set)
                    best_n_sum = len(best_sum_set)
                    improved = True
            
            # Try removing an element that hurts the score
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
                        best_diff_set, best_sum_set = get_diff_sum_sets(best_U)
                        best_n_diff = len(best_diff_set)
                        best_n_sum = len(best_sum_set)
                        improved = True
                        break
            
            # Try 2-element move: move two elements simultaneously
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        e1, e2 = best_U[i], best_U[j]
                        for d1 in range(-10, 11):
                            for d2 in range(-10, 11):
                                if d1 == 0 and d2 == 0:
                                    continue
                                new_v1 = e1 + d1
                                new_v2 = e2 + d2
                                if new_v1 < 1 or new_v2 < 1:
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
                                    best_diff_set, best_sum_set = get_diff_sum_sets(best_U)
                                    best_n_diff = len(best_diff_set)
                                    best_n_sum = len(best_sum_set)
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
    
    # Generate diverse Sidon-like initial candidates
    random.seed(42)
    candidates = []
    
    # 1. Pure Sidon sets of various sizes
    for n in range(5, 20):
        for max_val in [100, 200, 300, 400]:
            sidon = generate_sidon_set(n, max_val, seed=random.randint(0, 10000))
            if len(sidon) >= 3:
                candidates.append(sidon)
    
    # 2. Mian-Chowla sequences (greedy Sidon)
    mc_sets = [
        [0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100, 115, 131, 148],
        [0, 1, 3, 7, 12, 20, 30, 44, 60, 79, 100, 123, 148, 175, 204],
        [0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100],
    ]
    for mc in mc_sets:
        for length in range(5, len(mc) + 1):
            candidates.append(mc[:length])
    
    # 3. Structured sets: APs with controlled perturbations
    for base in [5, 7, 9, 11, 13]:
        for n in range(5, 15):
            U = [i * base for i in range(n)]
            # Small perturbations to break perfect AP
            for i in range(1, len(U)):
                if random.random() < 0.3:
                    U[i] += random.choice([-1, 1, 2])
            U = sorted(set(U))
            if 0 in U and len(U) >= 3:
                candidates.append(U)
    
    # 4. Hybrid: small Sidon core + structured tail
    for core_n in range(4, 8):
        core = generate_sidon_set(core_n, 50, seed=random.randint(0, 10000))
        if len(core) < core_n:
            continue
        max_core = max(core)
        for tail_base in [10, 15, 20]:
            for tail_n in range(3, 10):
                tail = [max_core + i * tail_base for i in range(1, tail_n + 1)]
                U = sorted(set(core + tail))
                if 0 in U and len(U) >= 5:
                    candidates.append(U)
    
    # 5. Random constructions with Sidon bias
    for trial in range(30):
        n = random.randint(6, 18)
        max_val = random.randint(150, 400)
        U = [0]
        U_set = {0}
        existing_diffs = set()
        
        for _ in range(n - 1):
            best_candidate = None
            best_spread = -1
            
            for _ in range(100):
                candidate = random.randint(1, max_val)
                if candidate in U_set:
                    continue
                
                new_diffs = set()
                valid = True
                for u in U_set:
                    d1 = candidate - u
                    d2 = u - candidate
                    new_diffs.add(d1)
                    new_diffs.add(d2)
                    if d1 in existing_diffs or d2 in existing_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                spread = max(abs(d) for d in new_diffs)
                if spread > best_spread:
                    best_spread = spread
                    best_candidate = candidate
            
            if best_candidate is not None:
                U.append(best_candidate)
                U_set.add(best_candidate)
                # Update existing differences
                for u in U_set - {best_candidate}:
                    existing_diffs.add(best_candidate - u)
                    existing_diffs.add(u - best_candidate)
        
        if len(U) >= 3:
            candidates.append(sorted(U))
    
    # Run dual-view local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    start_time = time.time()
    time_limit = 100  # Leave buffer for evaluation
    
    for idx, U in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
        
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        
        # Adaptive iteration count based on time
        remaining_time = time_limit - (time.time() - start_time)
        max_iters = max(50, int(remaining_time * 3))
        
        searched_U, searched_score = minimize_sumset(U, max_iters=max_iters, max_val=450)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
