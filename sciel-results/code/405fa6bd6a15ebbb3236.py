import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Prescribed elements
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Helper: check if we can add element x to set a (with existing sums)
    def can_add(a, sums, x):
        for y in a:
            s = x + y
            if s in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def get_new_sums(a, x):
        return {x + y for y in a} | {2 * x}
    
    def compute_sums(a):
        s = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                s.add(x + y)
        return s
        
    def compute_conflicts(a, sums):
        """
        Returns a set of values x such that x cannot be added to a.
        """
        forbidden = set()
        for y in a:
            for s in sums:
                x = s - y
                if 1 <= x <= 284:
                    forbidden.add(x)
        for s in sums:
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= 284:
                    forbidden.add(x)
        return forbidden

    def smart_refill(a, sums, a_set, candidates, deadline):
        """
        Refills the set using a 'minimum impact' greedy heuristic.
        At each step, among valid candidates, pick the one that forbids the fewest other candidates.
        """
        a = list(a)
        sums = set(sums)
        a_set = set(a_set)
        
        # Precompute which candidates are currently valid
        forbidden = compute_conflicts(a, sums)
        
        # Filter candidates to those not in a and not forbidden
        active_candidates = [c for c in candidates if c not in a_set and c not in forbidden]
        
        while active_candidates:
            if time.time() > deadline:
                break
            
            best_cand = None
            min_impact = float('inf')
            best_new_sums = None
            best_new_a = None
            
            # Evaluate each active candidate
            for c in active_candidates:
                new_sums_candidate = get_new_sums(a, c)
                
                # Calculate impact: how many other active candidates become forbidden?
                impact = 0
                new_a_set_temp = a_set | {c}
                new_sums_temp = sums | new_sums_candidate
                
                for c2 in active_candidates:
                    if c2 == c:
                        continue
                    valid = True
                    for y in a:
                        if c2 + y in new_sums_temp:
                            valid = False
                            break
                    if valid:
                        if c2 + c in new_sums_temp:
                            valid = False
                        elif 2 * c2 in new_sums_temp:
                            valid = False
                    
                    if not valid:
                        impact += 1
                        
                if impact < min_impact:
                    min_impact = impact
                    best_cand = c
                    best_new_sums = new_sums_temp
                    best_new_a = a + [c]
                    
            if best_cand is None:
                break
                
            a = best_new_a
            a_set.add(best_cand)
            sums = best_new_sums
            forbidden = compute_conflicts(a, sums)
            active_candidates = [c for c in active_candidates if c != best_cand and c not in forbidden]
            
        return a, sums, a_set

    def greedy_expand_fast(initial_list, order, forbidden_cache=None):
        a = list(initial_list)
        sums = compute_sums(a)
        a_set = set(a)
        forbidden = compute_conflicts(a, sums)
        
        for x in order:
            if x in a_set:
                continue
            if x in forbidden:
                continue
            
            a.append(x)
            a_set.add(x)
            new_sums = get_new_sums(a, x)
            sums |= new_sums
            forbidden = compute_conflicts(a, sums)
            
        return a

    def try_add_to_set(a, sums, a_set, x):
        """Try to add x to the set, return (success, new_a, new_sums, new_a_set)"""
        if x in a_set:
            return False, a, sums, a_set
        for y in a:
            if x + y in sums:
                return False, a, sums, a_set
        if 2 * x in sums:
            return False, a, sums, a_set
        new_a = a + [x]
        new_sums = sums | get_new_sums(a, x)
        new_a_set = a_set | {x}
        return True, new_a, new_sums, new_a_set

    def strong_local_search(a, max_rounds=50, deadline=None):
        """
        Local search with 1-opt (remove+refill) and 2-opt (swap) moves.
        """
        a = list(a)
        sums = compute_sums(a)
        a_set = set(a)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            cur_len = len(a)
            
            # 1. Try to fill up using Smart Refill
            candidates = [x for x in range(1, 285) if x not in a_set]
            if candidates:
                a, sums, a_set = smart_refill(a, sums, a_set, candidates, deadline)
                if len(a) > cur_len:
                    improved = True
                    continue
            
            # 2. Try 1-opt: remove one non-fixed element and refill
            non_fixed_indices = [i for i in range(len(a)) if a[i] not in fixed_set]
            
            if not non_fixed_indices:
                break
            
            best_a = None
            best_len = len(a)
            
            # Try removing each non-fixed element (or a random subset if too many)
            if len(non_fixed_indices) <= 30:
                indices_to_try = non_fixed_indices
            else:
                indices_to_try = random.sample(non_fixed_indices, 30)
            
            for idx in indices_to_try:
                if time.time() > deadline:
                    break
                
                removed_val = a[idx]
                temp_a = [x for i, x in enumerate(a) if i != idx]
                temp_sums = compute_sums(temp_a)
                temp_set = set(temp_a)
                
                refill_candidates = [x for x in range(1, 285) if x not in temp_set]
                random.shuffle(refill_candidates)
                
                temp_a, temp_sums, temp_set = smart_refill(temp_a, temp_sums, temp_set, refill_candidates, deadline)
                
                if len(temp_a) > best_len:
                    best_len = len(temp_a)
                    best_a = (temp_a, temp_sums, temp_set)
                    if best_len >= 20:
                        break
            
            if best_a is not None:
                a, sums, a_set = best_a
                improved = True
                continue
            
            # 3. Try 2-opt: swap two non-fixed elements
            if len(non_fixed_indices) >= 2:
                # Try random pairs of non-fixed elements to swap
                num_swap_trials = min(50, len(non_fixed_indices) * (len(non_fixed_indices) - 1) // 2)
                
                for _ in range(num_swap_trials):
                    if time.time() > deadline:
                        break
                    
                    i, j = random.sample(non_fixed_indices, 2)
                    vi = a[i]
                    vj = a[j]
                    
                    # Remove both
                    temp_a = [x for k, x in enumerate(a) if k != i and k != j]
                    temp_sums = compute_sums(temp_a)
                    temp_set = set(temp_a)
                    
                    # Try to add vi and vj back in different positions, or add two new elements
                    # Strategy: remove vi and vj, then try to add vi and vj plus potentially more
                    # Actually, for a swap: remove vi, add some new element, or remove both and add two new ones
                    
                    # Let's try: remove vi and vj, refill
                    refill_candidates = [x for x in range(1, 285) if x not in temp_set]
                    random.shuffle(refill_candidates)
                    
                    temp_a, temp_sums, temp_set = smart_refill(temp_a, temp_sums, temp_set, refill_candidates, deadline)
                    
                    if len(temp_a) > best_len:
                        best_len = len(temp_a)
                        best_a = (temp_a, temp_sums, temp_set)
                        if best_len >= 20:
                            break
                
                if best_a is not None:
                    a, sums, a_set = best_a
                    improved = True
                    continue
            
            if not improved:
                break
            
        return a

    # Main search logic
    best = []
    
    # Strategy 1: Ascending order
    a1 = greedy_expand_fast(fixed_elements, range(12, 285))
    if len(a1) > len(best):
        best = a1
    
    # Strategy 2: Descending order
    a2 = greedy_expand_fast(fixed_elements, range(284, 11, -1))
    if len(a2) > len(best):
        best = a2

    # Strategy 3: Randomized greedy with local search
    random.seed(42)
    n_restarts = 150
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        a3 = greedy_expand_fast(fixed_elements, candidates)
        
        # Apply local search
        a3_improved = strong_local_search(a3, max_rounds=30, deadline=deadline)
        
        if len(a3_improved) > len(best):
            best = a3_improved
            if len(best) >= 20:
                break

    # Final polish
    best = strong_local_search(best, max_rounds=50, deadline=deadline)
    
    return best
