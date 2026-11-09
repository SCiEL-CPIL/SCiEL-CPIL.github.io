import time
import sys
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    N = 17
    MIN_D = 6
    max_val = 1 << N
    
    # Precompute weights for all 2^N words
    weights = [0] * max_val
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
    
    def build_linear_code(seed_order=None):
        """Build a linear code using greedy basis selection."""
        basis = []
        current_code = [0]
        
        if seed_order is None:
            candidates = list(range(1, max_val))
        else:
            candidates = seed_order
        
        for g in candidates:
            if time.time() > deadline - 20:
                break
            if g == 0:
                continue
            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                old_code = current_code
                current_code = []
                for c in old_code:
                    current_code.append(c)
                    current_code.append(c ^ g)
                if len(basis) >= 8:
                    break
        
        return current_code if len(current_code) >= 2 else None
    
    def nonlinear_expand(base_code, time_budget):
        """Expand a base code using multi-restart greedy nonlinear addition."""
        base_set = set(base_code)
        
        # Compute scores for all candidates against the base code
        candidates_by_score = defaultdict(list)
        
        for x in range(max_val):
            if x in base_set:
                continue
            min_d = max_val
            valid_candidate = True
            for c in base_code:
                d = weights[x ^ c]
                if d < MIN_D:
                    valid_candidate = False
                    break
                if d < min_d:
                    min_d = d
            if valid_candidate:
                candidates_by_score[min_d].append(x)
        
        if not candidates_by_score:
            return base_code[:]
        
        sorted_scores = sorted(candidates_by_score.keys(), reverse=True)
        
        best_result = base_code[:]
        best_size = len(base_code)
        
        # Multi-restart greedy
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            added_in_phase2 = []
            existing_set = set(base_code)
            
            for s in sorted_scores:
                group = candidates_by_score[s][:]
                random.shuffle(group)
                
                for x in group:
                    if time.time() > deadline - 5:
                        break
                    # Check against added_in_phase2
                    valid = True
                    for c in added_in_phase2:
                        if weights[x ^ c] < MIN_D:
                            valid = False
                            break
                    if valid:
                        added_in_phase2.append(x)
                        existing_set.add(x)
            
            current_size = len(base_code) + len(added_in_phase2)
            if current_size > best_size:
                best_size = current_size
                best_result = base_code + added_in_phase2
        
        return best_result
    
    # Phase 1: Try multiple linear code constructions
    base_codes = []
    
    # First, deterministic scan
    base_code = build_linear_code()
    if base_code and len(base_code) >= 256:
        base_codes.append(base_code)
    
    # Try a few random orderings if time permits
    if time.time() < deadline - 30:
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            random_order = list(range(1, max_val))
            random.shuffle(random_order)
            bc = build_linear_code(random_order)
            if bc and len(bc) >= 256:
                base_codes.append(bc)
    
    if not base_codes:
        # Fallback: return what we have
        return [0]
    
    # Phase 2: Nonlinear expansion for each base code
    best_overall = []
    best_overall_size = 0
    
    for i, bc in enumerate(base_codes):
        if time.time() > deadline - 10:
            break
        # Allocate remaining time
        remaining = deadline - 8 - time.time()
        if remaining < 2:
            break
        result = nonlinear_expand(bc, remaining)
        if len(result) > best_overall_size:
            best_overall_size = len(result)
            best_overall = result
    
    if not best_overall:
        best_overall = base_codes[0]
    
    return best_overall
