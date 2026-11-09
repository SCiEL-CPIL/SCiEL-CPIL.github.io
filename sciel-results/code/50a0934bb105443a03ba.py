import numpy as np

def construct():
    n = 128
    targets = list(range(128, 257))  # 129 targets
    target_set = set(targets)
    
    A = set()
    # representation count for each target
    rep_count = {t: 0 for t in targets}
    
    # Precompute for efficiency
    all_candidates = list(range(0, 129))
    
    while True:
        # Check if all targets covered
        if all(rep_count[t] > 0 for t in targets):
            break
        
        best_x = None
        best_score = float('inf')
        
        for x in all_candidates:
            if x in A:
                continue
            
            # Simulate adding x
            new_rep = dict(rep_count)
            # x pairs with each element in A (including x itself)
            for a in A:
                s = x + a
                if s in new_rep:
                    new_rep[s] += 1
            # x + x
            s2 = x + x
            if s2 in new_rep:
                new_rep[s2] += 1
            
            # Check coverage and compute score
            max_count = 0
            covered = True
            for t in targets:
                if new_rep[t] == 0:
                    covered = False
                    break
                if new_rep[t] > max_count:
                    max_count = new_rep[t]
            
            if not covered:
                # Penalize uncovered targets heavily
                uncovered = sum(1 for t in targets if new_rep[t] == 0)
                score = max_count + uncovered * 1000
            else:
                score = max_count
            
            if score < best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            break
        
        A.add(best_x)
        # Update rep_count
        for a in A:
            s = best_x + a
            if s in rep_count:
                rep_count[s] += 1
        s2 = best_x + best_x
        if s2 in rep_count:
            rep_count[s2] += 1
    
    return list(A)
