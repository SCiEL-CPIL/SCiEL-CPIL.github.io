import time
import math
import random

def score_set(U):
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    if len(U) < 2:
        return 0.0
    
    max_u = max(U)
    # Score formula: 1 + log(|U-U|/|U+U|) / log(2*max_u + 1)
    # Note: log base doesn't matter for the ratio part, but the +1 is absolute.
    # The prompt says: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    # Usually log in these contexts is natural log or log2, but the ratio of logs is base invariant.
    # So we can use math.log.
    
    sum_set = set()
    diff_set = set()
    
    # Optimization: Use list for U
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 1.0:
        # Score will be <= 1
        # But we need to return the value
        pass
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    score = 1.0 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial seed
    current_set = {0, 1, 3}
    best_set = current_set
    best_score = score_set(best_set)
    
    # Local Search
    # We will try to improve by adding, removing, or swapping elements.
    # To avoid infinite loops or getting stuck, we use a simple hill climbing.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Generate candidates
        # 1. Add a new element
        # 2. Remove an element
        # 3. Swap an element
        
        # Determine search range for new elements
        min_u = min(current_set)
        max_u = max(current_set)
        
        # Heuristic: new elements should be close to existing ones
        # Range: [min_u - 5, max_u + 5]
        search_range = list(range(max(0, min_u - 10), max_u + 11))
        
        # Try all additions
        for x in search_range:
            if x in current_set:
                continue
            new_set = current_set.copy()
            new_set.add(x)
            s = score_set(new_set)
            if s > best_score:
                best_score = s
                best_set = new_set
                current_set = new_set
                improved = True
                break # Restart from new best
        
        if improved:
            continue
            
        # Try all removals (don't remove 0)
        for x in list(current_set):
            if x == 0:
                continue
            if len(current_set) <= 2:
                continue
            new_set = current_set.copy()
            new_set.remove(x)
            s = score_set(new_set)
            if s > best_score:
                best_score = s
                best_set = new_set
                current_set = new_set
                improved = True
                break
        
        if improved:
            continue
            
        # Try swaps
        # Swap x in current_set with y in search_range \ current_set
        for x in list(current_set):
            if x == 0:
                continue
            for y in search_range:
                if y in current_set:
                    continue
                new_set = current_set.copy()
                new_set.remove(x)
                new_set.add(y)
                s = score_set(new_set)
                if s > best_score:
                    best_score = s
                    best_set = new_set
                    current_set = new_set
                    improved = True
                    break
            if improved:
                break
                
    return sorted(best_set)

# EVOLVE-BLOCK-START
def construct():
    # Re-use the logic above but wrapped in the required function
    # The prompt asks to keep function names and signatures.
    # The current program has:
    # def construct():
    #     return [0, 1, 3]
    
    # I will implement the search inside construct.
    
    import time
    import math
    
    def _score_set(U):
        if not U:
            return 0.0
        U = list(set(U))
        if 0 not in U:
            U.append(0)
        if len(U) < 2:
            return 0.0
        
        max_u = max(U)
        sum_set = set()
        diff_set = set()
        
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
                
        size_sum = len(sum_set)
        size_diff = len(diff_set)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    start_time = time.time()
    deadline = start_time + 110.0
    
    current_set = {0, 1, 3}
    best_set = current_set
    best_score = _score_set(best_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        min_u = min(current_set)
        max_u = max(current_set)
        
        # Expand search range slightly as set grows
        margin = 10 + len(current_set)
        search_range = list(range(max(0, min_u - margin), max_u + margin + 1))
        
        # 1. Add
        for x in search_range:
            if x in current_set:
                continue
            new_set = current_set.copy()
            new_set.add(x)
            s = _score_set(new_set)
            if s > best_score:
                best_score = s
                best_set = new_set
                current_set = new_set
                improved = True
                break
        
        if improved:
            continue
        
        # 2. Remove
        for x in list(current_set):
            if x == 0:
                continue
            if len(current_set) <= 2:
                continue
            new_set = current_set.copy()
            new_set.remove(x)
            s = _score_set(new_set)
            if s > best_score:
                best_score = s
                best_set = new_set
                current_set = new_set
                improved = True
                break
        
        if improved:
            continue
        
        # 3. Swap
        for x in list(current_set):
            if x == 0:
                continue
            for y in search_range:
                if y in current_set:
                    continue
                new_set = current_set.copy()
                new_set.remove(x)
                new_set.add(y)
                s = _score_set(new_set)
                if s > best_score:
                    best_score = s
                    best_set = new_set
                    current_set = new_set
                    improved = True
                    break
            if improved:
                break
                
    return sorted(best_set)
# EVOLVE-BLOCK-END
