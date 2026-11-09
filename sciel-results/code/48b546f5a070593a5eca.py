import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin
    
    prescribed = [1, 2, 5, 11]
    N = 284
    target_size = 20
    
    def get_diffs(S):
        diffs = set()
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                diffs.add(S_sorted[j] - S_sorted[i])
        return diffs
    
    def is_valid_add(S, D, x):
        # Check if x can be added to S given current diffs D
        # New diffs will be x - s for s in S
        # These must not be in D, and must be distinct from each other
        new_diffs = set()
        for s in S:
            d = x - s
            if d in D or d in new_diffs:
                return False
            new_diffs.add(d)
        return True
        
    def greedy_with_lookahead(prescribed, N, target_size, deadline, use_randomization=False):
        S = list(prescribed)
        D = get_diffs(S)
        
        # Pre-calculate valid candidates? No, dynamic.
        
        while len(S) < target_size:
            if time.time() > deadline:
                break
                
            candidates = []
            # Candidates must be > max(S) ? No, just not in S.
            # But to keep it simple and ordered, we can just iterate 1..N
            # Actually, adding small numbers first is usually good for packing.
            # But we must ensure distinctness.
            
            # Find all valid x
            valid_xs = []
            for x in range(1, N + 1):
                if x in S:
                    continue
                if is_valid_add(S, D, x):
                    valid_xs.append(x)
                    
            if not valid_xs:
                break
                
            if use_randomization and random.random() < 0.1 and len(valid_xs) > 1:
                # Pick a random candidate
                x = random.choice(valid_xs)
            else:
                # Pick the one with minimum cost (blocks fewest future marks)
                best_x = None
                min_cost = float('inf')
                
                for x in valid_xs:
                    # Calculate cost: how many y > x become invalid?
                    # Or just how many total y become invalid?
                    # Let's count total invalid y in [1, N] \ (S U {x})
                    cost = 0
                    S_temp = S + [x]
                    D_temp = D.union(x - s for s in S)
                    
                    # To speed up, we can just check a subset or estimate.
                    # Full check is O(N * |S|)
                    # N=284, |S|<=20. 284*20 = 5680 ops per candidate.
                    # Candidates ~ 284. Total ~ 1.6M ops per step.
                    # Steps ~ 16. Total ~ 25M ops.
                    # Python might take ~1-2 seconds. Acceptable.
                    
                    for y in range(1, N + 1):
                        if y in S_temp:
                            continue
                        # Check if y is valid for S_temp
                        # We can reuse is_valid_add logic but it's slower due to set creation
                        # Let's inline a fast check
                        valid = True
                        new_diffs = set()
                        for s in S_temp:
                            d = y - s
                            if d in D_temp or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if not valid:
                            cost += 1
                            
                    if cost < min_cost:
                        min_cost = cost
                        best_x = x
                        
                x = best_x
                
            S.append(x)
            D.update(x - s for s in S[:-1]) # S[:-1] is S without x
            
        return S

    best_set = None
    best_size = 0
    
    # Try deterministic lookahead first
    S = greedy_with_lookahead(prescribed, N, target_size, DEADLINE, use_randomization=False)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    if best_size == target_size:
        return best_set

    # If not found, try randomized restarts
    while time.time() < DEADLINE:
        S = greedy_with_lookahead(prescribed, N, target_size, DEADLINE, use_randomization=True)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            if best_size == target_size:
                break
                
    return best_set
