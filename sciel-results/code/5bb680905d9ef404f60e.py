"""Combined construction: structural seed + greedy addition + local search refinement."""
import time

def construct():
    start = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill

    # Helper: compute reachable sums up to target for a given set S
    def compute_reachable(S, target=200):
        reachable = {0}
        for x in sorted(S):
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable |= new_sums
        return reachable

    # Helper: check if a set is valid (no subset sums to 200)
    def is_valid(S, target=200):
        reachable = compute_reachable(S, target)
        return target not in reachable

    # Helper: find a violating subset (one that sums to target)
    def find_violation(S, target=200):
        # DP with witness tracking
        # dp[s] = (prev_s, item) that first reached s
        witness = {0: None}
        curr_reachable = {0}
        for x in sorted(S):
            new_sums = set()
            for s in curr_reachable:
                ns = s + x
                if ns <= target:
                    if ns not in witness:
                        witness[ns] = (s, x)
                    new_sums.add(ns)
            curr_reachable |= new_sums
            if target in curr_reachable:
                # Backtrack
                subset = []
                curr_s = target
                while witness[curr_s] is not None:
                    prev_s, item = witness[curr_s]
                    subset.append(item)
                    curr_s = prev_s
                return subset
        return None

    # Step 1: Start with structural seed {67, ..., 100}
    S = set(range(67, 101))  # 34 elements, provably valid
    reachable = compute_reachable(S)

    # Step 2: Greedy addition from 66 down to 1
    for x in range(66, 0, -1):
        if time.time() - start > DEADLINE:
            break
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 not in new_sums:
            S.add(x)
            reachable |= new_sums

    # Step 3: Local search refinement
    # Try to swap elements: remove one, add another to increase cardinality or maintain
    # While we have time, try to improve via local moves
    
    best_set = set(S)
    best_size = len(S)
    
    # Try adding remaining elements in different orders
    remaining = [x for x in range(1, 101) if x not in S]
    
    # Try greedy addition with different orderings
    for ordering in [remaining, sorted(remaining, reverse=True), sorted(remaining)]:
        if time.time() - start > DEADLINE:
            break
        S_try = set(S)
        reachable_try = compute_reachable(S_try)
        for x in ordering:
            if x in S_try:
                continue
            new_sums = {s + x for s in reachable_try if s + x <= 200}
            if 200 not in new_sums:
                S_try.add(x)
                reachable_try |= new_sums
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try
    
    # Step 4: Simplified local search - try adding remaining elements in shuffled orders
    # This is O(N*T) per pass, much faster than O(N^2*T) swap search
    import random
    random.seed(42) # For reproducibility if needed, though not strictly required
    S_work = set(best_set)
    # Try a few shuffled greedy passes
    for _ in range(10): # Limit number of passes to ensure speed
        if time.time() - start > DEADLINE:
            break
        S_try = set(S_work)
        reachable_try = compute_reachable(S_try)
        remaining = [x for x in range(1, 101) if x not in S_try]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() - start > DEADLINE:
                break
            new_sums = {s + x for s in reachable_try if s + x <= 200}
            if 200 not in new_sums:
                S_try.add(x)
                reachable_try |= new_sums
        if len(S_try) > len(S_work):
            S_work = S_try
    
    # Step 5: Final greedy addition pass on the best set found
    S_final = set(S_work)
    
    reachable_final = compute_reachable(S_final)
    for x in range(100, 0, -1):
        if time.time() - start > DEADLINE:
            break
        if x in S_final:
            continue
        new_sums = {s + x for s in reachable_final if s + x <= 200}
        if 200 not in new_sums:
            S_final.add(x)
            reachable_final |= new_sums
    
    result = list(S_final)
    if not result:
        result = [100]
    
    # Final validation
    if not is_valid(result):
        # Fallback to known valid set
        result = list(range(67, 101))
    
    return result
