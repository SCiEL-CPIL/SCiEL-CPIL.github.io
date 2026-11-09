import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
        
    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    def covers_all(S):
        diffs = get_diffs(S)
        return target_diffs.issubset(diffs)

    best_max = float('inf')
    best_set = None
    
    # CSP Backtracking with Forward Checking
    # State: (S, used_diffs, covered_mask)
    # S is a tuple of sorted elements
    # We build the set incrementally, always adding the smallest possible valid element
    # that covers the most missing differences, with backtracking.
    
    # For efficiency, we use a recursive backtracking approach.
    # To avoid deep recursion, we limit the depth and use iterative deepening or a fixed depth.
    # Given we need at most 32 elements, and we start with 0, we have up to 31 more elements.
    
    # A better approach: 
    # 1. Start with S = [0].
    # 2. At each step, generate candidates for the next element.
    # 3. A candidate x is valid if:
    #    a) x > max(S)
    #    b) All differences (x - s) for s in S are not in used_diffs.
    #    c) x <= current_best_max (if we have one)
    # 4. We want to minimize the final max.
    # 5. We use a priority queue (min-heap) where the priority is the current max element.
    #    This is essentially A* search or Dijkstra's on the state space, where the cost is the max element.
    #    However, the state space is huge.
    
    # Let's stick to a modified Beam Search but with a different candidate generation strategy.
    # Instead of generating candidates for ALL missing diffs, we focus on the "most critical" missing diff.
    # The most critical diff is the one that is hardest to cover, e.g., the largest missing diff,
    # or the one that requires the largest new element.
    
    # Actually, the directive suggests a CSP approach. Let's try a simple backtracking with pruning.
    # We will try to construct the set in increasing order.
    # We will use a time limit to return the best found.
    
    # Let's try a different structural change: 
    # Use a simulated annealing approach on the set of elements.
    # Start with a random valid Sidon set that covers all diffs (if possible), 
    # then try to reduce the max element by swapping elements.
    
    # But finding an initial valid set is hard.
    
    # Let's try a hybrid: 
    # 1. Use a greedy construction to get a valid set.
    # 2. Use local search to minimize the max element.
    
    # Greedy Construction:
    # Start with S = [0].
    # While not all diffs covered:
    #   Find the smallest x > max(S) such that S + [x] is Sidon and covers at least one new diff.
    #   Add x to S.
    
    def greedy_construct():
        S = [0]
        used_diffs = set()
        covered_mask = 0
        
        for _ in range(31):  # Max 32 elements
            if covered_mask == full_mask:
                break
                
            missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
            if not missing_diffs:
                break
                
            best_x = None
            best_mask = 0
            best_diffs = None
            
            # Try candidates in increasing order
            # Start from max(S) + 1
            # Limit search range to avoid infinite loop
            max_search = max(S) + 1000 if S else 1000
            
            for x in range(max(S) + 1, max_search + 1):
                new_diffs = []
                valid = True
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                # Prefer x that covers more missing diffs
                new_missing = 30 - bin(new_mask).count('1')
                if best_x is None or new_missing < (30 - bin(best_mask).count('1')):
                    best_x = x
                    best_mask = new_mask
                    best_diffs = new_diffs
                    
                    # If we cover all missing diffs, we can break early
                    if new_mask == full_mask:
                        break
            
            if best_x is not None:
                S.append(best_x)
                used_diffs.update(best_diffs)
                covered_mask = best_mask
            else:
                # Fallback: just add a large number
                x = max(S) + 100
                while True:
                    new_diffs = []
                    valid = True
                    for s in S:
                        d = x - s
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        break
                    x += 1
                S.append(x)
                used_diffs.update(new_diffs)
                covered_mask = 0
                # Recalculate mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        covered_mask |= (1 << (d - 1))
                
        return S, covered_mask
    
    # Local Search:
    # Try to reduce the max element by swapping it with a smaller one.
    def local_search(S, time_limit):
        S = sorted(S)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Try to reduce each element, starting from the largest
            for i in range(len(S) - 1, 0, -1):
                if (time.time() - start_time) > time_limit:
                    break
                
                s_i = S[i]
                # Try to replace s_i with a smaller value
                # We only try values in the range (S[i-1], s_i)
                # But this can be expensive.
                # Instead, try a few random smaller values?
                # Or try s_i - 1, s_i - 2, etc.?
                
                # Let's try a limited range: s_i - 100 to s_i - 1
                # But we must ensure the new value is > S[i-1]
                low = S[i-1] + 1
                high = s_i - 1
                
                if low > high:
                    continue
                
                # Try a few candidates in this range
                # We want to find a candidate that keeps the set Sidon and covers all diffs
                candidates = []
                # Sample a few values
                if high - low <= 50:
                    candidates = list(range(low, high + 1))
                else:
                    # Random sampling
                    candidates = [random.randint(low, high) for _ in range(50)]
                
                for new_val in candidates:
                    S_new = S[:i] + S[i+1:] + [new_val]
                    S_new.sort()
                    if is_sidon(S_new) and covers_all(S_new):
                        S = S_new
                        improved = True
                        break
                if improved:
                    break
        
        return S
    
    # 1. Greedy Construction
    S_greedy, mask_greedy = greedy_construct()
    
    if mask_greedy == full_mask:
        best_set = S_greedy
        best_max = max(S_greedy)
    else:
        # Fallback if greedy fails to cover all diffs
        # Use the original beam search as a fallback
        best_set = None
    
    # 2. Local Search
    if best_set:
        best_set = local_search(best_set, deadline)
    
    if not best_set:
        # Fallback to a simple valid set
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32:
            missing = target_diffs - covered
            found = False
            for d in missing:
                for s in S:
                    c = s + d
                    if c in S or c < 0:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                c = 2 * max(S) + 3
                while True:
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        break
                    c += 1
        best_set = S
    
    return best_set
