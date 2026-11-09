import random
import time

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def can_add(x, S):
        # Check if adding x to S creates a 4-AP
        # Patterns involving x:
        # 1. x, x+d, x+2d, x+3d (x is 1st)
        # 2. x-d, x, x+d, x+2d (x is 2nd)
        # 3. x-2d, x-d, x, x+d (x is 3rd)
        # 4. x-3d, x-2d, x-d, x (x is 4th)
        
        # Case 1: x is first term
        # x+3d <= N => d <= (N-x)//3
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        
        # Case 2: x is second term
        # x-d >= 1 and x+2d <= N => d <= (N-x)//2 and d <= x-1
        max_d = min((N - x) // 2, x - 1)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        
        # Case 3: x is third term
        # x-2d >= 1 and x+d <= N => d <= (x-1)//2 and d <= N-x
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        
        # Case 4: x is fourth term
        # x-3d >= 1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
                
        return True

    def greedy_random():
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        return S

    best_set = set()
    best_size = 0
    
    # Phase 1: Randomized Greedy
    # Each trial is fast. We run as many as possible until we reserve time for local search.
    # Let's reserve 20s for local search.
    greedy_deadline = start_time + 65.0
    
    while time.time() < greedy_deadline:
        S = greedy_random()
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Phase 2: Local Search
    # We try to improve best_set by removing one element and adding others.
    S = best_set
    current_size = len(S)
    
    # We will do a series of improvement attempts.
    # Each attempt: pick a random u in S, remove it, try to add all v not in S.
    # If size increases, accept.
    # If we get stuck (no improvement in many tries), we might try a different strategy or stop.
    
    no_improve_count = 0
    max_no_improve = 1000 # Reset or stop if stuck too long? 
    # Actually, if we can't improve by removing 1, we are likely at a local maximum for this metric.
    # We can also try "Swap" (remove 1, add 1) to move to a different local max, but that's complex.
    # Let's stick to Remove-Add-Multiple.
    
    while time.time() < deadline:
        if not S:
            break
            
        # Pick a random element to remove
        u = random.choice(list(S))
        S_temp = S.copy()
        S_temp.remove(u)
        
        # Try to add any element not in S_temp
        # To make this faster, we can just iterate through all 1..N
        # But we only care if we can add *more* than we removed (1).
        # So we need to add at least 1 element to keep size, or more to improve.
        # If we add 0, size decreases. If we add 1, size same. If >1, size increases.
        # We only accept if size increases.
        
        added_count = 0
        # We can optimize by only checking elements that were blocked by u?
        # Hard to know which were blocked. Just check all.
        # N=512 is small.
        
        # Optimization: If we already know we can't add enough, stop early?
        # Max possible additions is N - current_size.
        # We need added_count > 1 to improve.
        
        for v in range(1, N + 1):
            if v in S_temp:
                continue
            if can_add(v, S_temp):
                S_temp.add(v)
                added_count += 1
                # Early exit? If added_count is already > 1, we have improved.
                # But adding more might allow even more? No, we are just filling.
                # Once we have added > 1, the set is larger. We can stop checking and accept.
                # Wait, if we stop early, we might miss adding *more* elements that are now available?
                # No, `can_add` depends on the current set. As we add, the set grows, making it *harder* to add more.
                # So the greedy fill (checking in order) is optimal for maximizing size given the removal?
                # Not necessarily, order matters. But for a quick local search, it's fine.
                # However, if we accept as soon as added_count > 1, we might have a smaller set than if we had continued adding.
                # So we should complete the fill to get the maximum possible size for this removal.
                
        if len(S_temp) > current_size:
            S = S_temp
            current_size = len(S)
            best_set = S
            best_size = current_size
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If we've tried many times without improvement, we might be stuck.
            # We can break early if no_improve_count is very high, but let's let the deadline handle it.
            # To avoid infinite loops on bad random choices, we can just let it run.
            
    # Final Polish: Try to add any remaining element to best_set
    # This is redundant if Phase 2 did a full fill, but just in case.
    for v in range(1, N + 1):
        if v not in best_set and can_add(v, best_set):
            best_set.add(v)
            
    return sorted(best_set)
