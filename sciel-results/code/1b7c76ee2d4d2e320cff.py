import time
import random
import heapq
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    N = 1000
    MAX_SUM = 2 * N
    
    # Data structures
    # sum_counts[s] = number of pairs (a,b) in S such that a+b=s and a<=b
    sum_counts = [0] * (MAX_SUM + 1)
    # in_set[x] = 1 if x is in S, 0 otherwise
    in_set = [0] * (N + 1)
    # active_elements: list of elements currently in S
    active_elements = []
    
    # Helper to check if adding x is valid (doesn't make any sum exceed 2)
    # Returns True if valid, and updates nothing.
    def is_valid_add(x):
        if in_set[x]:
            return False
        # Check 2x
        if sum_counts[2 * x] >= 2:
            return False
        # Check x + y for all y in S
        for y in active_elements:
            if sum_counts[x + y] >= 2:
                return False
        return True

    # Helper to add x to the set
    def add_to_set(x):
        in_set[x] = 1
        active_elements.append(x)
        sum_counts[2 * x] += 1
        for y in active_elements:
            if y == x:
                continue
            sum_counts[x + y] += 1

    # Helper to remove x from the set
    def remove_from_set(x):
        in_set[x] = 0
        # Remove from active_elements (O(n) but n is small)
        if x in active_elements:
            active_elements.remove(x)
        sum_counts[2 * x] -= 1
        for y in active_elements:
            if y == x:
                continue
            sum_counts[x + y] -= 1

    # Phase 1: Greedy Construction with "Pressure" Heuristic
    # Instead of just picking the smallest valid number, we pick the number
    # that leaves the most "room" for future additions.
    # A simple proxy: pick the valid number that minimizes the number of sums
    # that become "full" (count == 2) or are close to full.
    # However, calculating this for all candidates is expensive.
    # Alternative: Randomized Greedy with Restart.
    # We will run multiple random greedy passes and keep the best.
    
    best_size = 0
    best_state = None
    best_sum_counts = None
    best_in_set = None
    best_active = None
    
    num_restarts = 50
    time_per_restart = 2.0 # seconds
    
    for r in range(num_restarts):
        if time.time() > deadline - 20: # Leave 20s for final polish
            break
            
        # Reset state
        sum_counts[:] = [0] * (MAX_SUM + 1)
        in_set[:] = [0] * (N + 1)
        active_elements = []
        
        # Randomized Greedy
        # We iterate through a shuffled list of candidates? 
        # Or we pick from a pool of valid candidates randomly?
        # Let's try: At each step, collect all valid candidates. 
        # If none, stop. If some, pick one randomly (or with bias).
        # To avoid O(N^2) collection, we can just iterate 1..N and pick the first valid?
        # That's the standard greedy. 
        # Let's try: Shuffled order 1..N. Add if valid.
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 20:
                break
            if is_valid_add(x):
                add_to_set(x)
                
        current_size = len(active_elements)
        if current_size > best_size:
            best_size = current_size
            best_state = active_elements[:]
            best_sum_counts = sum_counts[:]
            best_in_set = in_set[:]
            best_active = active_elements[:]
            
    # Phase 2: Local Search (Swap Moves)
    # Start from the best greedy result found.
    if best_state is None:
        # Fallback: simple greedy
        sum_counts[:] = [0] * (MAX_SUM + 1)
        in_set[:] = [0] * (N + 1)
        active_elements = []
        for x in range(1, N + 1):
            if is_valid_add(x):
                add_to_set(x)
        best_state = active_elements[:]
        best_sum_counts = sum_counts[:]
        best_in_set = in_set[:]
        best_active = active_elements[:]
        best_size = len(active_elements)
        
    # Load best state into working variables
    sum_counts = best_sum_counts[:]
    in_set = best_in_set[:]
    active_elements = best_active[:]
    
    # Local Search: Try to swap one element out for one or more in?
    # Since we are maximizing size, we want to find a swap (remove u, add v) 
    # that maintains validity and potentially allows adding more later?
    # Actually, a direct swap (remove u, add v) keeps size same. 
    # We need a "Kick": Remove u, add v, w. Size +1.
    # Or: Remove u, add v (size same), then try to add w (size +1).
    
    # Let's implement a Hill Climbing loop:
    # 1. Try to add any valid number (Greedy Fill). If added, loop.
    # 2. If no add possible, try a "Kick": 
    #    - Pick a random element u in S to remove.
    #    - Try to add a random element v not in S.
    #    - If valid, accept the swap (size unchanged).
    #    - Then try to add another element w. If valid, accept (size +1).
    #    - If no w found, revert to original? Or keep the swap if it leads to a better "state"?
    #    Since size is the only metric, keeping a swap that doesn't increase size immediately 
    #    is risky unless it enables future growth. 
    #    Let's try: Remove u. Try to add v. If valid, we are in a new state. 
    #    Try to add w. If valid, size+1. Keep.
    #    If w not valid, try to add v2...
    #    If no improvement after k tries, revert.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Greedy Fill
        added_any = False
        for x in range(1, N + 1):
            if in_set[x]:
                continue
            if is_valid_add(x):
                add_to_set(x)
                added_any = True
                improved = True
                # Restart fill loop immediately? 
                # Since adding x might invalidate others, we just continue.
                # But we might have missed some? No, we iterate 1..N.
                # If we add x, it only increases counts, so previously valid 
                # numbers might become invalid. Previously invalid remain invalid.
                # So we don't need to restart, just continue to N.
        
        if not added_any:
            # 2. Try a Kick (Swap to enable growth)
            # We will try a few random kicks
            kick_success = False
            for _ in range(100): # Limit kicks per stagnation phase
                if time.time() > deadline - 5:
                    break
                    
                if not active_elements:
                    break
                    
                u = random.choice(active_elements)
                
                # Save state to revert if kick fails
                # Since state is small, we can just track changes
                # Remove u
                remove_from_set(u)
                
                # Try to add a new element v
                # We want v such that adding v is valid.
                # Then we try to add w.
                # To be efficient, we pick v randomly from non-in_set.
                # But random pick might be slow to find a valid v that allows w.
                
                # Strategy: Pick v randomly. If is_valid_add(v), add v.
                # Then try to add any w. If found, success.
                # If not found, remove v, restore u.
                
                v = random.randint(1, N)
                if in_set[v]:
                    continue
                    
                if is_valid_add(v):
                    add_to_set(v)
                    
                    # Now try to add w
                    w_found = False
                    for w in range(1, N + 1):
                        if in_set[w]:
                            continue
                        if is_valid_add(w):
                            add_to_set(w)
                            w_found = True
                            break
                            
                    if w_found:
                        # We have increased size by 1 (removed u, added v, added w)
                        # Wait, we removed u (size-1), added v (size), added w (size+1).
                        # Net change: +1.
                        # This is a valid improvement!
                        improved = True
                        kick_success = True
                        break
                    else:
                        # Revert v
                        remove_from_set(v)
                        # Restore u
                        add_to_set(u)
                else:
                    # v was not valid, so we didn't add it. 
                    # But we removed u! We must restore u.
                    add_to_set(u)
                    
            # If kick_success, loop continues (improved=True)
            # If not, we are stuck. 
            # To avoid infinite loop, we might need a more aggressive restart or break.
            # For now, if no kick works in 100 tries, we break out of local search.
            if not kick_success:
                # Try a few more times with different randomness? 
                # Or break.
                # Let's break to avoid wasting time.
                break
                
    # Final Greedy Fill to be safe
    for x in range(1, N + 1):
        if in_set[x]:
            continue
        if is_valid_add(x):
            add_to_set(x)
            
    return active_elements

# Test locally? No, just return the function.
if __name__ == "__main__":
    # Quick sanity check
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    s = set(res)
    counts = {}
    valid = True
    for i, a in enumerate(res):
        for b in res[i:]:
            sum_val = a + b
            counts[sum_val] = counts.get(sum_val, 0) + 1
    for k, v in counts.items():
        if v > 2:
            valid = False
            break
    print(f"Valid: {valid}")
    if not valid:
        print("ERROR: Invalid set generated")
