import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    max_val = 284
    
    def get_initial_greedy():
        S = list(prescribed)
        S.sort()
        Sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                Sums.add(S[i] + S[j])
        
        # Greedy fill
        for x in range(12, max_val + 1):
            valid = True
            # Check 2*x
            if 2 * x in Sums:
                valid = False
                break
            # Check x+y for y in S
            for y in S:
                if x + y in Sums:
                    valid = False
                    break
            if valid:
                S.append(x)
                Sums.add(2 * x)
                for y in S[:-1]: # Note: S changed, careful. 
                    # Actually, we should add to Sums after appending or just compute new sums
                    pass
                
                # Let's do this properly
                S.pop() # Undo append for logic clarity, or just append and update sums
                S.append(x)
                Sums.add(2 * x)
                for y in S:
                    if y != x:
                        Sums.add(x + y)
                        
        return S, Sums

    def can_add(x, S, Sums):
        if 2 * x in Sums:
            return False
        for y in S:
            if x + y in Sums:
                return False
        return True

    def add_element(x, S, Sums):
        Sums.add(2 * x)
        for y in S:
            Sums.add(x + y)
        S.append(x)
        S.sort()

    def remove_element(x, S, Sums):
        # Remove x from S
        S.remove(x)
        # Remove sums involving x
        Sums.discard(2 * x)
        for y in S:
            Sums.discard(x + y)

    def get_conflicts(x, S, Sums):
        # Returns list of y in S such that x+y in Sums, or 2x in Sums
        # This is useful for swap moves
        pass

    best_size = 0
    best_S = []
    
    # Run multiple restarts
    num_restarts = 10000
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        S, Sums = get_initial_greedy()
        
        # Local Search
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
            
        stuck_counter = 0
        max_stuck = 50
        
        while time.time() < deadline:
            # Try to add a random element
            candidates = [x for x in range(1, max_val + 1) if x not in S]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if can_add(x, S, Sums):
                    add_element(x, S, Sums)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                    added = True
                    break
            
            if added:
                stuck_counter = 0
                continue
            
            # If no add possible, try to swap
            # Pick a random non-prescribed element to remove
            removable = [x for x in S if x not in prescribed]
            if not removable:
                break # Can't do anything
            
            random.shuffle(removable)
            swap_done = False
            for r in removable:
                # Temporarily remove r
                remove_element(r, S, Sums)
                
                # Try to add a random element
                candidates = [x for x in range(1, max_val + 1) if x not in S]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add(x, S, Sums):
                        add_element(x, S, Sums)
                        # Size remains same, but structure changed
                        # Check if we can add more now? The loop will handle it.
                        swap_done = True
                        break
                
                if swap_done:
                    stuck_counter = 0
                    break
                else:
                    # Restore r
                    add_element(r, S, Sums)
            
            if not swap_done:
                stuck_counter += 1
                if stuck_counter > max_stuck:
                    break # Restart
            
        # End of local search for this restart
        
        if best_size >= 20:
            break # We found the theoretical max, stop early

    return best_S
