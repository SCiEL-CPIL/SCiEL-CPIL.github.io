import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Greedy initialization
    S = set()
    counts = {}
    
    # Helper to check if x can be added to current S with current counts
    def can_add(x):
        # Check 2*x
        if counts.get(2*x, 0) >= 2:
            return False
        # Check x + s for s in S
        for s in S:
            if counts.get(x + s, 0) >= 2:
                return False
        return True

    # Helper to add x
    def add(x):
        S.add(x)
        counts[2*x] = counts.get(2*x, 0) + 1
        for s in S:
            if s != x:
                counts[x + s] = counts.get(x + s, 0) + 1

    # Helper to remove x
    def remove(x):
        S.remove(x)
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for s in S:
            counts[x + s] -= 1
            if counts[x + s] == 0:
                del counts[x + s]

    # Initial greedy: pick smallest available
    for x in range(1, 1001):
        if can_add(x):
            add(x)
        if time.time() > deadline:
            break

    # Local search
    # We want to maximize |S|. 
    # Strategy: Try to add. If fail, try swap.
    
    candidates = [x for x in range(1, 1001) if x not in S]
    
    while time.time() < deadline:
        # Try to add a random candidate
        if candidates:
            idx = random.randint(0, len(candidates) - 1)
            y = candidates[idx]
            if can_add(y):
                add(y)
                candidates.remove(y)
                continue
        
        # Try a swap
        if S:
            x = random.choice(list(S))
            if candidates:
                y = random.choice(candidates)
                # Simulate swap: remove x, check if y can be added
                # To be safe, we can just check if S\{x} U {y} is valid
                # This requires checking all sums in S\{x} with y, and sums in S\{x} among themselves (unchanged)
                # Actually, sums in S\{x} are unchanged. Only sums involving x or y change.
                # Sums involving x are removed. Sums involving y are added.
                # We need to ensure that for all s in S\{x}, count(x+y) < 2? No.
                # We need count(s+y) < 2 for all s in S\{x}.
                # And count(2y) < 2.
                
                # Let's check validity of adding y to S\{x}
                # We can temporarily remove x, check can_add(y), then revert if needed.
                
                # To avoid modifying state, we can just check:
                # For all s in S:
                #   if s == x: skip
                #   else: if counts.get(s+y, 0) >= 2: invalid (but counts includes x+s, so we need to be careful)
                
                # Actually, counts currently includes x.
                # If we remove x, counts[s+y] for s!=x is unchanged.
                # So we just need to check if counts.get(s+y, 0) < 2 for all s in S, s!=x.
                # And counts.get(2y, 0) < 2.
                
                valid_swap = True
                if counts.get(2*y, 0) >= 2:
                    valid_swap = False
                else:
                    for s in S:
                        if s == x:
                            continue
                        if counts.get(s + y, 0) >= 2:
                            valid_swap = False
                            break
                
                if valid_swap:
                    # Perform swap
                    remove(x)
                    add(y)
                    candidates.remove(y)
                    # Note: x is now a candidate again? 
                    # We removed x from S, so x is in complement.
                    # We should add x to candidates.
                    candidates.append(x)
                    
    return sorted(list(S))
