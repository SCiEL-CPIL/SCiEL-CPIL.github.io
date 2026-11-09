import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Data structures
    S = []  # List of chosen integers
    in_S = [False] * (max_val + 1)
    counts = [0] * (2 * max_val + 1)  # counts[s] for s in 0..2000
    
    # Helper to add a number to the set and update counts
    # Returns True if successful, False if it would violate B2[2]
    def try_add(x):
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        # Perform addition
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    # Helper to remove a number
    def try_remove(x):
        if x not in S:
            return
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    # Find all pairs (a, b) in S such that a + b = s and a <= b
    # Returns a list of pairs
    def find_pairs_for_sum(s):
        pairs = []
        for i in range(len(S)):
            a = S[i]
            b = s - a
            if b < a:
                continue
            if b > max_val:
                continue
            if in_S[b]:
                pairs.append((a, b))
        return pairs

    # Get elements in S that block adding c
    def get_blockers(c):
        blockers = set()
        
        # Check 2*c
        if counts[2 * c] == 2:
            pairs = find_pairs_for_sum(2 * c)
            for a, b in pairs:
                blockers.add(a)
                blockers.add(b)
                
        # Check c + y
        for y in S:
            s = c + y
            if counts[s] == 2:
                pairs = find_pairs_for_sum(s)
                for a, b in pairs:
                    blockers.add(a)
                    blockers.add(b)
                    
        return blockers

    # Check if swapping a with c is valid
    # We assume a is in S and c is not in S.
    def try_swap(a, c):
        # Temporarily remove a
        try_remove(a)
        
        # Try to add c
        if try_add(c):
            return True
        else:
            # Restore a
            try_add(a)
            return False

    # 1. Greedy Initialization (Randomized)
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    for i in candidates:
        if try_add(i):
            pass
            
    best_size = len(S)
    
    # 2. Local Search
    # We iterate multiple times or until time limit
    # To avoid getting stuck in a local optimum, we can try random restarts 
    # but that loses progress. Instead, we just keep searching.
    
    # To speed up, we can pre-calculate a list of unused numbers?
    # No, just pick random c.
    
    while time.time() < deadline:
        c = random.randint(1, max_val)
        if in_S[c]:
            continue
            
        # Try direct add
        if try_add(c):
            continue
            
        # If direct add failed, find blockers
        blockers = get_blockers(c)
        
        # If no blockers found (shouldn't happen if counts logic is correct 
        # and counts==2 implies blockers exist), skip.
        if not blockers:
            continue
            
        # Try to swap each blocker with c
        swapped = False
        # Shuffle blockers to randomize choice
        blocker_list = list(blockers)
        random.shuffle(blocker_list)
        
        for b in blocker_list:
            if b not in S: # Safety check
                continue
            if try_swap(b, c):
                swapped = True
                break
                
        if not swapped:
            # Optional: If we have lots of time, we could try 2-blocker swaps?
            # Or just skip.
            pass

    return S
