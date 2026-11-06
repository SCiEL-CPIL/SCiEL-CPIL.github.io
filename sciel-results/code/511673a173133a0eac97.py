import time
import random

def _is_valid_set(S):
    """Check if a set S is valid."""
    S_sorted = sorted(S)
    n = len(S_sorted)
    for i in range(n):
        a = S_sorted[i]
        for j in range(i + 1, n):
            b = S_sorted[j]
            for k in range(j, n):
                c = S_sorted[k]
                if (b + c) % a == 0:
                    return False
    return True

def _can_add(S, x):
    """Check if x can be added to S."""
    if x in S:
        return False
    # S is a set, so we iterate over pairs in S
    # We need to check if x divides b+c for any b<=c in S
    # Note: The condition is for a<b<=c in the *final* set.
    # If we add x, x becomes the new 'a' for pairs (b,c) where b,c > x?
    # No, the condition is: For every a < b <= c in the set, a does NOT divide b+c.
    # If we add x, we must check:
    # 1. x as 'a': Does x divide b+c for any b,c in S such that x < b <= c?
    # 2. Existing elements as 'a': Do they still not divide sums involving x?
    #    If a is in S, and we add x, we must check pairs (a, x, y) or (x, a, y) etc?
    #    The condition is symmetric in b,c but not a.
    #    Let S be the current set. We want to add x.
    #    New set S' = S U {x}.
    #    We need to check all triples (a, b, c) in S' with a < b <= c.
    #    Cases:
    #    A) x is a: x < b <= c. b,c in S. Check x | (b+c).
    #    B) x is b: a < x <= c. a in S, c in S. Check a | (x+c).
    #    C) x is c: a < b <= x. a in S, b in S. Check a | (b+x).
    #    Note: If x is the largest, only case A applies (if x is smallest in the triple? No, a is smallest).
    #    If x is the largest element in S', then for any a,b in S with a<b<=x, we check a | (b+x).
    #    If x is the smallest, then for any b,c in S with x<b<=c, we check x | (b+c).
    #    If x is in the middle, we have both types.
    
    # Efficient check:
    # 1. Check x as 'a':
    for b in S:
        if b < x: continue
        for c in S:
            if c < b: continue
            if (b + c) % x == 0:
                return False
    # 2. Check existing 'a' in S with x involved in sum:
    #    Pairs (a, x, c) where a < x <= c. a in S, c in S.
    #    Pairs (a, b, x) where a < b <= x. a in S, b in S.
    #    Actually, we can just check all pairs (b,c) in S' involving x.
    #    The sums are x+b, x+c, b+c (if x is a).
    #    Let's just iterate through S elements as 'a' and check if any sum involving x is divisible.
    
    for a in S:
        # x must be > a for a to be the 'a' in the triple?
        # Condition: a < b <= c.
        # If a is in S, and we add x.
        # If x > a:
        #   - b could be x, c could be some c in S with x <= c. Sum: x+c. Check a | (x+c).
        #   - b could be some b in S with a < b <= x. c could be x. Sum: b+x. Check a | (b+x).
        #   - b,c in S, x not involved. Already valid.
        # If x < a:
        #   - a cannot be 'a' if x is smaller? No, a is still 'a' if a < b <= c.
        #   - If x < a, then x cannot be b or c for this 'a'.
        #   - So no new checks for this 'a'.
        
        if x > a:
            # Case: b=x, c in S, c >= x
            for c in S:
                if c >= x and (x + c) % a == 0:
                    return False
            # Case: b in S, a < b <= x, c=x
            for b in S:
                if a < b <= x and (b + x) % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    S = []
    # Greedy construction
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if _can_add(set(S), x):
            S.append(x)
    
    S_set = set(S)
    
    # Local Search
    # We want to maximize |S|.
    # Loop:
    # 1. Try to add any unused element.
    # 2. If none can be added, try a swap.
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Add
        for x in range(1, n + 1):
            if x in S_set:
                continue
            if _can_add(S_set, x):
                S_set.add(x)
                improved = True
                break # Restart the scan to catch new possibilities
        
        if improved:
            continue
        
        # Phase 2: Swap
        # If we can't add, try swapping an element v in S_set with u not in S_set
        # to potentially enable an addition later.
        # To avoid getting stuck in a loop of equal-size swaps, we can be conservative.
        # Only accept a swap if it "improves" the potential? Hard to define.
        # Let's just do random swaps if no addition is possible, for a limited number of steps?
        # Or better: Try all possible swaps. If any swap leads to a state where an addition is possible, take it.
        
        swap_made = False
        unused = [x for x in range(1, n + 1) if x not in S_set]
        
        # To keep it fast, limit the number of swap attempts or use a heuristic.
        # Let's try to find a swap that allows an immediate addition.
        # For each u in unused:
        #   For each v in S_set:
        #     S_temp = (S_set - {v}) U {u}
        #     If S_temp is valid:
        #        Check if any w in unused (w != u) can be added to S_temp.
        #        If yes, perform swap (S_set = S_temp) and break.
        
        # This is O(|unused| * |S_set| * (Validity + AddCheck)).
        # Validity is O(N^2). AddCheck is O(N^2).
        # 256 * 256 * 65k = 4 billion ops. Too slow.
        
        # Simplification: Just perform a random valid swap if no addition is found.
        # This explores the space of fixed-size sets.
        
        # Pick a random u from unused and random v from S_set.
        # If (S_set - {v} + {u}) is valid, do it.
        # This is cheap: O(N^2) for validity check.
        
        if unused:
            u = random.choice(unused)
            v = random.choice(list(S_set))
            S_temp = S_set.copy()
            S_temp.remove(v)
            S_temp.add(u)
            if _is_valid_set(S_temp):
                S_set = S_temp
                swap_made = True
        
        if not swap_made:
            # No addition, no random swap found (or we decided to stop swapping)
            # We might be in a local optimum.
            # To prevent infinite loop, we break.
            break
            
    return sorted(S_set)
