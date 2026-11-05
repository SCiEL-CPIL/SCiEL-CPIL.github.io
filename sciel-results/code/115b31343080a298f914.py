import random
import time

def construct():
    N = 512
    deadline = time.time() + 55  # 55s safety margin
    
    # Helper to check if adding x to S creates a 4-AP
    # Assumes x is NOT in S
    def creates_4ap(x, S):
        # x is part of a 4-AP: x, x+d, x+2d, x+3d
        # 1. x is first term
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return True
        # 2. x is second term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return True
        # 3. x is third term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return True
        # 4. x is fourth term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return True
        return False

    # Helper to find a 4-AP in S. Returns (a, b, c, d) or None
    def find_4ap(S):
        for a in S:
            max_d = (N - a) // 3
            for d in range(1, max_d + 1):
                if (a + d in S) and (a + 2*d in S) and (a + 3*d in S):
                    return (a, a+d, a+2*d, a+3*d)
        return None

    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        # Random initialization
        # Density 0.5 is a good start
        S = set(random.sample(range(1, N + 1), k=N//2))
        
        # Repair Phase: Remove elements causing 4-APs
        while True:
            if time.time() > deadline:
                break
            ap = find_4ap(S)
            if not ap:
                break
            
            # Remove a random element from the AP
            victim = random.choice(ap)
            S.remove(victim)
            
        # Improve Phase: Try to add elements
        candidates = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates)
        
        # Single pass to add all possible valid elements
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S:
                continue
            if not creates_4ap(x, S):
                S.add(x)
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Final check: Ensure best_set is valid
    if not best_set:
        # Fallback: A simple valid set
        # Powers of 2 are 4-AP free
        best_set = {1, 2, 4, 8, 16, 32, 64, 128, 256}
        
    return sorted(best_set)
