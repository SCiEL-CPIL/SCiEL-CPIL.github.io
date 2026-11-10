import time
import random

def construct():
    N = 512
    deadline = time.time() + 55.0 # 55s to be safe within 60s limit
    
    # Helper to check if x can be added to set S
    def can_add(x, S):
        # Check if adding x creates a 4-AP
        # x is 4th term: x-3d, x-2d, x-d in S
        # x is 3rd term: x-2d, x-d, x+d in S
        # x is 2nd term: x-d, x+d, x+2d in S
        # x is 1st term: x+d, x+2d, x+3d in S
        
        # Max d such that terms are in [1, N]
        # For 4th term: x-3d >= 1 => d <= (x-1)//3
        max_d_4 = (x - 1) // 3
        for d in range(1, max_d_4 + 1):
            if (x - d) in S and (x - 2*d) in S and (x - 3*d) in S:
                return False
        
        # For 3rd term: x-2d >= 1 and x+d <= N => d <= (x-1)//2 and d <= N-x
        max_d_3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d_3 + 1):
            if (x - d) in S and (x - 2*d) in S and (x + d) in S:
                return False
        
        # For 2nd term: x-d >= 1 and x+2d <= N => d <= x-1 and d <= (N-x)//2
        max_d_2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d_2 + 1):
            if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                return False
        
        # For 1st term: x+3d <= N => d <= (N-x)//3
        max_d_1 = (N - x) // 3
        for d in range(1, max_d_1 + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return False
        
        return True

    # Initial Greedy Construction
    S = set()
    for x in range(1, N + 1):
        if can_add(x, S):
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    # Local Search: Kick and Recover
    # We will randomly remove a few elements and then greedily refill.
    # We also try randomizing the order of refilling.
    
    candidates = list(range(1, N + 1))
    
    while time.time() < deadline:
        # Kick: Remove 1 to 4 random elements
        kick_size = random.randint(1, 4)
        if len(S) > kick_size:
            to_remove = random.sample(list(S), kick_size)
            S_kicked = set(S)
            for x in to_remove:
                S_kicked.discard(x)
        else:
            S_kicked = set(S)
            
        # Recover: Try to add elements from the complement
        # Randomize the order of candidates to explore different local optima
        remaining = [x for x in candidates if x not in S_kicked]
        random.shuffle(remaining)
        
        S_recovered = set(S_kicked)
        
        for x in remaining:
            if can_add(x, S_recovered):
                S_recovered.add(x)
                
        if len(S_recovered) > best_size:
            best_size = len(S_recovered)
            best_S = set(S_recovered)
            # Update S to the new best for the next kick
            S = set(S_recovered)
            
        # Occasionally, try a single swap to see if we can improve structure or size
        # If size is same, we might want to diversify, but here we only care about size.
        # So we only update S if it improved.
        
    return sorted(best_S)
