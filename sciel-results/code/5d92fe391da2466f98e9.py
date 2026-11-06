import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n_max = 4096
    best_size = 0
    best_set = []
    
    def can_add(x, S, sums):
        """
        Check if x can be added to S without creating triple sum collisions.
        S: list of current elements
        sums: set of all triple sums of S
        """
        k = len(S)
        
        # Generate new sums involving x
        # Types:
        # 1. (x, x, x) -> 3x
        # 2. (x, x, s) -> 2x + s
        # 3. (x, s, s) -> x + 2s
        # 4. (x, s, t) -> x + s + t (s <= t)
        
        new_sums = set()
        new_sums.add(3 * x)
        
        for s in S:
            new_sums.add(2 * x + s)
            new_sums.add(x + 2 * s)
            
        for i in range(k):
            for j in range(i, k):
                new_sums.add(x + S[i] + S[j])
        
        # Check for collisions with existing sums
        if new_sums & sums:
            return False
        
        # Check for internal collisions among new sums
        # Count expected number of distinct triples involving x
        # 1 (xxx) + k (xxs) + k (xss) + k(k-1)/2 (xst)
        expected_count = 1 + 2 * k + k * (k - 1) // 2
        
        if len(new_sums) != expected_count:
            return False
            
        return True

    def update_sums(S, sums, x):
        """Update the sums set after adding x to S."""
        k = len(S) - 1 # S includes x at the end
        sums.add(3 * x)
        for i in range(k):
            sums.add(2 * x + S[i])
            sums.add(x + 2 * S[i])
        for i in range(k):
            for j in range(i, k):
                sums.add(x + S[i] + S[j])

    # Run multiple greedy attempts
    while time.time() < deadline:
        S = []
        sums = set()
        
        # Random order of candidates
        candidates = list(range(1, n_max + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S, sums):
                S.append(x)
                update_sums(S, sums, x)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = list(S)
        
        # If we found a very good set, we might want to stop early
        # But let's keep going to see if we can do better
        if best_size >= 25:
            break

    return best_set
