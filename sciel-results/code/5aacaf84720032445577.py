import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 4096
    best_set = []
    
    seed = 42
    while time.time() < deadline:
        # Shuffle candidates
        candidates = list(range(1, n + 1))
        random.seed(seed)
        random.shuffle(candidates)
        seed += 1
        
        S = []
        existing_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Check if adding x creates a conflict
            # New sums involving x:
            # 1. x + x + x = 3x
            # 2. x + x + y = 2x + y for y in S
            # 3. x + y + z = x + y + z for y, z in S with y <= z
            
            new_sums = {3 * x}
            
            # Add 2x + y
            for y in S:
                new_sums.add(2 * x + y)
            
            # Add x + y + z
            # Iterate unique pairs (y, z) from S with y <= z
            k = len(S)
            for i in range(k):
                y = S[i]
                # y == z case
                new_sums.add(x + 2 * y)
                # y < z case
                for j in range(i + 1, k):
                    z = S[j]
                    new_sums.add(x + y + z)
            
            # Check for internal collisions in new_sums
            expected_count = 1 + k + k * (k + 1) // 2
            
            if len(new_sums) != expected_count:
                continue
                
            # Check for collisions with existing sums
            if new_sums & existing_sums:
                continue
                
            # Accept x
            S.append(x)
            existing_sums |= new_sums
            
        # Update best set
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
