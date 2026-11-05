import random
import time

def construct():
    best_set = []
    start_time = time.time()
    
    # We can run multiple trials to find a larger set
    # The time limit is 60s.
    # Each trial is O(N * K^2). With N=4096, K~40, it's ~6.5M ops.
    # In Python, 6.5M simple ops might take ~0.1-0.5s.
    # So we can run ~100-500 trials.
    
    for trial in range(500):
        if time.time() - start_time > 55:
            break
            
        A = []
        sum_set = set()
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        for x in candidates:
            k = len(A)
            # Check if x can be added
            # Generate new sums
            s_new = set()
            s_new.add(3 * x)
            for y in A:
                s_new.add(2 * x + y)
            # Triples (x, y, z) with y <= z from A
            for i in range(k):
                y = A[i]
                for j in range(i, k):
                    z = A[j]
                    s_new.add(x + y + z)
            
            num_expected = (k + 1) * (k + 2) // 2
            
            # Check for internal collisions
            if len(s_new) != num_expected:
                continue
                
            # Check for collisions with existing sums
            if s_new & sum_set:
                continue
            
            # Valid, add x
            A.append(x)
            sum_set.update(s_new)
            
        if len(A) > len(best_set):
            best_set = A
            
    return best_set
