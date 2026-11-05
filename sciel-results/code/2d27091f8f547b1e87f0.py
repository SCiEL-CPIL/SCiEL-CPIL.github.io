import time
import random

def construct():
    N = 256
    # 0-indexed indices for values 1..256
    # val = idx + 1
    
    # Build conflict list
    # conflict[i] is a set of j
    conflict = [set() for _ in range(N)]
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict[i].add(j)
                conflict[j].add(i)
                
    # Initial set: all odds (indices 0, 2, 4, ...)
    # Value 1 is index 0. Value 3 is index 2.
    # Odds are values 1, 3, 5... -> indices 0, 2, 4...
    
    evens_indices = [i for i in range(1, N, 2)]
    
    def can_add(x, S):
        # Check if x conflicts with any y in S
        # x's conflicts are in conflict[x]
        # We need to see if any y in conflict[x] is in S
        for y in conflict[x]:
            if y in S:
                return False
        return True
        
    best_S = set()
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Run multiple random restarts
    # The greedy order of evens matters.
    
    while time.time() < deadline:
        S = set(i for i in range(0, N, 2))
        
        # Shuffle evens
        random.shuffle(evens_indices)
        
        for e in evens_indices:
            if e not in S:
                if can_add(e, S):
                    S.add(e)
                    
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    # Return values (indices + 1)
    return [idx + 1 for idx in best_S]
