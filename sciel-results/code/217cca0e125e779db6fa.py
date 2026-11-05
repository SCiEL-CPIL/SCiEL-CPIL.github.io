import time
import random
import sys

def construct():
    N = 256
    # 0-indexed: 0 corresponds to 1, ... 255 corresponds to 256
    # Or just use 1..256
    
    # Precompute conflicts
    # conflict[i][j] = True if (i+1) and (j+1) are bad pair
    # Actually, let's use 0..255 indices for numbers 1..256
    # num = idx + 1
    
    # We want a subset of indices.
    
    # Build adjacency list or matrix
    # N=256, matrix is 256x256 = 65k entries. Fast.
    
    conflict = [[False] * N for _ in range(N)]
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict[i][j] = True
                conflict[j][i] = True
                
    # Initial set: all odds
    # Odds are 1, 3, 5, ... 255
    # Indices: 0, 2, 4, ... 254
    S = set(i for i in range(0, N, 2))
    
    # Function to check if x can be added
    def can_add(x):
        for y in S:
            if conflict[x][y]:
                return False
        return True
        
    # Greedy add
    # Try to add evens
    evens = [i for i in range(1, N, 2)]
    # Sort evens by number of conflicts with current S?
    # Or just iterate
    
    added_any = True
    while added_any:
        added_any = False
        for e in evens:
            if e not in S and can_add(e):
                S.add(e)
                added_any = True
                # Restart loop to check others again? 
                # Since adding e only increases conflicts, 
                # if e couldn't be added before, it still can't.
                # But if we remove something, it might.
                # Here we only add.
                
    # Local Search
    # Try to swap: remove y, add x
    # If we remove y, we might be able to add x and maybe more.
    # Simple move: if x conflicts only with y, we can swap y with x.
    # Size stays same. But maybe x has fewer conflicts with others, allowing more adds later?
    # Or maybe we can do 2-for-1: remove y1, y2, add x.
    
    # Let's try a simple hill climbing on the set.
    # We want to maximize |S|.
    
    # Actually, the "add if possible" greedy is quite good for sparse graphs.
    # Let's try to improve by checking if we can replace a subset of S with a larger subset of not-S.
    
    # Given time, let's just return the greedy result.
    # But we can try random restarts or shuffling the order of evens.
    
    best_S = set(S)
    best_size = len(S)
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try a few random orders of evens for greedy
    for _ in range(100):
        if time.time() > deadline:
            break
        S = set(i for i in range(0, N, 2))
        random.shuffle(evens)
        for e in evens:
            if e not in S:
                if can_add(e):
                    S.add(e)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    # If we have time, do local search on best_S
    # Try to find any x not in best_S that can be added (should be none if greedy was thorough, 
    # but maybe order matters)
    
    # Check for single swaps that increase size? No, swap keeps size same.
    # We need to find a set of removals R and additions A such that |A| > |R| and A is compatible with S\R.
    
    # Convert indices (0..255) back to values (1..256)
    return [idx + 1 for idx in best_S]
