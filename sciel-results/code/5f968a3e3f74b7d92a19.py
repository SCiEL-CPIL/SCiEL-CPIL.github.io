import math
import time

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    start_time = time.time()
    deadline = 60.0  # Keep margin from 90s
    
    # Generate all pairs (a, b) with a <= b such that a+b is a square
    # We can precompute a boolean matrix or a list of conflicts for each node
    # n = 256
    # Max sum = 512
    # Squares: 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484
    
    n = 256
    conflicts = [[] for _ in range(n + 1)]
    squares = [i*i for i in range(2, int(math.isqrt(2*n)) + 2)]
    
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            s = a + b
            if s in squares:
                conflicts[a].append(b)
                if a != b:
                    conflicts[b].append(a)
                
    # Seed set: numbers congruent to 1 mod 3
    S = set()
    for i in range(1, n + 1):
        if i % 3 == 1:
            S.add(i)
            
    # Candidates to add
    candidates = [i for i in range(1, n + 1) if i not in S]
    
    # Sort candidates by degree (number of conflicts) to maybe add "easier" ones first?
    # Or just iterate. Let's sort by degree ascending.
    candidates.sort(key=lambda x: len(conflicts[x]))
    
    added_count = 0
    for c in candidates:
        if time.time() > deadline - 1.0:
            break
        # Check if c conflicts with any in S
        # Since S can be large, and conflicts[c] is small, check against S
        # conflicts[c] contains all b such that c+b is square.
        # We need to check if any b in conflicts[c] is in S.
        # Note: conflicts[c] includes c itself if 2c is square.
        is_conflict = False
        for b in conflicts[c]:
            if b in S:
                is_conflict = True
                break
        if not is_conflict:
            S.add(c)
            added_count += 1
            
    return list(S)
