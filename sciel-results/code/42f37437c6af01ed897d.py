import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s
    
    N = 128
    
    # Start with the safe base: {65, ..., 128}
    # This has 64 elements and no pair sum can be in S (all sums > 128)
    base = set(range(65, N + 1))
    
    # Try to add elements from {1, ..., 64}
    # For a triple (a,b,c) in S, violation if a+b, a+c, b+c all in S
    # Key: if we add small elements, we need to check triples involving them
    
    def is_valid(S, candidate):
        """Check if adding candidate to S maintains validity."""
        S_new = S | {candidate}
        
        # 1. Check triples where candidate is a member: (candidate, x, y)
        #    Violation if (candidate+x), (candidate+y), (x+y) are all in S_new
        elems = list(S)
        for i in range(len(elems)):
            x = elems[i]
            for j in range(i + 1, len(elems)):
                y = elems[j]
                s1 = candidate + x
                s2 = candidate + y
                s3 = x + y
                if s1 in S_new and s2 in S_new and s3 in S_new:
                    return False
        
        # 2. Check triples where candidate is a sum: (a, b, c) with a+b == candidate
        #    Violation if a+c and b+c are in S (and thus in S_new)
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i + 1, len(elems)):
                b = elems[j]
                if a + b == candidate:
                    # Check if there is a c in S (c != a, c != b)
                    # such that a+c in S and b+c in S
                    for c in elems:
                        if c == a or c == b:
                            continue
                        if (a + c) in S and (b + c) in S:
                            return False
                            
        return True
    
    # Greedy: try to add elements from 1 to 64
    S = set(base)
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if is_valid(S, x):
            S.add(x)
    
    # Local search: try to add more elements by removing some
    # Try random restarts with different orderings
    best_S = set(S)
    
    # Try multiple random orderings
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Start with base and try random ordering of small elements
        S_try = set(base)
        small_elements = list(range(1, 65))
        rng.shuffle(small_elements)
        
        for x in small_elements:
            if time.time() > deadline:
                break
            if is_valid(S_try, x):
                S_try.add(x)
        
        if len(S_try) > len(best_S):
            best_S = set(S_try)
    
    # Further local search: try to swap elements
    # Try removing one element and adding another
    for trial in range(500):
        if time.time() > deadline:
            break
        
        S_try = set(best_S)
        improved = False
        
        # Try to add any element not in S_try
        missing = [x for x in range(1, N + 1) if x not in S_try]
        rng.shuffle(missing)
        
        for x in missing:
            if time.time() > deadline:
                break
            if is_valid(S_try, x):
                S_try.add(x)
                improved = True
        
        if improved and len(S_try) > len(best_S):
            best_S = set(S_try)
    
    # Convert to list of Python ints
    result = sorted(best_S)
    return [int(x) for x in result]
