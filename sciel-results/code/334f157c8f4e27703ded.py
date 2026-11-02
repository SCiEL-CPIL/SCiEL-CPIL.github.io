import time
import random
from itertools import combinations

def score_set(U):
    if not U or 0 not in U:
        return -1
    U = sorted(list(set(U)))
    max_u = max(U)
    if max_u == 0:
        return 1.0 # log(1/1)/log(1) is undefined, but limit is 1? 
        # Actually if U={0}, U+U={0}, U-U={0}. Ratio 1. log(1)=0.
        # Denom log(1)=0. 0/0.
        # But task says nonnegative integers.
        # Let's assume U has at least 2 elements or max_u > 0.
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    s_size = len(sums)
    d_size = len(diffs)
    
    if s_size == 0 or d_size == 0:
        return -1
        
    ratio = d_size / s_size
    if ratio <= 1:
        # Score <= 1
        # log(ratio) <= 0
        return 1.0 + (0.0 if ratio == 1 else -1) # Just return low score
    
    import math
    # Handle log(1) in denominator if max_u=0
    if max_u == 0:
        return 1.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit, 20s buffer
    
    best_score = 0.0
    best_set = [0, 1] # Initial guess
    
    # Update best with seed
    seed_score = score_set([0, 1, 3])
    if seed_score > best_score:
        best_score = seed_score
        best_set = [0, 1, 3]
        
    # Exhaustive search for small sets
    # k=2,3,4,5,6
    for k in range(2, 7):
        for M in range(k-1, 50):
            # Generate combinations of {1..M} of size k-1
            # 0 is always included
            candidates = list(combinations(range(1, M+1), k-1))
            for c in candidates:
                U = [0] + list(c)
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                if time.time() > deadline:
                    return best_set
                    
    # Random search / Hill Climbing for larger sets or M
    # We will try to find sets with larger M or k if they improve score
    # But note that score tends to decrease with M for fixed structure?
    # Let's try random sets with k up to 10, M up to 100
    
    random.seed(42)
    for _ in range(5000): # 5000 restarts
        if time.time() > deadline:
            break
            
        k = random.randint(3, 10)
        M = random.randint(k, 100)
        
        # Generate random set
        U = set([0])
        while len(U) < k:
            U.add(random.randint(1, M))
        U = list(U)
        
        current_score = score_set(U)
        if current_score > best_score:
            best_score = current_score
            best_set = U
            
        # Local search
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Try to swap, add, or remove
            op = random.choice(['swap', 'add', 'remove'])
            new_U = list(U)
            
            if op == 'swap' and len(new_U) > 1:
                i = random.randint(0, len(new_U)-1)
                if new_U[i] == 0:
                    continue # Keep 0
                new_val = random.randint(1, M+10)
                if new_val not in new_U:
                    new_U[i] = new_val
                    s = score_set(new_U)
                    if s > current_score:
                        U = new_U
                        current_score = s
                        best_score = max(best_score, s)
                        if best_score == s:
                            best_set = U
            elif op == 'add' and len(new_U) < 15:
                new_val = random.randint(1, M+10)
                if new_val not in new_U:
                    new_U.append(new_val)
                    s = score_set(new_U)
                    if s > current_score:
                        U = new_U
                        current_score = s
                        best_score = max(best_score, s)
                        if best_score == s:
                            best_set = U
            elif op == 'remove' and len(new_U) > 2:
                i = random.randint(0, len(new_U)-1)
                if new_U[i] == 0:
                    continue
                new_U.pop(i)
                s = score_set(new_U)
                if s > current_score:
                    U = new_U
                    current_score = s
                    best_score = max(best_score, s)
                    if best_score == s:
                        best_set = U
                        
    return best_set
