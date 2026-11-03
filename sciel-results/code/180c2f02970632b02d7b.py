import time
import random
import math

def score_set(U):
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Calculate sizes
    # If N is small, use sets
    N = len(U)
    if N < 100:
        diffs = set()
        sums = set()
        for u1 in U:
            for u2 in U:
                diffs.add(u1 - u2)
                sums.add(u1 + u2)
        num_diffs = len(diffs)
        num_sums = len(sums)
    else:
        # Fallback for larger sets (less likely to be optimal for this score)
        diffs = set()
        sums = set()
        for u1 in U:
            for u2 in U:
                diffs.add(u1 - u2)
                sums.add(u1 + u2)
        num_diffs = len(diffs)
        num_sums = len(sums)
        
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        # Score < 1
        # 1 + negative
        return 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return 1 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # 1. Evaluate specific known good structures
    candidates = []
    
    # Seed
    candidates.append([0, 1, 3])
    
    # Small exhaustive search for N=4, 5
    # Too slow to do full exhaustive, so sample
    
    # Block structures
    for m in range(2, 12):
        for k in range(2, 12):
            # B should be around 2k to balance sum/diff overlap
            for B in range(max(2, 2*k - 2), 2*k + 4):
                U = []
                for j in range(m):
                    for i in range(k):
                        U.append(j * B + i)
                U = list(set(U))
                if 0 not in U:
                    U.append(0)
                U.sort()
                # Prune if max is too large relative to N?
                # Score drops if max is large.
                if max(U) < 200: 
                    candidates.append(U)
                
    # Geometric progressions
    for base in range(2, 10):
        for n in range(3, 8):
            U = [base**i for i in range(n)]
            U.append(0)
            U = list(set(U))
            U.sort()
            if max(U) < 200:
                candidates.append(U)
                
    # Random sets
    for _ in range(50):
        N = random.randint(3, 12)
        max_val = random.randint(5, 50)
        U = set([0])
        while len(U) < N:
            U.add(random.randint(1, max_val))
        U = list(U)
        U.sort()
        candidates.append(U)
        
    for U in candidates:
        if time.time() > deadline:
            break
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # 2. Local Search from best candidate
    current_set = best_set[:]
    current_score = best_score
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            curr = best_set[:]
            curr_score = best_score
        else:
            # Random restart
            N = random.randint(3, 15)
            max_val = random.randint(5, 60)
            curr = set([0])
            while len(curr) < N:
                curr.add(random.randint(1, max_val))
            curr = list(curr)
            curr_score = score_set(curr)
            
        # Hill Climbing
        for step in range(200):
            if time.time() > deadline:
                return best_set
            
            neighbor = curr[:]
            op = random.randint(0, 2)
            if op == 0: # Add
                if len(neighbor) < 20: # Limit size
                    max_u = max(neighbor) if neighbor else 0
                    new_el = random.randint(1, max_u + 15)
                    if new_el not in neighbor:
                        neighbor.append(new_el)
            elif op == 1: # Remove
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    neighbor.pop(idx)
            else: # Swap
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    max_u = max(neighbor)
                    new_el = random.randint(1, max_u + 10)
                    if new_el not in neighbor:
                        neighbor[idx] = new_el
            
            n_score = score_set(neighbor)
            if n_score > curr_score:
                curr = neighbor
                curr_score = n_score
                if curr_score > best_score:
                    best_score = curr_score
                    best_set = curr[:]
                    
    return best_set
