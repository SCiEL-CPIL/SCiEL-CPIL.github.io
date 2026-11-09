"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 256
    max_time = 55.0
    
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                diffs.add(abs(S[i] - S[j]))
        return diffs

    best_A = []
    best_B = []
    best_score = 0
    
    start_time = time.time()
    
    for trial in range(50):
        if time.time() - start_time > max_time:
            break
            
        random.seed(1000 + trial)
        
        A = []
        B = []
        dA = set()
        dB = set()
        
        # Randomized Greedy Initialization
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            added = False
            new_dA = set()
            can_add_A = True
            for a in A:
                d = abs(x - a)
                if d in dA or d in dB:
                    can_add_A = False
                    break
                new_dA.add(d)
            
            if can_add_A:
                temp_A = A + [x]
                if is_sidon(temp_A):
                    A.append(x)
                    dA.update(new_dA)
                    added = True
            
            # Try adding to B
            if not added:
                new_dB = set()
                can_add_B = True
                for b in B:
                    d = abs(x - b)
                    if d in dA or d in dB:
                        can_add_B = False
                        break
                    new_dB.add(d)
                
                if can_add_B:
                    temp_B = B + [x]
                    if is_sidon(temp_B):
                        B.append(x)
                        dB.update(new_dB)
        
        current_score = len(A) + len(B)
        
        # Local Search: Hill Climbing
        # We try to add elements. If stuck, we try swaps (remove one, add another).
        # To keep it fast, we only do a few iterations of local search.
        
        improved = True
        local_iter = 0
        max_local_iter = 100
        
        while improved and local_iter < max_local_iter and time.time() - start_time < max_time:
            improved = False
            local_iter += 1
            
            # 1. Try adding any element
            for x in range(1, n + 1):
                if x not in A:
                    new_dA = set()
                    can_add_A = True
                    for a in A:
                        d = abs(x - a)
                        if d in dA or d in dB:
                            can_add_A = False
                            break
                        new_dA.add(d)
                    
                    if can_add_A and is_sidon(A + [x]):
                        A.append(x)
                        dA.update(new_dA)
                        improved = True
                        break
                
                if x not in B:
                    new_dB = set()
                    can_add_B = True
                    for b in B:
                        d = abs(x - b)
                        if d in dA or d in dB:
                            can_add_B = False
                            break
                        new_dB.add(d)
                    
                    if can_add_B and is_sidon(B + [x]):
                        B.append(x)
                        dB.update(new_dB)
                        improved = True
                        break
            
            # 2. If no addition, try a random swap to escape local minimum
            if not improved:
                # Remove a random element from A or B
                # Then try to add any element
                if A:
                    idx = random.randint(0, len(A) - 1)
                    x = A.pop(idx)
                    dA = get_diffs(A)
                    # Try to add any element to A or B
                    for y in range(1, n + 1):
                        if y not in A:
                            ndA = set()
                            can_add_A = True
                            for a in A:
                                d = abs(y - a)
                                if d in dA or d in dB:
                                    can_add_A = False
                                    break
                                ndA.add(d)
                            if can_add_A and is_sidon(A + [y]):
                                A.append(y)
                                dA.update(ndA)
                                improved = True
                                break
                        if y not in B:
                            ndB = set()
                            can_add_B = True
                            for b in B:
                                d = abs(y - b)
                                if d in dA or d in dB:
                                    can_add_B = False
                                    break
                                ndB.add(d)
                            if can_add_B and is_sidon(B + [y]):
                                B.append(y)
                                dB.update(ndB)
                                improved = True
                                break
                    if not improved:
                        # Put it back if we didn't improve
                        A.append(x)
                        dA = get_diffs(A)
                
                elif B:
                    idx = random.randint(0, len(B) - 1)
                    x = B.pop(idx)
                    dB = get_diffs(B)
                    for y in range(1, n + 1):
                        if y not in A:
                            ndA = set()
                            can_add_A = True
                            for a in A:
                                d = abs(y - a)
                                if d in dA or d in dB:
                                    can_add_A = False
                                    break
                                ndA.add(d)
                            if can_add_A and is_sidon(A + [y]):
                                A.append(y)
                                dA.update(ndA)
                                improved = True
                                break
                        if y not in B:
                            ndB = set()
                            can_add_B = True
                            for b in B:
                                d = abs(y - b)
                                if d in dA or d in dB:
                                    can_add_B = False
                                    break
                                ndB.add(d)
                            if can_add_B and is_sidon(B + [y]):
                                B.append(y)
                                dB.update(ndB)
                                improved = True
                                break
                    if not improved:
                        B.append(x)
                        dB = get_diffs(B)
        
        current_score = len(A) + len(B)
        if current_score > best_score:
            best_score = current_score
            best_A = A[:]
            best_B = B[:]

    return [best_A, best_B]

# EVOLVE-BLOCK-END
