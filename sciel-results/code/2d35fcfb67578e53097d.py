import random
import time
def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_set = set(range(65, 129))
    best_size = 64
    
    # Try descending order
    S = set()
    for x in range(128, 0, -1):
        valid = True
        # Check conflicts
        for u in S:
            for v in S:
                if u == v: continue
                if (x + u) in S and (x + v) in S and (u + v) in S:
                    valid = False
                    break
            if not valid: break
        if valid:
            for u in S:
                v = x - u
                if v in S and u != v:
                    for c in S:
                        if c == u or c == v: continue
                        if (u + c) in S and (v + c) in S:
                            valid = False
                            break
                    if not valid: break
        if valid:
            S.add(x)
            
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # Try random shuffles
    for _ in range(50):
        if time.time() > deadline: break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = set()
        for x in candidates:
            valid = True
            for u in S:
                for v in S:
                    if u == v: continue
                    if (x + u) in S and (x + v) in S and (u + v) in S:
                        valid = False
                        break
                if not valid: break
            if valid:
                for u in S:
                    v = x - u
                    if v in S and u != v:
                        for c in S:
                            if c == u or c == v: continue
                            if (u + c) in S and (v + c) in S:
                                valid = False
                                break
                        if not valid: break
            if valid:
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return list(best_set)
