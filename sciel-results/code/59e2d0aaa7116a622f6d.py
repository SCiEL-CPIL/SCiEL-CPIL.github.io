import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill

    def compute_metrics(A):
        """Compute sizes of A+A and A*A."""
        n = len(A)
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return len(sums), len(prods)

    def objective(A):
        s, p = compute_metrics(A)
        return max(s, p)

    def try_candidate(A):
        if len(A) != 20:
            return False
        if len(set(A)) != 20:
            return False
        if any(x < 1 or x > 10000 for x in A):
            return False
        return True

    best_A = None
    best_obj = float('inf')

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, 21)))
    
    # 2. Powers of 2 (grows fast, keeps products distinct)
    p2 = [2**i for i in range(20)]
    if all(x <= 10000 for x in p2):
        candidates.append(p2)
    
    # 3. Arithmetic progression with small step
    for step in [1, 2, 3, 5]:
        ap = [1 + i * step for i in range(20)]
        if all(x <= 10000 for x in ap):
            candidates.append(ap)
    
    # 4. Geometric-like sequences
    for r in [1.1, 1.2, 1.3, 1.5]:
        gp = []
        v = 1.0
        while len(gp) < 20:
            iv = max(1, int(v))
            if iv > 10000:
                break
            if iv not in gp:
                gp.append(iv)
            v = v * r
        while len(gp) < 20:
            x = random.randint(1, 10000)
            if x not in gp:
                gp.append(x)
        candidates.append(gp)
    
    # 5. Random sets
    for _ in range(20):
        rs = random.sample(range(1, 10001), 20)
        candidates.append(rs)
    
    # 6. Sets biased toward small numbers
    for _ in range(10):
        rs = random.sample(range(1, 200), 20)
        candidates.append(rs)

    # Evaluate all initial candidates
    for A in candidates:
        if try_candidate(A):
            obj = objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = A[:]

    # Multi-start local search with different strategies
    for restart in range(50):
        if time.time() >= deadline:
            break
        
        # Start from best or a random candidate
        if restart == 0:
            A = best_A[:]
        elif restart < 10:
            A = candidates[random.randint(0, len(candidates)-1)][:]
        else:
            A = random.sample(range(1, 10001), 20)
        
        if not try_candidate(A):
            continue
            
        cur_obj = objective(A)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_A = A[:]
        
        # Local search: try replacing each element with nearby values
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(20):
                if time.time() >= deadline:
                    break
                
                # Try several replacement values around A[i]
                tried = set()
                for delta in range(-30, 31):
                    if delta == 0:
                        continue
                    new_val = A[i] + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in A and A.index(new_val) != i:
                        continue
                    if new_val in tried:
                        continue
                    tried.add(new_val)
                    
                    A[i] = new_val
                    if not try_candidate(A):
                        A[i] = A[i]  # revert
                        continue
                    
                    new_obj = objective(A)
                    if new_obj < cur_obj:
                        cur_obj = new_obj
                        improved = True
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best_A = A[:]
                        break
                    else:
                        A[i] = A[i]  # revert
                # Also try a few random global replacements
                for _ in range(5):
                    new_val = random.randint(1, 10000)
                    if new_val in A and A.index(new_val) != i:
                        continue
                    A[i] = new_val
                    if not try_candidate(A):
                        A[i] = A[i]  # revert
                        continue
                    new_obj = objective(A)
                    if new_obj < cur_obj:
                        cur_obj = new_obj
                        improved = True
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best_A = A[:]
                        break
                    else:
                        A[i] = A[i]  # revert
        
        # Periodically check time
        if time.time() >= deadline - 5:
            break

    return best_A
