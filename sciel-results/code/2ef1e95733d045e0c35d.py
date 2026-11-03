import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _elapsed():
    return time.time() - _START

def _score_direct(U):
    U = sorted(set(U))
    if not U or U[0] != 0:
        U = [0] + [x for x in U if x > 0]
    n = len(U)
    if n < 2:
        return 0.0
    M = U[-1]
    if M < 1:
        return 0.0
    
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(n):
            sums.add(U[i] + U[j])
            diffs.add(U[i] - U[j])
            
    sumset = len(sums)
    diffset = len(diffs)
    
    if sumset == 0:
        return 0.0
    ratio = diffset / sumset
    if ratio <= 1.0:
        return 1.0
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _golomb_ruler(n):
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    if n == 3:
        return [0, 1, 3]
    if n == 4:
        return [0, 1, 4, 6]
    if n == 5:
        return [0, 1, 4, 9, 11]
    if n == 6:
        return [0, 1, 4, 10, 12, 17]
    if n == 7:
        return [0, 1, 4, 10, 12, 17, 22]
    if n == 8:
        return [0, 1, 4, 10, 12, 17, 22, 28]
    # Fallback for larger n
    U = [0]
    for i in range(1, n):
        c = 1
        while True:
            U_test = sorted(U + [c])
            diffs = set()
            valid = True
            for j in range(len(U_test)):
                for k in range(j+1, len(U_test)):
                    d = U_test[k] - U_test[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                U.append(c)
                break
            c += 1
    return U

def construct():
    best_U = [0, 1, 4, 9, 11]
    best_score = _score_direct(best_U)
    rng = np.random.RandomState(42)
    
    # Initial candidates
    candidates = []
    for n in range(3, 9):
        candidates.append(_golomb_ruler(n))
    
    # Squares
    for k in range(3, 9):
        candidates.append([0] + [i**2 for i in range(1, k)])
    
    # Cubes
    for k in range(3, 7):
        candidates.append([0] + [i**3 for i in range(1, k)])
    
    # Random
    for _ in range(20):
        size = rng.randint(3, 8)
        max_val = rng.randint(10, 50)
        cands = rng.choice(range(1, max_val + 1), size=size, replace=False)
        candidates.append(sorted([0] + list(cands)))
    
    # Evaluate candidates
    for U in candidates:
        s = _score_direct(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
            
    # Local Search
    U = list(best_U)
    cur_score = best_score
    improved = True
    no_improve = 0
    
    while improved and _elapsed() < _DEADLINE:
        improved = False
        no_improve = 0
        max_val = max(U) if U else 1
        
        # 1. Try adding an element
        for c in range(1, max_val * 2 + 20):
            if _elapsed() > _DEADLINE: break
            if c not in U:
                trial = sorted(U + [c])
                s = _score_direct(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
                    break
            if improved: break
            no_improve += 1
            if no_improve > 50: break
            
        # 2. Try removing an element
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > _DEADLINE: break
                trial = U[:idx] + U[idx+1:]
                s = _score_direct(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
                    break
        
        # 3. Try swapping an element
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > _DEADLINE: break
                old_val = U[idx]
                for new_val in range(1, max_val * 2 + 10):
                    if new_val == old_val or new_val in U:
                        continue
                    trial = sorted(U[:idx] + [new_val] + U[idx+1:])
                    s = _score_direct(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        best_U = U
                        best_score = cur_score
                        improved = True
                        break
                if improved: break

    return best_U
