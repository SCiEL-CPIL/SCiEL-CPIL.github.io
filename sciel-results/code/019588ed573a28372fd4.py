import time
import math
import numpy as np
import random
from itertools import combinations

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    # Compute U+U and U-U using numpy for speed
    U_arr = np.array(U)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def local_optimize(U, deadline, max_iters=500):
    """Local search: try adding/removing/adjusting elements to improve score."""
    U = sorted(set(U))
    current_score = compute_score(U)
    best_U = U[:]
    best_score = current_score
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        
        # Try adding a new element
        if n < 15:
            # Try elements near existing ones or at strategic positions
            candidates = set()
            for u in U:
                candidates.add(u + 1)
                candidates.add(max(1, u - 1))
                candidates.add(u * 2)
                candidates.add(u + 3)
                candidates.add(u + 7)
            # Also try some larger jumps
            maxU = U[-1] if U else 1
            for factor in [2, 3, 5, 10]:
                candidates.add(maxU * factor)
                candidates.add(maxU * factor + 1)
            
            for cand in sorted(candidates):
                if cand < 0 or cand in U:
                    continue
                new_U = sorted(U + [cand])
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break
        
        # Try removing an element (keep at least 2)
        if n > 2:
            for i in range(n):
                new_U = U[:i] + U[i+1:]
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break
        
        # Try adjusting an element
        if n > 1:
            for i in range(n):
                if i == 0:
                    continue  # Keep 0
                for delta in [-2, -1, 1, 2, 3, 5]:
                    new_val = U[i] + delta
                    if new_val < 0 or new_val in U:
                        continue
                    new_U = U[:i] + [new_val] + U[i+1:]
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    s = compute_score(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U
                        U = new_U
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_U, best_score

def construct():
    start = time.time()
    deadline = start + 140  # safety margin
    
    best_score = 0
    best_U = [0, 1, 3]
    
    # Try the seed
    s = compute_score([0, 1, 3])
    if s > best_score:
        best_score = s
        best_U = [0, 1, 3]
    
    # Try parameterized families
    # Family 1: {0, 1, M, M+1}
    for M in [2, 3, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
        if time.time() > deadline:
            break
        U = [0, 1, M, M+1]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 2: {0, 1, 2, M, M+1}
    for M in [3, 5, 10, 50, 100, 500, 1000]:
        if time.time() > deadline:
            break
        U = [0, 1, 2, M, M+1]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 3: Geometric progression {0, 1, r, r^2, ...}
    for r in [2, 3, 4, 5, 10]:
        for k in [3, 4, 5, 6, 7]:
            if time.time() > deadline:
                break
            U = [0]
            val = 1
            for i in range(k-1):
                U.append(val)
                val *= r
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Family 4: {0, 1, 2, ..., k-1, M, M+1, ..., M+k-1}
    for k in [2, 3, 4, 5]:
        for M in [k+1, 2*k, 5*k, 10*k, 50*k, 100*k]:
            if time.time() > deadline:
                break
            U = list(range(k)) + list(range(M, M+k))
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Family 5: {0, 1, M, M+2, 2M}
    for M in [3, 5, 10, 50, 100, 500]:
        if time.time() > deadline:
            break
        U = [0, 1, M, M+2, 2*M]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 6: Mersenne-like numbers {0, 1, 2^a-1, 2^b-1, ...}
    for a in [2, 3, 4, 5, 6]:
        for b in range(a+1, min(a+5, 12)):
            if time.time() > deadline:
                break
            U = [0, 1, (1<<a)-1, (1<<b)-1]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Family 7: {0, 1, M, 2M, 3M, ..., kM} with small k
    for k in [2, 3, 4, 5]:
        for M in [2, 3, 5, 7, 10, 20, 50]:
            if time.time() > deadline:
                break
            U = [0, 1] + [i*M for i in range(1, k+1)]
            U = sorted(set(U))
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Family 8: Random structured sets
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        # Random geometric-ish progression
        r = random.choice([2, 3, 4, 5, 7])
        k = random.randint(3, 7)
        U = [0, 1]
        val = random.randint(2, 10)
        for i in range(k-2):
            U.append(val)
            val *= r
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Local optimization on best found so far
    if time.time() < deadline - 10:
        opt_U, opt_score = local_optimize(best_U, deadline, max_iters=200)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U
    
    # Try local optimization on a few other promising candidates
    if time.time() < deadline - 5:
        for candidate in [[0, 1, 3, 7], [0, 1, 4, 9], [0, 1, 2, 5, 10]]:
            if time.time() > deadline - 2:
                break
            opt_U, opt_score = local_optimize(candidate, deadline, max_iters=100)
            if opt_score > best_score:
                best_score = opt_score
                best_U = opt_U
    
    return best_U
