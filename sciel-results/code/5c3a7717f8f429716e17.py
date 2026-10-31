import time
import math
import itertools

def compute_score(U):
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -1
    max_u = max(U)
    if max_u == 0:
        return 1.0
    # Compute U+U
    sum_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
    # Compute U-U
    diff_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try parametric constructions
    def try_set(U):
        nonlocal best_U, best_score
        U = sorted(set(U))
        if not U or U[0] != 0:
            return
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Construction 1: blocks of consecutive pairs with growing gaps
    for k in range(1, 15):
        if time.time() > deadline:
            break
        U = set()
        for i in range(k + 1):
            base = 0
            for j in range(i):
                base += 2 * (3 ** j) + 1  # gap after each block
            U.add(base)
            U.add(base + 1)
        try_set(list(U))
    
    # Construction 2: U = {0, 1, 3, 4, 9, 10, 12, 13, ...} pattern
    for k in range(1, 12):
        if time.time() > deadline:
            break
        U = set()
        pos = 0
        for i in range(k + 1):
            U.add(pos)
            U.add(pos + 1)
            pos += 2 + 2 * (3 ** i)  # gap
        try_set(list(U))
    
    # Construction 3: arithmetic progression with perturbations
    for n in range(2, 30):
        if time.time() > deadline:
            break
        for d in range(1, 10):
            if time.time() > deadline:
                break
            U = [i * d for i in range(n)]
            # Add some perturbations
            for variant in range(5):
                Uv = list(U)
                # Perturb some elements
                for idx in range(1, n):
                    if (idx + variant) % 3 == 0:
                        Uv[idx] += 1
                try_set(Uv)
    
    # Local search: try adding/removing elements near existing ones
    def local_search(U, max_iters=500):
        nonlocal best_U, best_score
        U = sorted(set(U))
        if not U or U[0] != 0:
            return
        current_score = compute_score(U)
        for _ in range(max_iters):
            if time.time() > deadline:
                return
            improved = False
            max_u = max(U)
            # Try adding elements
            candidates = set()
            for u in U:
                for delta in range(-3, 4):
                    c = u + delta
                    if c >= 0 and c not in U:
                        candidates.add(c)
            # Also try elements just above max
            for c in range(max_u + 1, max_u + 5):
                candidates.add(c)
            
            for c in candidates:
                if time.time() > deadline:
                    return
                U_new = sorted(set(U + [c]))
                s = compute_score(U_new)
                if s > current_score:
                    U = U_new
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_U = list(U)
                    break
            
            # Try removing elements (not 0)
            if not improved:
                for idx in range(1, len(U)):
                    if time.time() > deadline:
                        return
                    U_new = sorted(set(U[:idx] + U[idx+1:]))
                    if not U_new or U_new[0] != 0:
                        continue
                    s = compute_score(U_new)
                    if s > current_score:
                        U = U_new
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_U = list(U)
                        break
            
            if not improved:
                break
        return U
    
    # Run local search from several starting points
    starts = [
        [0, 1, 3],
        [0, 1, 3, 4],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 2, 5, 6, 9, 10, 11],
    ]
    
    for start in starts:
        if time.time() > deadline:
            break
        U = list(start)
        # Ensure 0 is present
        if 0 not in U:
            U.append(0)
        U = sorted(set(U))
        local_search(U)
    
    # Final: try a few more structured sets
    for n in range(2, 20):
        if time.time() > deadline:
            break
        # U = {0, 1, n, n+1, 2n, 2n+1, ...}
        U = []
        for i in range(n):
            U.append(i * n)
            U.append(i * n + 1)
        try_set(U)
        local_search(U, max_iters=200)
    
    return best_U
