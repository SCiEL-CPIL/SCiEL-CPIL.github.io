import time
import math
import random

def _score_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0
    maxU = max(U_set)
    if maxU == 0:
        return 0
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
    P = len(sumset)
    D = len(diffset)
    if P == 0:
        return 0
    ratio = D / P
    if ratio <= 1:
        return 1.0
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def _local_search(U, time_limit, seed=42):
    """Local search to improve a set."""
    rng = random.Random(seed)
    best_U = list(U)
    best_score = _score_set(best_U)
    deadline = time.time() + time_limit
    current_U = list(best_U)
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        maxU = max(current_U) if current_U else 0
        
        # Try adding elements
        candidates = [x for x in range(1, maxU + 25) if x not in current_U]
        rng.shuffle(candidates)
        for cand in candidates[:40]:
            new_U = current_U + [cand]
            s = _score_set(new_U)
            if s > current_score + 1e-12:
                current_U = new_U
                current_score = s
                if s > best_score:
                    best_U = list(current_U)
                    best_score = s
                improved = True
                break
        
        if not improved:
            # Try removing
            removable = [x for x in current_U if x != 0]
            rng.shuffle(removable)
            for cand in removable[:20]:
                new_U = [x for x in current_U if x != cand]
                if len(new_U) < 2:
                    continue
                s = _score_set(new_U)
                if s > current_score + 1e-12:
                    current_U = new_U
                    current_score = s
                    if s > best_score:
                        best_U = list(current_U)
                        best_score = s
                    improved = True
                    break
        
        if not improved:
            # Try swap
            if len(current_U) >= 3:
                removable = [x for x in current_U if x != 0]
                if removable:
                    remove_elem = rng.choice(removable)
                    maxU = max(current_U)
                    candidates = [x for x in range(1, maxU + 25) if x not in current_U]
                    rng.shuffle(candidates)
                    for cand in candidates[:25]:
                        new_U = [x for x in current_U if x != remove_elem] + [cand]
                        s = _score_set(new_U)
                        if s > current_score + 1e-12:
                            current_U = new_U
                            current_score = s
                            if s > best_score:
                                best_U = list(current_U)
                                best_score = s
                            improved = True
                            break
        
        if not improved:
            # Random perturbation to escape local optima
            if len(current_U) >= 3:
                removable = [x for x in current_U if x != 0]
                for _ in range(2):
                    if not removable:
                        break
                    remove_elem = rng.choice(removable)
                    removable.remove(remove_elem)
                    current_U = [x for x in current_U if x != remove_elem]
                    if len(current_U) < 2:
                        break
                    maxU = max(current_U) if current_U else 0
                    candidates = [x for x in range(1, maxU + 25) if x not in current_U]
                    if candidates:
                        current_U.append(rng.choice(candidates))
                    current_score = _score_set(current_U)
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 108  # 12s safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Phase 1: Brute force subsets of {1,...,K} for small K
    for K in range(2, 14):
        if time.time() > deadline:
            break
        for mask in range(1, (1 << K)):
            if time.time() > deadline:
                break
            S = [i + 1 for i in range(K) if mask & (1 << i)]
            U = [0] + S
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
    
    # Phase 2: Local search from best brute-force result
    if time.time() < deadline - 3:
        time_left = deadline - time.time() - 3
        best_U, best_score = _local_search(best_U, time_left, seed=42)
    
    # Phase 3: Try structured sets and local search from multiple starts
    if time.time() < deadline - 1:
        # Try Mersenne-like sets {0, 1, 3, 7, 15, ...}
        for K in range(2, 13):
            if time.time() > deadline:
                break
            U = [0] + [2**i - 1 for i in range(1, K)]
            if max(U) > 8000:
                break
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
        
        # Local search from various starting points
        starts = [
            [0, 1, 3],
            [0, 1, 4],
            [0, 1, 3, 7],
            [0, 1, 4, 6],
            [0, 2, 3, 5],
            [0, 1, 3, 8],
            [0, 1, 4, 9],
        ]
        for start in starts:
            if time.time() > deadline:
                break
            s = _score_set(start)
            if s > best_score:
                best_score = s
                best_U = list(start)
            if time.time() < deadline - 1:
                t_left = min(3.0, deadline - time.time() - 1)
                u, sc = _local_search(start, t_left, seed=random.randint(0, 100000))
                if sc > best_score:
                    best_score = sc
                    best_U = u
    
    return sorted(best_U)
