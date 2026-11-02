import time
import random
import math

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    mx = max(U)
    if mx == 0:
        return 1.0
    n = len(U)
    sset = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sset.add(ui + U[j])
    dset = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            dset.add(ui - U[j])
    ratio = len(dset) / len(sset)
    if ratio <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start = time.time()
    deadline = start + 110.0
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    rng = random.Random(42)
    
    def local_search(U, max_iter=1000):
        U = list(U)
        cur_score = _score_set(U)
        improved = True
        it = 0
        while improved and it < max_iter and time.time() < deadline:
            improved = False
            it += 1
            n = len(U)
            
            # Move 1: Change one element
            if not improved:
                for i in range(1, n):
                    orig = U[i]
                    # Search range
                    search_range = range(max(1, orig-10), max(1, orig-10) + 21)
                    for new_val in search_range:
                        if new_val == orig:
                            continue
                        trial = U[:]
                        trial[i] = new_val
                        if len(set(trial)) != len(trial):
                            continue
                        s = _score_set(trial)
                        if s > cur_score + 1e-12:
                            cur_score = s
                            U = trial
                            improved = True
                            break
                    if improved:
                        break
            
            # Move 2: Add element
            if not improved and n < 15:
                # Try adding small numbers
                for new_val in range(1, max(U) + 5):
                    if new_val in U:
                        continue
                    trial = U + [new_val]
                    s = _score_set(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        improved = True
                        break
            
            # Move 3: Remove element
            if not improved and n > 3:
                for i in range(1, n):
                    trial = U[:i] + U[i+1:]
                    s = _score_set(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        improved = True
                        break
                        
            # Move 4: Swap two elements
            if not improved and n >= 4:
                for i in range(1, n):
                    for j in range(i+1, n):
                        U[i], U[j] = U[j], U[i]
                        if len(set(U)) != len(U):
                            U[i], U[j] = U[j], U[i]
                            continue
                        s = _score_set(U)
                        if s > cur_score + 1e-12:
                            cur_score = s
                            improved = True
                            break
                        else:
                            U[i], U[j] = U[j], U[i]
                    if improved:
                        break
            
        return U, cur_score

    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 4],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 12],
    ]
    
    # Generate Mian-Chowla
    def mian_chowla(n):
        U = [0, 1]
        while len(U) < n:
            best_c = None
            best_s = -1
            for c in range(max(U) + 1, max(U) + 20):
                trial = U + [c]
                s = _score_set(trial)
                if s > best_s:
                    best_s = s
                    best_c = c
            if best_c is None:
                break
            U.append(best_c)
        return U

    for n in range(3, 10):
        candidates.append(mian_chowla(n))

    # Random starts
    for _ in range(20):
        n = rng.randint(3, 8)
        vals = rng.sample(range(1, 20), n-1)
        candidates.append([0] + sorted(vals))

    for c in candidates:
        s = _score_set(c)
        if s > best_score:
            best_score = s
            best_set = list(c)

    # Run local search on best and some others
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        # Pick a random candidate or best
        if rng.random() < 0.5:
            start_set = list(best_set)
        else:
            start_set = list(rng.choice(candidates))
        
        refined, score = local_search(start_set)
        if score > best_score:
            best_score = score
            best_set = refined

    return best_set
