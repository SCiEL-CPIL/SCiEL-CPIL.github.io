"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_mean_of_subset(a, others):
    if len(others) < 2:
        return False
    deltas = [y - a for y in others]
    reachable = {0: 0}
    for d in deltas:
        updates = {}
        for s, cnt in reachable.items():
            ns = s + d
            nc = cnt + 1
            if ns not in updates or nc < updates[ns]:
                updates[ns] = nc
        for ns, nc in updates.items():
            if ns not in reachable or nc < reachable[ns]:
                reachable[ns] = nc
            if ns == 0 and nc >= 2:
                return True
    return reachable.get(0, 0) >= 2

def construct():
    start = time.time()
    deadline = start + 80
    best_set = []
    
    for trial in range(50):
        if time.time() > deadline:
            break
        S = []
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S:
                continue
            S_new = S + [x]
            valid = True
            for a in S_new:
                others = [y for y in S_new if y != a]
                if _is_mean_of_subset(a, others):
                    valid = False
                    break
            if valid:
                S = S_new
        
        if len(S) > len(best_set):
            best_set = S[:]
    
    return best_set
# EVOLVE-BLOCK-END
