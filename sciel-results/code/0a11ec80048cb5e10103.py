import time
import random

N = 256
DEADLINE = 85.0

# Precomputation done inside construct to ensure isolation, 
# but since N is constant, we can do it at module level if we want.
# However, the prompt asks to keep function names. 
# I will define helpers inside construct or use module-level constants if they don't conflict.
# Given the strict interface, I'll keep everything inside construct or use module-level vars if safe.
# Module-level is faster for re-calls, but construct is called once.
# I'll use module-level for precomputed data to keep construct clean.

# Precompute 4-AP conflicts
# conf_4ap[x] = list of masks (bitsets of 3 elements)
# If mask & conflict_mask == conflict_mask, then adding x creates a 4-AP.
_CONF_4AP = [None] * (N + 1)
for x in range(1, N + 1):
    _CONF_4AP[x] = []
    # 1st term
    for d in range(1, (N - x) // 3 + 1):
        others = [x + d, x + 2 * d, x + 3 * d]
        m = 0
        for y in others: m |= (1 << (y - 1))
        _CONF_4AP[x].append(m)
    # 2nd term
    for d in range(1, x):
        if x + 2 * d > N: break
        others = [x - d, x + d, x + 2 * d]
        m = 0
        for y in others: m |= (1 << (y - 1))
        _CONF_4AP[x].append(m)
    # 3rd term
    for d in range(1, x // 2 + 1):
        if x + d > N: break
        others = [x - 2 * d, x - d, x + d]
        m = 0
        for y in others: m |= (1 << (y - 1))
        _CONF_4AP[x].append(m)
    # 4th term
    for d in range(1, x // 3 + 1):
        others = [x - 3 * d, x - 2 * d, x - d]
        m = 0
        for y in others: m |= (1 << (y - 1))
        _CONF_4AP[x].append(m)

# Precompute 3-AP pairs
# _PAIRS_3AP[x] = list of (a, b) such that {x, a, b} is a 3-AP
_PAIRS_3AP = [None] * (N + 1)
for x in range(1, N + 1):
    _PAIRS_3AP[x] = []
    # x middle
    for d in range(1, min(x, N - x) + 1):
        _PAIRS_3AP[x].append((x - d, x + d))
    # x end (larger)
    for d in range(1, (N - x) // 2 + 1):
        _PAIRS_3AP[x].append((x + d, x + 2 * d))
    # x end (smaller)
    for d in range(1, x // 2 + 1):
        _PAIRS_3AP[x].append((x - 2 * d, x - d))

def construct():
    start_time = time.time()
    random.seed(12345)
    
    def is_4ap_free(mask, x):
        for cm in _CONF_4AP[x]:
            if (mask & cm) == cm:
                return False
        return True

    def delta_3aps(mask, x):
        cnt = 0
        for a, b in _PAIRS_3AP[x]:
            if ((mask >> (a - 1)) & 1) and ((mask >> (b - 1)) & 1):
                cnt += 1
        return cnt

    def count_3aps(mask):
        total = 0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                total += delta_3aps(mask & ~(1 << (x - 1)), x)
        return total // 3

    # Greedy Seed
    mask = 0
    for _ in range(N):
        best_x = -1
        best_d = -1
        candidates = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]
        if not candidates: break
        random.shuffle(candidates)
        for x in candidates:
            if not is_4ap_free(mask, x): continue
            d = delta_3aps(mask, x)
            if d > best_d:
                best_d = d
                best_x = x
        if best_x == -1: break
        mask |= (1 << (best_x - 1))
        
    best_mask = mask
    best_score = count_3aps(mask)
    
    # Local Search
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        present = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        missing = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]
        
        if not present or not missing:
            break
            
        random.shuffle(present)
        random.shuffle(missing)
        
        for u in present:
            if (time.time() - start_time) > DEADLINE: break
            cost_u = delta_3aps(mask & ~(1 << (u - 1)), u)
            
            for v in missing:
                if (time.time() - start_time) > DEADLINE: break
                trial_mask = mask & ~(1 << (u - 1))
                if not is_4ap_free(trial_mask, v): continue
                gain_v = delta_3aps(trial_mask, v)
                if gain_v - cost_u > 0:
                    mask = trial_mask | (1 << (v - 1))
                    score = count_3aps(mask)
                    if score > best_score:
                        best_score = score
                        best_mask = mask
                    improved = True
                    break
        
        if not improved:
            # Perturbation
            if len(present) >= 2:
                u, w = random.sample(present, 2)
                mask &= ~(1 << (u - 1))
                mask &= ~(1 << (w - 1))
            else:
                break

    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return sorted(result)
