import time
import random

N = 16
TOTAL = N * N
DEADLINE = time.time() + 280.0

def _precompute():
    iso_with = [[0] * TOTAL for _ in range(TOTAL)]
    for p in range(TOTAL):
        px, py = p % N, p // N
        for a in range(TOTAL):
            if p == a: continue
            ax, ay = a % N, a // N
            d_pa_sq = (px - ax)**2 + (py - ay)**2
            mask = 0
            for b in range(TOTAL):
                if b == p or b == a: continue
                bx, by = b % N, b // N
                d_pb_sq = (px - bx)**2 + (py - by)**2
                d_ab_sq = (ax - bx)**2 + (ay - by)**2
                if d_pa_sq == d_pb_sq or d_pa_sq == d_ab_sq or d_pb_sq == d_ab_sq:
                    mask |= (1 << b)
            iso_with[p][a] = mask
    return iso_with

ISO_WITH = _precompute()

def _can_add(p, S_mask):
    s = S_mask
    while s:
        lsb = s & -s
        a = lsb.bit_length() - 1
        if (ISO_WITH[p][a] & S_mask):
            return False
        s ^= lsb
    return True

def _is_maximal(S_mask):
    for p in range(TOTAL):
        if not (S_mask & (1 << p)):
            if _can_add(p, S_mask):
                return False
    return True

def _shrink(S, S_mask):
    changed = True
    while changed:
        changed = False
        for i in range(len(S) - 1, -1, -1):
            p = S[i]
            new_S_mask = S_mask & ~(1 << p)
            if _is_maximal(new_S_mask):
                S.pop(i)
                S_mask = new_S_mask
                changed = True
    return S, S_mask

def _greedy_constrained(rng):
    S = []
    S_mask = 0
    while True:
        if time.time() > DEADLINE:
            break
        valid = []
        for p in range(TOTAL):
            if not (S_mask & (1 << p)):
                if _can_add(p, S_mask):
                    valid.append(p)
        if not valid:
            break
            
        # Score each valid point by how many it blocks
        best_p = None
        best_score = -1
        # Add random noise to break ties
        scores = {}
        for p in valid:
            new_S_mask = S_mask | (1 << p)
            blocks = 0
            for q in valid:
                if q == p: continue
                if not _can_add(q, new_S_mask):
                    blocks += 1
            # Add small random noise
            score = blocks + rng.random() * 0.1
            scores[p] = score
            if score > best_score:
                best_score = score
                best_p = p
                
        if best_p is None:
            break
            
        S.append(best_p)
        S_mask |= (1 << best_p)
        
    return S, S_mask

def construct():
    rng = random.Random(42)
    best_S = []
    best_len = float('inf')
    
    # Multi-start Constrained Greedy
    for i in range(50):
        if time.time() > DEADLINE - 30:
            break
        S, S_mask = _greedy_constrained(rng)
        S, S_mask = _shrink(S, S_mask)
        if len(S) < best_len:
            best_len = len(S)
            best_S = S[:]
            
    # Local Search
    if best_S:
        current_S = best_S[:]
        current_mask = 0
        for p in current_S:
            current_mask |= (1 << p)
        
        for iter in range(200):
            if time.time() > DEADLINE - 5:
                break
            improved = False
            for _ in range(20):
                if time.time() > DEADLINE - 5:
                    break
                if not current_S:
                    break
                r_idx = rng.randrange(len(current_S))
                r = current_S[r_idx]
                candidates = [p for p in range(TOTAL) if not (current_mask & (1 << p))]
                if not candidates:
                    continue
                p = rng.choice(candidates)
                new_S = [x for x in current_S if x != r] + [p]
                new_mask = current_mask & ~(1 << r) | (1 << p)
                if _is_maximal(new_mask):
                    new_S, new_mask = _shrink(new_S, new_mask)
                    if len(new_S) < len(current_S):
                        current_S = new_S
                        current_mask = new_mask
                        improved = True
                        if len(current_S) < best_len:
                            best_len = len(current_S)
                            best_S = current_S[:]
                        break
            if not improved:
                # Try removing 2, adding 2?
                pass

    return best_S
