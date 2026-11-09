import itertools
import time
import random

N = 16
M = N * N
_DEADLINE = time.time() + 280

def _precompute_iso_sets():
    points = [(x, y) for y in range(N) for x in range(N)]
    dist2 = [[0]*M for _ in range(M)]
    for i in range(M):
        for j in range(i+1, M):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d = dx*dx + dy*dy
            dist2[i][j] = d
            dist2[j][i] = d

    iso_set = [[set() for _ in range(M)] for __ in range(M)]
    
    for i in range(M):
        for j in range(i+1, M):
            d_ij = dist2[i][j]
            if d_ij == 0: continue
            for k in range(M):
                if k == i or k == j: continue
                d_ik = dist2[i][k]
                d_jk = dist2[j][k]
                if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                    iso_set[i][j].add(k)
                    iso_set[j][i].add(k)
    return iso_set, points

def construct():
    iso_set, points = _precompute_iso_sets()
    all_indices = list(range(M))
    best_S = None
    best_size = float('inf')
    
    for restart in range(2000):
        if time.time() > _DEADLINE:
            break
            
        S = []
        unblocked = set(all_indices)
        compatible_qs = set(all_indices)
        
        while unblocked:
            if time.time() > _DEADLINE:
                break
                
            if not S:
                q = random.choice(list(unblocked))
                S.append(q)
                unblocked.remove(q)
                compatible_qs.remove(q)
                continue
            
            if len(S) == 1:
                candidates = list(unblocked)
                if not candidates: break
                q = random.choice(candidates)
                S.append(q)
                unblocked.remove(q)
                compatible_qs.remove(q)
                
                a, b = S[0], S[1]
                # Update unblocked
                for p in iso_set[a][b]:
                    if p in unblocked:
                        unblocked.discard(p)
                # Update compatible_qs
                # Remove q' that form iso with (a, b)
                for qp in iso_set[a][b]:
                    if qp in compatible_qs:
                        compatible_qs.discard(qp)
                continue
            
            # len(S) >= 2
            p = random.choice(list(unblocked))
            
            valid_qs = []
            for q in compatible_qs:
                if q == p:
                    continue
                # Check if q helps block p
                helps = False
                for a in S:
                    if p in iso_set[a][q]:
                        helps = True
                        break
                if not helps:
                    continue
                
                # Score q
                score = 0
                for p_prime in unblocked:
                    if p_prime == p: continue
                    blocked = False
                    for a in S:
                        if p_prime in iso_set[a][q]:
                            blocked = True
                            break
                    if blocked:
                        score += 1
                valid_qs.append((score, q))
            
            if not valid_qs:
                break
            
            valid_qs.sort(reverse=True)
            best_score, q = valid_qs[0]
            
            S.append(q)
            unblocked.discard(q)
            compatible_qs.discard(q)
            
            # Update unblocked and compatible_qs
            for a in S[:-1]:
                # Points blocked by (a, q)
                for p in iso_set[a][q]:
                    if p in unblocked:
                        unblocked.discard(p)
                    if p in compatible_qs:
                        compatible_qs.discard(p)
        
        if not unblocked:
            if best_S is None or len(S) < len(best_S):
                best_S = S.copy()
                best_size = len(S)
                if best_size < 10:
                    break

    if best_S is None:
        # Fallback
        S = []
        for p in all_indices:
            ok = True
            for a in S:
                for b in S:
                    if a == b: continue
                    if p in iso_set[a][b]:
                        ok = False
                        break
                if not ok: break
            if ok:
                S.append(p)
        return S

    return best_S
