import time
from itertools import combinations

def _can_add(x, S):
    n = len(S)
    # Check 1: x as divisor of any triple sum in S
    for i in range(n):
        si = S[i]
        for j in range(i+1, n):
            sij = si + S[j]
            for k in range(j+1, n):
                if (sij + S[k]) % x == 0:
                    return False
    # Check 2: for each a in S, a must not divide x+b+c for any b,c in S\{a}
    for i in range(n):
        a = S[i]
        for j in range(n):
            if j == i:
                continue
            sj = S[j]
            for k in range(j+1, n):
                if k == i:
                    continue
                if (x + sj + S[k]) % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 80
    
    # Phase 1: Greedy descending
    S = []
    for x in range(192, 0, -1):
        if _can_add(x, S):
            S.append(x)
    best = S[:]
    
    if time.time() > deadline:
        return best
    
    # Phase 2: Local search (single swap)
    all_nums = set(range(1, 193))
    selected = set(best)
    non_sel = sorted(all_nums - selected, reverse=True)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best)):
            old = best[i]
            remaining = best[:i] + best[i+1:]
            for new_x in non_sel:
                if new_x in selected:
                    continue
                if _can_add(new_x, remaining):
                    best = remaining + [new_x]
                    selected.add(new_x)
                    selected.discard(old)
                    non_sel = sorted(all_nums - selected, reverse=True)
                    improved = True
                    break
            if improved:
                break
    
    return best
