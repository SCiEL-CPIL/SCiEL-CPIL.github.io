import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    n = 128
    
    aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            e = a + 3 * d
            if e > n:
                break
            aps.append((a-1, a+d-1, a+2*d-1, e-1))
    
    def count_viol(col):
        v = 0
        for a,b,c,e in aps:
            if len({col[a],col[b],col[c],col[e]}) < 3:
                v += 1
        return v
    
    colors = []
    for x in range(1, n+1):
        for c in range(128):
            ok = True
            for a,b,cc,e in aps:
                if e == x-1:
                    if len({c, colors[a], colors[b], colors[cc]}) < 3:
                        ok = False
                        break
            if ok:
                colors.append(c)
                break
    
    mapping = {}
    nid = 0
    for c in colors:
        if c not in mapping:
            mapping[c] = nid
            nid += 1
    colors = [mapping[c] for c in colors]
    nc = nid
    
    best = colors[:]
    best_v = count_viol(colors)
    
    random.seed(42)
    
    for it in range(200000):
        if time.time() > deadline:
            break
        i = random.randint(0, n-1)
        old = colors[i]
        new_c = random.randint(0, nc-1)
        if new_c == old:
            continue
        colors[i] = new_c
        v = count_viol(colors)
        if v < best_v:
            best_v = v
            best = colors[:]
        if v > best_v:
            colors[i] = old
        if best_v == 0:
            break
    
    if best_v == 0 and time.time() < deadline:
        colors = best[:]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            used = sorted(set(colors))
            for i, c1 in enumerate(used):
                if time.time() > deadline:
                    break
                for c2 in used[i+1:]:
                    test = [c1 if c == c2 else c for c in colors]
                    if count_viol(test) == 0:
                        colors = test
                        changed = True
                        break
                if changed:
                    break
        
        mapping2 = {}
        nid2 = 0
        for c in sorted(set(colors)):
            mapping2[c] = nid2
            nid2 += 1
        return [mapping2[c] for c in colors]
    
    if best_v == 0:
        mapping2 = {}
        nid2 = 0
        for c in sorted(set(best)):
            mapping2[c] = nid2
            nid2 += 1
        return [mapping2[c] for c in best]
    return best
