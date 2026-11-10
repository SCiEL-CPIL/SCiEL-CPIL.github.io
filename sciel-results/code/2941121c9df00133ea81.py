import time
import random

def _max_sidon_size(arr):
    n = len(arr)
    best = 0
    arr_sorted = sorted(arr)
    
    def bt(start, cur, sums):
        nonlocal best
        if len(cur) > best:
            best = len(cur)
        if start >= n:
            return
        # Upper bound: can we add all remaining?
        if len(cur) + (n - start) <= best:
            return
        for i in range(start, n):
            v = arr_sorted[i]
            ok = True
            if 2*v in sums:
                ok = False
            else:
                for j in range(len(cur)):
                    if v + cur[j] in sums:
                        ok = False
                        break
            if ok:
                new_sums = sums | {2*v}
                for j in range(len(cur)):
                    new_sums.add(v + cur[j])
                bt(i+1, cur+[v], new_sums)
    
    bt(0, [], set())
    return best

def construct():
    t0 = time.time()
    dl = t0 + 70
    
    best = list(range(12))
    bs = _max_sidon_size(best)
    
    # Try APs
    for d in range(1, 12):
        if 11*d > 128: break
        c = list(range(0, 12*d, d))
        s = _max_sidon_size(c)
        if s < bs:
            bs = s; best = c
    
    # Random search
    random.seed(42)
    for _ in range(50000):
        if time.time() > dl: break
        c = sorted(random.sample(range(129), 12))
        s = _max_sidon_size(c)
        if s < bs:
            bs = s; best = c
    
    # Local search
    if time.time() < dl - 5:
        cur = list(best)
        cs = bs
        while time.time() < dl - 2:
            imp = False
            for i in range(12):
                if time.time() > dl - 2: break
                for _ in range(20):
                    v = random.randint(0, 128)
                    if v in cur: continue
                    new = cur[:]; new[i] = v; new.sort()
                    s = _max_sidon_size(new)
                    if s < cs:
                        cur = new; cs = s; best = list(cur); bs = s; imp = True; break
                    elif s == cs and random.random() < 0.05:
                        cur = new; break
                if imp: break
            if not imp:
                # Perturb
                for _ in range(5):
                    i = random.randint(0, 11)
                    cur[i] = random.randint(0, 128)
                    cur.sort()
    
    return best
