import time
import random

_start_time = time.time()
DEADLINE = _start_time + 85

N = 12
M = 40
TOTAL = 1 << N  # 4096

def construct():
    # Precompute comparability matrix
    # comp[i][j] is 1 if i and j are comparable (one is strict subset of the other)
    comp = [bytearray(TOTAL) for _ in range(TOTAL)]
    for i in range(TOTAL):
        for j in range(TOTAL):
            if i != j and (i | j) == i or (i | j) == j:
                comp[i][j] = 1
                comp[j][i] = 1

    def greedy_init():
        """Initialize with a greedy approach."""
        fam = []
        seen = set()
        # Start with empty and full set
        fam.append(0)
        fam.append(TOTAL - 1)
        seen.update(fam)
        
        for _ in range(M - 2):
            best_m = -1
            best_score = -1
            for m in range(TOTAL):
                if m in seen:
                    continue
                s = 0
                for f in fam:
                    s += comp[m][f]
                if s > best_score:
                    best_score = s
                    best_m = m
            if best_m == -1:
                while True:
                    m = random.randrange(TOTAL)
                    if m not in seen:
                        best_m = m
                        break
            fam.append(best_m)
            seen.add(best_m)
        return fam

    def compute_degrees(fam):
        """Compute degree for all masks relative to the family."""
        deg = [0] * TOTAL
        for m in fam:
            c_m = comp[m]
            for k in range(TOTAL):
                deg[k] += c_m[k]
        return deg

    def local_search(fam, deg, max_time):
        fam = list(fam)
        deg = list(deg)
        seen = set(fam)
        
        no_improve_count = 0
        while time.time() < max_time:
            if no_improve_count > 500:
                break
            
            i = random.randrange(M)
            old_m = fam[i]
            
            best_gain = 0
            best_new_m = -1
            
            old_deg_val = deg[old_m]
            
            for m in range(TOTAL):
                if m == old_m or m in seen:
                    continue
                gain = (deg[m] - comp[m][old_m]) - old_deg_val
                if gain > best_gain:
                    best_gain = gain
                    best_new_m = m
            
            if best_new_m != -1 and best_gain > 0:
                seen.remove(old_m)
                seen.add(best_new_m)
                fam[i] = best_new_m
                
                c_new = comp[best_new_m]
                c_old = comp[old_m]
                for k in range(TOTAL):
                    deg[k] += c_new[k] - c_old[k]
                
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return fam, deg

    best_family = None
    best_score = -1
    
    # Initial greedy
    fam = greedy_init()
    deg = compute_degrees(fam)
    
    score = 0
    for i in range(M):
        score += deg[fam[i]]
    score //= 2
    
    best_family = list(fam)
    best_score = score
    
    while time.time() < DEADLINE - 5:
        fam, deg = local_search(best_family, deg, DEADLINE - 2)
        
        score = 0
        for i in range(M):
            score += deg[fam[i]]
        score //= 2
        
        if score > best_score:
            best_score = score
            best_family = list(fam)
        
        # Random restart to escape local optima
        if random.random() < 0.05:
             fam = greedy_init()
             deg = compute_degrees(fam)
             score = 0
             for i in range(M):
                 score += deg[fam[i]]
             score //= 2
             if score > best_score:
                 best_score = score
                 best_family = list(fam)

    result = []
    for mask in best_family:
        lst = [int(i) for i in range(N) if (mask >> i) & 1]
        result.append(lst)
        
    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i + 1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1 - t0:.2f}s, Comparables: {cnt}")
