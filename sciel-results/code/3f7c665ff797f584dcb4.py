import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    # Precompute all 4-term progressions
    progressions = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            progressions.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    def count_violations(colors):
        v = 0
        for (p1, p2, p3, p4) in progressions:
            if len({colors[p1], colors[p2], colors[p3], colors[p4]}) < 3:
                v += 1
        return v
    
    def local_search(num_colors, seed, max_iter=50000):
        rng = random.Random(seed)
        colors = [rng.randint(0, num_colors - 1) for _ in range(n)]
        best_colors = colors[:]
        best_v = count_violations(colors)
        
        if best_v == 0:
            return colors
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            pos = rng.randint(0, n - 1)
            old = colors[pos]
            new = rng.randint(0, num_colors - 1)
            if new == old:
                continue
            colors[pos] = new
            v = count_violations(colors)
            if v <= best_v:
                best_v = v
                best_colors = colors[:]
                if v == 0:
                    return colors
            else:
                colors[pos] = old
        
        return best_colors if count_violations(best_colors) == 0 else None
    
    for nc in [3, 4, 5]:
        for seed in range(20):
            if time.time() > deadline:
                break
            res = local_search(nc, seed, max_iter=20000)
            if res is not None:
                return res
    
    # Fallback
    return [i % 3 for i in range(n)]
