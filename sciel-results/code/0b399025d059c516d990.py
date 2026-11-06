import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    aps = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    ap_end = [[] for _ in range(n)]
    for (a, b, c, d) in aps:
        ap_end[d].append((a, b, c))
    
    def verify(colors):
        for (a, b, c, d) in aps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True
    
    # Try random with few colors
    for nc in [4, 5, 6]:
        for _ in range(5000):
            if time.time() > deadline:
                break
            colors = [random.randint(0, nc - 1) for _ in range(n)]
            if verify(colors):
                return colors
        if time.time() > deadline:
            break
    
    # Greedy with diverse initialization
    for nc in [4, 5, 6, 8]:
        if time.time() > deadline:
            break
        colors = [None] * n
        # Initialize first few positions with distinct colors
        for i in range(min(nc, 6)):
            colors[i] = i
        for x in range(6, n):
            placed = False
            for c in range(nc):
                ok = True
                for (a, b, c2) in ap_end[x]:
                    if len({c, colors[a], colors[b], colors[c2]}) < 3:
                        ok = False
                        break
                if ok:
                    colors[x] = c
                    placed = True
                    break
            if not placed:
                break
        if all(colors[i] is not None for i in range(n)) and verify(colors):
            return colors
    
    # Fallback: greedy with 128 colors
    colors = []
    for x in range(n):
        for c in range(128):
            ok = True
            for (a, b, c2) in ap_end[x]:
                if len({c, colors[a], colors[b], colors[c2]}) < 3:
                    ok = False
                    break
            if ok:
                colors.append(c)
                break
    return colors
