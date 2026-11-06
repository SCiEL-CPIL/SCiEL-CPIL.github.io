import time

def construct():
    start = time.time()
    deadline = start + 130.0

    # 8 disjoint K4s: n=32, C5-free, alpha=8
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])

    # Verify C5-free: each connected component is K4, which has no C5
    # Verify alpha: each K4 has alpha=1, so total alpha = 8 < 9
    
    return {'n': n, 'edges': edges}
