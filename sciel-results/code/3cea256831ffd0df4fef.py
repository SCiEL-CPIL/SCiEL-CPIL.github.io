import time
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def check_no_c6(n, colors):
        adj = [[0]*n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i][j] = 1
                adj[c][j][i] = 1
                idx += 1
        for c in range(4):
            a = adj[c]
            for verts in combinations(range(n), 6):
                v0 = verts[0]
                remaining = verts[1:]
                for perm in permutations(remaining):
                    v1, v2, v3, v4, v5 = perm
                    if (a[v0][v1] and a[v1][v2] and a[v2][v3] and 
                        a[v3][v4] and a[v4][v5] and a[v5][v0]):
                        return False
        return True
    
    # Try n=17 with various constructions
    n = 17
    constructions = [
        lambda i,j: (i + 2*j) % 4,
        lambda i,j: (i + j) % 4,
        lambda i,j: (2*i + j) % 4,
        lambda i,j: (i * j) % 4,
        lambda i,j: (i + j + i*j) % 4,
    ]
    
    for f in constructions:
        if time.time() > deadline:
            break
        colors = [f(i,j) for i in range(n) for j in range(i+1, n)]
        if check_no_c6(n, colors):
            return {'n': n, 'colors': colors}
    
    # Fallback: n=9
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
