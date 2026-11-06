import numpy as np
import random
import time

def construct():
    n = 26
    colors = 5
    timeout = 115.0
    
    def calc_energy(col):
        e = 0
        for c in range(colors):
            for u in range(n):
                for v in range(u + 1, n):
                    common = 0
                    for w in range(n):
                        if w != u and w != v:
                            if col[u][w] == c and col[w][v] == c:
                                common += 1
                    if common > 1:
                        e += (common - 1) ** 2
        return e

    def get_colors_list(col):
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                res.append(col[i][j])
        return res

    # Initialize
    col = [[random.randint(0, colors - 1) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            col[j][i] = col[i][j]
            
    current_energy = calc_energy(col)
    
    T = 50.0
    T_min = 0.5
    cooling = 0.9995
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if current_energy == 0:
            return {'n': n, 'colors': get_colors_list(col)}
            
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        old_c = col[u][v]
        new_c = random.randint(0, colors - 1)
        if new_c == old_c:
            continue
            
        col[u][v] = new_c
        col[v][u] = new_c
        
        new_energy = calc_energy(col)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_energy = new_energy
        else:
            col[u][v] = old_c
            col[v][u] = old_c
            
        T *= cooling
        if T < T_min:
            T = T_min

    # If not solved, return n=1 to be safe? Or the best found?
    # The task requires a valid coloring.
    # If we didn't find one, we can't return an invalid one.
    # However, for n=26, it is almost certain to be found.
    # If we fail, we return n=1.
    return {'n': 1, 'colors': []}
