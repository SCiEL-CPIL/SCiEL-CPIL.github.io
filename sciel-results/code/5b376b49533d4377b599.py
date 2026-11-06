import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    for target_n in range(10, 21):
        if time.time() > deadline - 10:
            break
        result = _sa_search(target_n, deadline - 5)
        if result is not None:
            best_n = target_n
            best_colors = result
    
    return {'n': best_n, 'colors': best_colors}

def _sa_search(n, deadline):
    random.seed(42)
    edges = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            edges[i][j] = random.randint(0, 3)
    
    def cost():
        total = 0
        for c in range(4):
            for i in range(n):
                for j in range(i+1, n):
                    if edges[i][j] != c:
                        continue
                    for a in range(n):
                        if a == i or a == j or edges[i][a] != c:
                            continue
                        for b in range(n):
                            if b == i or b == j or b == a or edges[a][b] != c:
                                continue
                            for d in range(n):
                                if d == i or d == j or d == a or d == b or edges[b][d] != c:
                                    continue
                                if edges[d][j] == c:
                                    total += 1
        return total
    
    current = cost()
    if current == 0:
        return [edges[i][j] for i in range(n) for j in range(i+1, n)]
    
    T = 1.0
    while T > 0.01 and time.time() < deadline:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        old = edges[i][j]
        new = random.randint(0, 3)
        if new == old:
            continue
        edges[i][j] = new
        new_cost = cost()
        if new_cost < current or random.random() < pow(2.718, (current - new_cost) / T):
            current = new_cost
            if current == 0:
                return [edges[i][j] for i in range(n) for j in range(i+1, n)]
        else:
            edges[i][j] = old
        T *= 0.999
    
    return None
