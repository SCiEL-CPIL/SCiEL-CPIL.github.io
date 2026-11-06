import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    def get_seed_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        seed_colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                seed_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return 32, seed_colors

    best_n, best_colors = get_seed_32()

    def try_sa_v2(target_n):
        nonlocal best_n, best_colors
        if time.time() - start_time > time_limit:
            return False
            
        n = target_n
        colors = [[random.randint(0, 3) for _ in range(n)] for _ in range(n)]
        
        def calc_cost():
            c = 0
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        if colors[i][j] == colors[i][k] and colors[i][j] == colors[j][k]:
                            c += 1
            return c

        current_cost = calc_cost()
        
        if current_cost == 0:
            res_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    res_colors.append(colors[i][j])
            best_n = n
            best_colors = res_colors
            return True

        T = 1.0
        cooling_rate = 0.99995
        
        def get_delta(i, j, c_old, c_new):
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if colors[i][k] == c_old and colors[j][k] == c_old:
                    delta -= 1
                if colors[i][k] == c_new and colors[j][k] == c_new:
                    delta += 1
            return delta

        while time.time() - start_time < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            c_old = colors[i][j]
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
                
            delta = get_delta(i, j, c_old, c_new)
            
            if delta <= 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                colors[i][j] = c_new
                colors[j][i] = c_new
                current_cost += delta
                if current_cost == 0:
                    res_colors = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            res_colors.append(colors[u][v])
                    best_n = n
                    best_colors = res_colors
                    return True
                T *= cooling_rate
                if T < 0.001:
                    T = 1.0

        return False

    try_sa_v2(50)

    return {'n': best_n, 'colors': best_colors}
