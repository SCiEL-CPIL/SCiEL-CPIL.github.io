import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def lazebnik_woldar():
        """
        Lazebnik-Woldar type construction using GF(9) x Z_3.
        """
        def gf9_add(e1, e2):
            return ((e1[0]+e2[0])%3, (e1[1]+e2[1])%3)
        def gf9_sub(e1, e2):
            return ((e1[0]-e2[0])%3, (e1[1]-e2[1])%3)
        def gf9_mult(e1, e2):
            a1, b1 = e1
            a2, b2 = e2
            a = (a1*a2 + 2*b1*b2) % 3
            b = (a1*b2 + a2*b1) % 3
            return (a, b)
        
        gf9_elements = [(a,b) for a in range(3) for b in range(3)]
        nonzero = [e for e in gf9_elements if e != (0,0)]
        
        # Find generator and partition
        gen = (0,1)
        cur = gen
        order = 1
        while cur != (1,0) and order < 10:
            cur = gf9_mult(cur, gen)
            order += 1
        
        powers = [(1,0)]
        cur = (1,0)
        for i in range(order):
            cur = gf9_mult(cur, gen)
            powers.append(cur)
        
        set_a = set(powers[0::2])
        set_b = set(powers[1::2])
        
        n_full = 27
        vertices = [(e, c) for e in gf9_elements for c in range(3)]
        
        def get_color(v1, v2):
            e1, c1 = v1
            e2, c2 = v2
            d = gf9_sub(e1, e2)
            if d == (0,0):
                return (c1 - c2) % 3
            else:
                return 3 if d in set_a else 4
        
        colors = []
        for i in range(n_full):
            for j in range(i+1, n_full):
                colors.append(get_color(vertices[i], vertices[j]))
        
        def check_c4_free(n, col):
            neighbors = [[0]*n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = col[idx]
                    neighbors[c][i] |= (1 << j)
                    neighbors[c][j] |= (1 << i)
                    idx += 1
            for c in range(5):
                for u in range(n):
                    nu = neighbors[c][u]
                    if nu == 0: continue
                    for v in range(u+1, n):
                        nv = neighbors[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            return False
            return True
        
        if check_c4_free(n_full, colors):
            return {'n': 27, 'colors': colors}
        
        for rem in range(n_full):
            n = n_full - 1
            new_colors = []
            idx = 0
            for i in range(n_full):
                for j in range(i+1, n_full):
                    if i == rem or j == rem:
                        continue
                    new_colors.append(colors[idx])
                    idx += 1
            if check_c4_free(n, new_colors):
                return {'n': n, 'colors': new_colors}
        
        return None

    result = lazebnik_woldar()
    if result is not None:
        return result

    def count_violations(n, colors):
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        cost = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0: continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        cost += 1
        return cost

    def solve_local_search(n, time_limit, seed=None):
        if seed is not None:
            random.seed(seed)
        total_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        color_matrix = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                color_matrix[i][j] = c
                color_matrix[j][i] = c
                idx += 1
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        def get_cost():
            cost = 0
            for c in range(5):
                for u in range(n):
                    nu = neighbors[c][u]
                    if nu == 0: continue
                    for v in range(u + 1, n):
                        nv = neighbors[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            cost += 1
            return cost
        
        current_cost = get_cost()
        if current_cost == 0:
            return colors
        
        search_start = time.time()
        temperature = 1.0
        cooling_rate = 0.995
        stagnation = 0
        
        while time.time() - search_start < time_limit:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i,j = j,i
            
            c_old = color_matrix[i][j]
            best_nc = -1
            best_cost = current_cost
            
            for c_new in range(5):
                if c_new == c_old: continue
                delta = 0
                for w in range(n):
                    if w == i or w == j: continue
                    if neighbors[c_old][j] & (1 << w):
                        k = bin(neighbors[c_old][i] & neighbors[c_old][w]).count('1')
                        delta -= (k-1)
                    if neighbors[c_old][i] & (1 << w):
                        k = bin(neighbors[c_old][j] & neighbors[c_old][w]).count('1')
                        delta -= (k-1)
                for w in range(n):
                    if w == i or w == j: continue
                    if neighbors[c_new][j] & (1 << w):
                        k = bin(neighbors[c_new][i] & neighbors[c_new][w]).count('1')
                        delta += k
                    if neighbors[c_new][i] & (1 << w):
                        k = bin(neighbors[c_new][j] & neighbors[c_new][w]).count('1')
                        delta += k
                nc = current_cost + delta
                if nc < best_cost:
                    best_cost = nc
                    best_nc = c_new
            
            if best_nc != -1:
                neighbors[c_old][i] &= ~(1 << j)
                neighbors[c_old][j] &= ~(1 << i)
                neighbors[best_nc][i] |= (1 << j)
                neighbors[best_nc][j] |= (1 << i)
                color_matrix[i][j] = best_nc
                color_matrix[j][i] = best_nc
                idx = i*(2*n-i-1)//2 + j-i-1
                colors[idx] = best_nc
                current_cost = best_cost
                stagnation = 0
                if current_cost == 0:
                    return colors
            else:
                if random.random() < temperature:
                    ri = random.randint(0, n-1)
                    rj = random.randint(0, n-1)
                    if ri == rj: continue
                    if ri > rj: ri,rj = rj,ri
                    rc_old = color_matrix[ri][rj]
                    rc_new = random.randint(0,4)
                    if rc_new == rc_old: continue
                    neighbors[rc_old][ri] &= ~(1 << rj)
                    neighbors[rc_old][rj] &= ~(1 << ri)
                    neighbors[rc_new][ri] |= (1 << rj)
                    neighbors[rc_new][rj] |= (1 << ri)
                    color_matrix[ri][rj] = rc_new
                    color_matrix[rj][ri] = rc_new
                    ridx = ri*(2*n-ri-1)//2 + rj-ri-1
                    colors[ridx] = rc_new
                    current_cost = get_cost()
                    temperature *= cooling_rate
                    stagnation = 0
                else:
                    stagnation += 1
                    temperature *= cooling_rate
                    if stagnation > 200:
                        for _ in range(3):
                            ri = random.randint(0, n-1)
                            rj = random.randint(0, n-1)
                            if ri == rj: continue
                            if ri > rj: ri,rj = rj,ri
                            rc_old = color_matrix[ri][rj]
                            rc_new = random.randint(0,4)
                            if rc_new == rc_old: continue
                            neighbors[rc_old][ri] &= ~(1 << rj)
                            neighbors[rc_old][rj] &= ~(1 << ri)
                            neighbors[rc_new][ri] |= (1 << rj)
                            neighbors[rc_new][rj] |= (1 << ri)
                            color_matrix[ri][rj] = rc_new
                            color_matrix[rj][ri] = rc_new
                            ridx = ri*(2*n-ri-1)//2 + rj-ri-1
                            colors[ridx] = rc_new
                            current_cost = get_cost()
                            break
                        temperature = 0.5
                        stagnation = 0
        return None

    best_n = 2
    best_colors = [0]
    targets = [26, 25, 24, 23, 22, 21, 20, 18, 16, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    
    for n in targets:
        if time.time() > deadline - 3: break
        remaining = deadline - time.time()
        if remaining < 2: break
        num_restarts = max(1, int(remaining / 5))
        found = False
        for attempt in range(num_restarts):
            if time.time() > deadline - 1: break
            t_limit = min(4.0, deadline - time.time() - 0.5)
            if t_limit < 0.5: break
            sol = solve_local_search(n, t_limit, seed=42 + attempt * 7919)
            if sol is not None and count_violations(n, sol) == 0:
                best_n = n
                best_colors = sol
                found = True
                break
        if found: break

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
