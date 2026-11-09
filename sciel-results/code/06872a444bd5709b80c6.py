import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 108.0
    MARGIN = 6.0

    def time_left():
        return DEADLINE - (time.time() - t0)

    def verify(n, colors):
        adj = [[0]*n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]; idx += 1
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[c][u] & adj[c][v]).bit_count() >= 2:
                        return False
        return True

    best_n = 0
    best_colors = None

    def extract(color, n):
        out = []
        for i in range(n):
            for j in range(i+1, n):
                out.append(color[i][j])
        return out

    def sa(n, budget):
        nonlocal best_n, best_colors
        pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
        total = len(pairs)
        restarts = 0
        solved = False
        while time_left() > MARGIN and not solved:
            restarts += 1
            color = [[0]*n for _ in range(n)]
            for (i,j) in pairs:
                color[i][j] = random.randrange(5)
            adj = [[0]*n for _ in range(5)]
            for (i,j) in pairs:
                c = color[i][j]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
            common = [[[0]*n for _ in range(n)] for _ in range(5)]
            viol = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u+1, n):
                        cc = (adj[c][u] & adj[c][v]).bit_count()
                        common[c][u][v] = cc
                        common[c][v][u] = cc
                        if cc >= 2:
                            viol += 1
            if viol == 0:
                cols = extract(color, n)
                if verify(n, cols):
                    best_n = n; best_colors = cols
                    solved = True
                    break
            T = 1.0
            T_end = 0.005
            # steps this restart
            while time_left() > MARGIN:
                a,b = pairs[random.randrange(total)]
                c1 = color[a][b]
                c2 = random.randrange(5)
                if c2 == c1:
                    continue
                # remove from c1
                delta = 0
                # remove edge (a,b) color c1
                Na = adj[c1][a]; Nb = adj[c1][b]
                x = (Nb & ~(1<<a))
                while x:
                    lsb = x & (-x); xi = lsb.bit_length()-1; x ^= lsb
                    old = common[c1][a][xi]
                    new = old - 1
                    common[c1][a][xi] = new; common[c1][xi][a] = new
                    if old == 2: delta -= 1
                x = (Na & ~(1<<b))
                while x:
                    lsb = x & (-x); xi = lsb.bit_length()-1; x ^= lsb
                    old = common[c1][b][xi]
                    new = old - 1
                    common[c1][b][xi] = new; common[c1][xi][b] = new
                    if old == 2: delta -= 1
                adj[c1][a] &= ~(1<<b); adj[c1][b] &= ~(1<<a)
                # add edge (a,b) color c2
                Na = adj[c2][a]; Nb = adj[c2][b]
                x = (Nb & ~(1<<a))
                while x:
                    lsb = x & (-x); xi = lsb.bit_length()-1; x ^= lsb
                    old = common[c2][a][xi]
                    new = old + 1
                    common[c2][a][xi] = new; common[c2][xi][a] = new
                    if old == 1: delta += 1
                x = (Na & ~(1<<b))
                while x:
                    lsb = x & (-x); xi = lsb.bit_length()-1; x ^= lsb
                    old = common[c2][b][xi]
                    new = old + 1
                    common[c2][b][xi] = new; common[c2][xi][b] = new
                    if old == 1: delta += 1
                adj[c2][a] |= (1<<b); adj[c2][b] |= (1<<a)
                viol += delta
                # accept?
                if delta <= 0 or random.random() < (2.718281828 ** (-delta / T)):
                    color[a][b] = c2
                else:
                    # revert: remove from c2, add to c1
                    # remove edge (a,b) color c2
                    Na = adj[c2][a]; Nb = adj[c2][b]
                    x = (Nb & ~(1<<a))
                    while x:
                        lsb = x & (-x); xi = lsb.bit_length()-1; x ^= lsb
                        old = common[c2][a][xi]; new = old-1
                        common[c2][a][xi]=new; common[c2][xi][a]=new
                        if old==2: viol -= 1
                    x = (Na & ~(1<<b))
                    while x:
                        lsb = x & (-x); xi = lsb.bit_length()-1; x ^= lsb
                        old = common[c2][b][xi]; new = old-1
                        common[c2][b][xi]=new; common[c2][xi][b]=new
                        if old==2: viol -= 1
                    adj[c2][a] &= ~(1<<b); adj[c2][b] &= ~(1<<a)
                    # add edge (a,b) color c1
                    Na = adj[c1][a]; Nb = adj[c1][b]
                    x = (Nb & ~(1<<a))
                    while x:
                        lsb = x & (-x); xi = lsb.bit_length()-1; x ^= lsb
                        old = common[c1][a][xi]; new = old+1
                        common[c1][a][xi]=new; common[c1][xi][a]=new
                        if old==1: viol += 1
                    x = (Na & ~(1<<b))
                    while x:
                        lsb = x & (-x); xi = lsb.bit_length()-1; x ^= lsb
                        old = common[c1][b][xi]; new = old+1
                        common[c1][b][xi]=new; common[c1][xi][b]=new
                        if old==1: viol += 1
                    adj[c1][a] |= (1<<b); adj[c1][b] |= (1<<a)
                # cooling
                T = max(T_end, T * 0.9999)
                if viol == 0:
                    cols = extract(color, n)
                    if verify(n, cols):
                        best_n = n; best_colors = cols
                        solved = True
                        break
        return solved

    # attempt increasing n
    for target in [26, 27, 28]:
        if time_left() <= MARGIN + 1:
            break
        sa(target, None)
        if best_n >= target:
            continue
        # if failed, maybe still try next with less time
    if best_n == 0 or best_colors is None:
        # fallback seed
        n = 11
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    return {'n': best_n, 'colors': best_colors}
