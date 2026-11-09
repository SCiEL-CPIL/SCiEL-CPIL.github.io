import time
import random

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    if n < 2:
        return True
    adj = [[0] * n for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    for c in range(5):
        for u in range(n):
            mask_u = adj[c][u]
            for v in range(u + 1, n):
                common = mask_u & adj[c][v]
                if common and (common & (common - 1)):
                    return False
    return True

def _lz_woldar_coloring(n=26):
    """
    Construct the Lazebnik-Woldar 5-coloring of K_26.
    """
    def add(p, q):
        return ((p[0] + q[0]) % 5, (p[1] + q[1]) % 5)
    
    def sub(p, q):
        return ((p[0] - q[0]) % 5, (p[1] - q[1]) % 5)
    
    def mul(p, q):
        a, b = p
        c, d = q
        return ((a*c - 2*b*d) % 5, (a*d + b*c) % 5)
    
    def order(p):
        cur = (1, 0)
        for k in range(1, 25):
            cur = mul(cur, p)
            if cur == (1, 0):
                return k
        return -1
    
    generator = None
    for a in range(5):
        for b in range(5):
            if (a, b) == (0, 0):
                continue
            if order((a, b)) == 24:
                generator = (a, b)
                break
        if generator:
            break
    
    if not generator:
        return [0] * (26*25//2)
    
    g4 = mul(mul(generator, generator), mul(generator, generator))
    H = set()
    cur = (1, 0)
    for _ in range(6):
        H.add(cur)
        cur = mul(cur, g4)
    
    cosets = [set() for _ in range(4)]
    for k in range(4):
        gk = (1, 0)
        for _ in range(k):
            gk = mul(gk, generator)
        for h in H:
            cosets[k].add(mul(gk, h))
    
    elem_to_color = {}
    for c in range(4):
        for e in cosets[c]:
            elem_to_color[e] = c
    
    elements = []
    for a in range(5):
        for b in range(5):
            elements.append((a, b))
    
    colors = []
    for i in range(26):
        for j in range(i + 1, 26):
            if i == 25 or j == 25:
                colors.append(4)
            else:
                e_i = elements[i]
                e_j = elements[j]
                diff = sub(e_j, e_i)
                c = elem_to_color.get(diff, 0)
                colors.append(c)
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    lw_colors = _lz_woldar_coloring(26)
    if not _is_valid_coloring(26, lw_colors):
        return {'n': 1, 'colors': []}
    
    best_n = 26
    best_colors = lw_colors
    
    # Build matrix representation
    def build_matrix(colors, n):
        cm = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                cm[i][j] = colors[idx]
                cm[j][i] = colors[idx]
                idx += 1
        return cm
    
    def flatten(cm, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(cm[i][j])
        return colors
    
    def build_structures(cm, n):
        adj = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i][j]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        conflict = [[0] * n for _ in range(5)]
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                conf_mask = 0
                m = mask_u
                while m:
                    lsb = m & (-m)
                    w_idx = lsb.bit_length() - 1
                    conf_mask |= adj[c][w_idx]
                    m ^= lsb
                conflict[c][u] = conf_mask
        
        return adj, conflict
    
    def try_extend(cm, n, adj, conflict, max_attempts=10000):
        new_n = n + 1
        new_cm = [row[:] + [-1] for row in cm]
        new_cm.append([-1] * new_n)
        
        new_v = new_n - 1
        assigned = [-1] * n
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None, False
                
            order = list(range(n))
            random.shuffle(order)
            
            assigned_colors = [0] * 5
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    for i in range(n):
                        c = assigned[i]
                        new_cm[new_v][i] = c
                        new_cm[i][new_v] = c
                    return True
                
                u = order[idx]
                colors_to_try = [0, 1, 2, 3, 4]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    if assigned_colors[c] & conflict[c][u] == 0:
                        assigned[u] = c
                        assigned_colors[c] |= (1 << u)
                        if backtrack(idx + 1):
                            return True
                        assigned[u] = -1
                        assigned_colors[c] &= ~(1 << u)
                
                return False
            
            if backtrack(0):
                return new_cm, True
        
        return None, False
    
    cm = build_matrix(lw_colors, 26)
    n = 26
    adj, conflict = build_structures(cm, n)
    
    # Try to extend from n=26 to n=29
    while n < 29:
        if time.time() > deadline:
            break
            
        attempts_for_step = 15000
        new_cm, success = try_extend(cm, n, adj, conflict, attempts_for_step)
        
        if success:
            n += 1
            cm = new_cm
            adj, conflict = build_structures(cm, n)
            if n > best_n:
                best_n = n
                best_colors = flatten(cm, n)
        else:
            break

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Max n found: {res['n']}")
    if res['n'] > 1:
        print(f"Valid: {_is_valid_coloring(res['n'], res['colors'])}")
