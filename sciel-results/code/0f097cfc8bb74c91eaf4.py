import numpy as np
import time
import random

def _build_mask(adj, color):
    n = adj.shape[0]
    mask = [0] * n
    for v in range(n):
        m = 0
        for u in range(n):
            if u != v and adj[v][u] == color:
                m |= (1 << u)
        mask[v] = m
    return mask

def _has_k222_fast(adj):
    """Check if adj has a monochromatic K_{2,2,2}. Returns True/False."""
    n = adj.shape[0]
    for color in [0, 1]:
        mask = _build_mask(adj, color)
        for a in range(n):
            for b in range(a+1, n):
                common_ab = mask[a] & mask[b]
                common_ab &= ~(1 << a) & ~(1 << b)
                verts = [v for v in range(n) if (common_ab >> v) & 1]
                for ci in range(len(verts)):
                    for di in range(ci+1, len(verts)):
                        c = verts[ci]
                        d = verts[di]
                        common_cd = mask[c] & mask[d]
                        candidates = common_ab & common_cd
                        candidates &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                        if bin(candidates).count('1') >= 2:
                            return True
    return False

def _energy_fast(adj):
    n = adj.shape[0]
    energy = 0
    for color in [0, 1]:
        mask = _build_mask(adj, color)
        for a in range(n):
            for b in range(a+1, n):
                common = mask[a] & mask[b]
                common &= ~(1 << a) & ~(1 << b)
                cnt = bin(common).count('1')
                if cnt >= 2:
                    energy += cnt * (cnt - 1) // 2
    return energy

def _colors_from_adj(adj):
    n = adj.shape[0]
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(int(adj[i][j]))
    return colors

def _paley_29():
    p = 29
    squares = set()
    for x in range(1, p):
        squares.add((x*x) % p)
    colors = []
    for i in range(p):
        for j in range(i+1, p):
            diff = (j-i) % p
            colors.append(0 if diff in squares else 1)
    return colors

def construct():
    start = time.time()
    deadline = start + 140
    
    n = 30
    base = _paley_29()
    adj29 = np.zeros((29,29), dtype=np.int8)
    idx = 0
    for i in range(29):
        for j in range(i+1, 29):
            adj29[i][j] = base[idx]
            adj29[j][i] = base[idx]
            idx += 1
    
    best_adj = None
    best_energy = float('inf')
    
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize
        adj = np.zeros((n, n), dtype=np.int8)
        adj[:29, :29] = adj29
        adj[29, :29] = np.random.randint(0, 2, 29)
        adj[:29, 29] = adj[29, :29]
        
        cur_e = _energy_fast(adj)
        if cur_e < best_energy:
            best_energy = cur_e
            best_adj = adj.copy()
        
        if not _has_k222_fast(adj):
            return {'n': n, 'colors': _colors_from_adj(adj)}
        
        # SA
        T = 3.0
        T_min = 0.05
        cooling = 0.9998
        
        while time.time() < deadline:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old = adj[i][j]
            adj[i][j] = 1 - old
            adj[j][i] = 1 - old
            
            new_e = _energy_fast(adj)
            delta = new_e - cur_e
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                cur_e = new_e
                if cur_e < best_energy:
                    best_energy = cur_e
                    best_adj = adj.copy()
                    if cur_e == 0:
                        if not _has_k222_fast(adj):
                            return {'n': n, 'colors': _colors_from_adj(adj)}
            else:
                adj[i][j] = old
                adj[j][i] = old
            
            T = max(T * cooling, T_min)
            
            # Periodic exact check
            if cur_e < 5 and random.random() < 0.05:
                if not _has_k222_fast(adj):
                    return {'n': n, 'colors': _colors_from_adj(adj)}
    
    # Final check on best
    if best_adj is not None and not _has_k222_fast(best_adj):
        return {'n': n, 'colors': _colors_from_adj(best_adj)}
    
    return {'n': 29, 'colors': base}
