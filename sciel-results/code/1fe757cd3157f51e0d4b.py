import time
import random
import sys

def construct():
    deadline = time.time() + 115
    
    for n in [52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42]:
        if time.time() > deadline - 10:
            break
        result = _search_coloring(n, deadline - 10)
        if result is not None:
            return result
            
    return _fallback_42()

def _search_coloring(n, deadline):
    num_edges = n * (n - 1) // 2
    color_mat = [[0] * n for _ in range(n)]
    
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
    def set_color(i, j, c):
        if i > j: i, j = j, i
        old_c = color_mat[i][j]
        if old_c == c: return
        color_mat[i][j] = c
        color_mat[j][i] = c # Symmetric for convenience, though we only use i<j
        
        # Update bitsets
        adj[old_c][i] &= ~(1 << j)
        adj[old_c][j] &= ~(1 << i)
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
        
    adj = [[0] * n for _ in range(4)]
    
    def init_random():
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                color_mat[i][j] = c
                color_mat[j][i] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
    def find_triangle():
        # Try 50 random edges
        for _ in range(50):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            if i > j: i, j = j, i
            c = color_mat[i][j]
            common = adj[c][i] & adj[c][j]
            if common:
                b = common & (-common)
                w = b.bit_length() - 1
                return (i, j, w, c)
        return None

    def is_valid_move(i, j, new_c):
        if i > j: i, j = j, i
        common = adj[new_c][i] & adj[new_c][j]
        return common == 0

    for restart in range(100):
        if time.time() > deadline:
            return None
            
        init_random()
        
        stuck = 0
        for step in range(50000):
            if time.time() > deadline:
                return None
                
            tri = find_triangle()
            if tri is None:
                # Flatten
                out_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        out_colors.append(color_mat[i][j])
                return {'n': n, 'colors': out_colors}
                
            u, v, w, c = tri
            # Ensure u < v < w for edge indexing consistency? 
            # set_color handles ordering.
            
            edges = [(u,v), (v,w), (u,w)]
            random.shuffle(edges)
            
            moved = False
            for (i, j) in edges:
                if i > j: i, j = j, i
                old_c = color_mat[i][j]
                for new_c in range(4):
                    if new_c == old_c: continue
                    if is_valid_move(i, j, new_c):
                        set_color(i, j, new_c)
                        moved = True
                        break
                if moved: break
                
            if not moved:
                stuck += 1
                if stuck > 50:
                    break
            else:
                stuck = 0
    return None

def _fallback_42():
    p = 43
    inv = [0] * (p + 1)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    primitive = None
    for g in range(2, p):
        if pow(g, 21, p) != 1 and pow(g, 14, p) != 1 and pow(g, 6, p) != 1:
            primitive = g
            break
            
    powers = []
    current = 1
    for i in range(42):
        powers.append(current)
        current = (current * primitive) % p
        
    cubes = set(powers[3 * i] for i in range(14))
    
    coset = [0] * (p + 1)
    inv_prim = inv[primitive]
    for i in range(1, p):
        if i in cubes:
            coset[i] = 0
        else:
            e1 = (i * inv_prim) % p
            if e1 in cubes:
                coset[i] = 1
            else:
                coset[i] = 2
                
    colors = []
    for i in range(42):
        for j in range(i + 1, 42):
            ratio = ((i + 1) * inv[j + 1]) % p
            c = coset[ratio]
            colors.append(c)
            
    return {'n': 42, 'colors': colors}
