"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _build_nbrs(colors, n):
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            idx += 1
    return nbrs

def _has_c4(nbrs, n):
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                if len(nbrs[c][u] & nbrs[c][v]) >= 2:
                    return True
    return False

def construct():
    deadline = time.time() + 100.0
    random.seed(42)
    
    for target_n in range(29, 10, -1):
        if time.time() > deadline - 10:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        found = False
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            nbrs = _build_nbrs(colors, n)
            
            for step in range(10000):
                if time.time() > deadline - 2:
                    break
                
                if not _has_c4(nbrs, n):
                    found = True
                    break
                
                idx = random.randint(0, num_edges - 1)
                i, j = edges[idx]
                old_c = colors[idx]
                
                valid_moves = []
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    if len(nbrs[new_c][i] & nbrs[new_c][j]) < 2:
                        valid_moves.append(new_c)
                
                if valid_moves:
                    new_c = random.choice(valid_moves)
                    nbrs[old_c][i].discard(j)
                    nbrs[old_c][j].discard(i)
                    nbrs[new_c][i].add(j)
                    nbrs[new_c][j].add(i)
                    colors[idx] = new_c
            
            if found:
                break
        
        if found:
            return {'n': n, 'colors': colors}
    
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
# EVOLVE-BLOCK-END
