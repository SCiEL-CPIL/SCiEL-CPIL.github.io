import time
import random
from itertools import combinations

DEADLINE = 145.0
_start_time = time.time()

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _calc_beta(H, n=18):
    forbidden = []
    for u in range(n):
        for v in range(u+1, n):
            if H[u][v]:
                continue
            for w in range(v+1, n):
                if H[u][w] and H[v][w]:
                    forbidden.append((u, v, w))
                    
    if not forbidden:
        return n
        
    best = [0]
    
    def _search(chosen, remaining):
        if len(chosen) + len(remaining) <= best[0]:
            return
            
        if not remaining:
            if len(chosen) > best[0]:
                best[0] = len(chosen)
            return
            
        v = remaining[0]
        new_remaining = remaining[1:]
        
        can_include = True
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                u, w = chosen[i], chosen[j]
                if not H[v][u] and not H[v][w] and not H[u][w]:
                    can_include = False
                    break
            if not can_include:
                break
                
        if can_include:
            new_chosen = chosen + [v]
            _search(new_chosen, new_remaining)
            
        _search(chosen, new_remaining)
        
    _search([], list(range(n)))
    return best[0]

def construct():
    n = 18
    H = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if i // 6 == j // 6:
                H[i][j] = True
                H[j][i] = True
                
    best_beta = _calc_beta(H, n)
    best_H = [row[:] for row in H]
    
    rng = random.Random(42)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while _time_left() > 5.0:
        if no_improve_count > max_no_improve:
            # Restart with a random valid H
            # Generate random H with p=0.5, then fix alpha
            H = [[False]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < 0.5:
                        H[i][j] = True
                        H[j][i] = True
            # Ensure alpha <= 3
            # Simple loop: if alpha > 3, add random edge
            while True:
                # Check alpha
                has_4 = False
                for a in range(n):
                    for b in range(a+1, n):
                        if H[a][b]: continue
                        for c in range(b+1, n):
                            if H[a][c] or H[b][c]: continue
                            for d in range(c+1, n):
                                if H[a][d] or H[b][d] or H[c][d]:
                                    continue
                                has_4 = True
                                break
                            if has_4: break
                        if has_4: break
                    if has_4: break
                if not has_4:
                    break
                # Add random edge
                u = rng.randint(0, n-1)
                v = rng.randint(0, n-1)
                if u != v and not H[u][v]:
                    H[u][v] = True
                    H[v][u] = True
                    
            current_beta = _calc_beta(H, n)
            if current_beta < best_beta:
                best_beta = current_beta
                best_H = [row[:] for row in H]
            no_improve_count = 0
            max_no_improve = 500
            continue
            
        u = rng.randint(0, n-1)
        v = rng.randint(0, n-1)
        if u == v or not H[u][v]:
            continue
            
        H[u][v] = False
        H[v][u] = False
        
        valid = True
        candidates = []
        for w in range(n):
            if w == u or w == v: continue
            if not H[u][w] and not H[v][w]:
                candidates.append(w)
                
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                w, x = candidates[i], candidates[j]
                if not H[w][x]:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            new_beta = _calc_beta(H, n)
            if new_beta < best_beta:
                best_beta = new_beta
                best_H = [row[:] for row in H]
                no_improve_count = 0
                if best_beta <= 8:
                    break
            else:
                H[u][v] = True
                H[v][u] = True
                no_improve_count += 1
        else:
            H[u][v] = True
            H[v][u] = True
            no_improve_count += 1
            
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not best_H[i][j]:
                edges.append([i, j])
                
    return edges
