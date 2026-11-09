import random
import time
import math

N = 16
EDGES = []
for i in range(N):
    for j in range(i+1, N):
        EDGES.append((i, j))

edge_index = {}
for idx, (u, v) in enumerate(EDGES):
    edge_index[(u, v)] = idx
    edge_index[(v, u)] = idx

constraints = [[] for _ in range(len(EDGES))]
for idx, (u, v) in enumerate(EDGES):
    others = [k for k in range(N) if k != u and k != v]
    for x_idx in range(len(others)):
        for y_idx in range(x_idx + 1, len(others)):
            x = others[x_idx]
            y = others[y_idx]
            e1 = edge_index[(u, x)]
            e2 = edge_index[(x, y)]
            e3 = edge_index[(y, v)]
            constraints[idx].append((e1, e2, e3))
            
            e4 = edge_index[(u, y)]
            e5 = edge_index[(y, x)]
            e6 = edge_index[(x, v)]
            constraints[idx].append((e4, e5, e6))

def is_move_valid(colors, edge_idx, new_color):
    for a, b, c in constraints[edge_idx]:
        ca = colors[a]
        cb = colors[b]
        cc = colors[c]
        if ca != cb and cb != cc and ca != cc:
            if new_color != ca and new_color != cb and new_color != cc:
                return False
    return True

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Initialize with a valid 3-coloring
    current_colors = [random.randint(0, 2) for _ in range(len(EDGES))]
    
    # Build frequency array
    freq = [0] * 120
    for c in current_colors:
        freq[c] += 1
    current_count = sum(1 for f in freq if f > 0)
    
    best_colors = current_colors[:]
    best_count = current_count
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = 10.0
            current_colors = best_colors[:]
            freq = [0] * 120
            for c in current_colors:
                freq[c] += 1
            current_count = sum(1 for f in freq if f > 0)
            
        e_idx = random.randint(0, len(EDGES)-1)
        new_c = random.randint(0, 119)
        
        if is_move_valid(current_colors, e_idx, new_c):
            old_c = current_colors[e_idx]
            if old_c == new_c:
                continue
                
            # Update freq and count
            freq[old_c] -= 1
            freq[new_c] += 1
            
            delta = 0
            if freq[old_c] == 0:
                delta -= 1
            if freq[new_c] == 1:
                delta += 1
                
            new_count = current_count + delta
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_colors[e_idx] = new_c
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_colors = current_colors[:]
            else:
                # Revert
                current_colors[e_idx] = old_c
                freq[old_c] += 1
                freq[new_c] -= 1
                # current_count reverts implicitly as we didn't update it
                # But we need to revert freq correctly
                # The revert logic above restores freq.
                # current_count is still the old value.
        
        T *= cooling_rate

    return best_colors
