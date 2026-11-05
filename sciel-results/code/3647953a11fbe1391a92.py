import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2

edge_map = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_map[(i, j)] = idx
        idx += 1

def get_edge(i, j):
    if i > j:
        i, j = j, i
    return edge_map[(i, j)]

edge_vertices = [None] * NUM_EDGES
for (i, j), idx in edge_map.items():
    edge_vertices[idx] = (i, j)

def is_valid_change(colors, edge_idx, new_color):
    if colors[edge_idx] == new_color:
        return True
        
    i, j = edge_vertices[edge_idx]
    
    for x in range(N):
        if x == i or x == j: continue
        for y in range(N):
            if y == i or y == j or y == x: continue
            
            # Cycle 1: i-j-x-y-i
            e_jx = get_edge(j, x)
            e_xy = get_edge(x, y)
            e_yi = get_edge(y, i)
            
            c1 = new_color
            c2 = colors[e_jx]
            c3 = colors[e_xy]
            c4 = colors[e_yi]
            
            if len({c1, c2, c3, c4}) == 4:
                return False
            
            # Cycle 2: i-j-y-x-i
            e_jy = get_edge(j, y)
            e_yx = get_edge(y, x)
            e_xi = get_edge(x, i)
            
            c1 = new_color
            c2 = colors[e_jy]
            c3 = colors[e_yx]
            c4 = colors[e_xi]
            
            if len({c1, c2, c3, c4}) == 4:
                return False
                
    return True

def count_violations(colors):
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                for l in range(k + 1, N):
                    # Cycle 1
                    e1 = get_edge(i, j)
                    e2 = get_edge(j, k)
                    e3 = get_edge(k, l)
                    e4 = get_edge(l, i)
                    if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
                        count += 1
                    # Cycle 2
                    e1 = get_edge(i, j)
                    e2 = get_edge(j, l)
                    e3 = get_edge(l, k)
                    e4 = get_edge(k, i)
                    if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
                        count += 1
                    # Cycle 3
                    e1 = get_edge(i, k)
                    e2 = get_edge(k, j)
                    e3 = get_edge(j, l)
                    e4 = get_edge(l, i)
                    if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
                        count += 1
    return count

def construct():
    random.seed(42)
    start_time = time.time()
    
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = get_edge(i, j)
            colors[idx] = i ^ j
            
    # Fix violations
    while time.time() - start_time < 50:
        if count_violations(colors) == 0:
            break
        
        # Find one violation
        found = False
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    for l in range(k + 1, N):
                        # Cycle 1
                        e1 = get_edge(i, j)
                        e2 = get_edge(j, k)
                        e3 = get_edge(k, l)
                        e4 = get_edge(l, i)
                        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
                            # Fix by matching e1 to e2
                            if is_valid_change(colors, e1, colors[e2]):
                                colors[e1] = colors[e2]
                                found = True
                                break
                        # Cycle 2
                        e1 = get_edge(i, j)
                        e2 = get_edge(j, l)
                        e3 = get_edge(l, k)
                        e4 = get_edge(k, i)
                        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
                            if is_valid_change(colors, e1, colors[e2]):
                                colors[e1] = colors[e2]
                                found = True
                                break
                        # Cycle 3
                        e1 = get_edge(i, k)
                        e2 = get_edge(k, j)
                        e3 = get_edge(j, l)
                        e4 = get_edge(l, i)
                        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
                            if is_valid_change(colors, e1, colors[e2]):
                                colors[e1] = colors[e2]
                                found = True
                                break
                    if found: break
                if found: break
            if found: break
            
        if not found:
            # If no simple fix works, try random changes
            # This is a fallback
            for _ in range(100):
                e = random.randint(0, NUM_EDGES - 1)
                new_c = random.randint(0, 119)
                if is_valid_change(colors, e, new_c):
                    colors[e] = new_c
                    break

    # Maximize colors
    used_colors = set(colors)
    for _ in range(20000):
        if time.time() - start_time > 55: break
        e = random.randint(0, NUM_EDGES - 1)
        if random.random() < 0.5:
            available = [c for c in range(120) if c not in used_colors]
            if available:
                new_c = random.choice(available)
            else:
                new_c = random.randint(0, 119)
        else:
            new_c = random.randint(0, 119)
            
        if new_c == colors[e]: continue
        if is_valid_change(colors, e, new_c):
            colors[e] = new_c
            used_colors.add(new_c)
            
    return colors
