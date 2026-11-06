import time
import random

N = 16
M = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    edge_index = {}
    edges = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edges.append((i, j))
            idx += 1
            
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    e1 = [edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)]]
                    e2 = [edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)]]
                    e3 = [edge_index[(a,d)], edge_index[(d,c)], edge_index[(c,b)], edge_index[(b,a)]]
                    cycles.append(e1)
                    cycles.append(e2)
                    cycles.append(e3)
                    
    edge_to_cycles = [[] for _ in range(M)]
    for i, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(i)
            
    colors = [0] * M
    used_colors = {0}
    used_colors_list = [0]
    
    def check_edge_change(e, new_color):
        for c_idx in edge_to_cycles[e]:
            cycle = cycles[c_idx]
            new_colors = []
            for ce in cycle:
                if ce == e:
                    new_colors.append(new_color)
                else:
                    new_colors.append(colors[ce])
            if len(set(new_colors)) == 4:
                return False
        return True

    def can_add_new_color(e):
        intersection = set(range(120))
        has_constraint = False
        
        for c_idx in edge_to_cycles[e]:
            cycle = cycles[c_idx]
            S = set()
            for ce in cycle:
                if ce != e:
                    S.add(colors[ce])
            if len(S) == 3:
                intersection &= S
                has_constraint = True
            elif len(S) > 3:
                return None
                
        if has_constraint:
            candidates = intersection - used_colors
            if candidates:
                return next(iter(candidates))
            else:
                return None
        else:
            candidates = set(range(120)) - used_colors
            if candidates:
                return next(iter(candidates))
            else:
                return None

    edge_order = list(range(M))
    
    while time.time() < deadline:
        progress = False
        random.shuffle(edge_order)
        for e in edge_order:
            c = can_add_new_color(e)
            if c is not None:
                colors[e] = c
                used_colors.add(c)
                used_colors_list.append(c)
                progress = True
        if not progress:
            relaxed = False
            for e in edge_order:
                if len(used_colors_list) < 2: continue
                c = random.choice(used_colors_list)
                if c == colors[e]:
                    continue
                if check_edge_change(e, c):
                    colors[e] = c
                    relaxed = True
            if not relaxed:
                break
    return colors
