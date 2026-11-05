import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # Precomputation
    edge_map = {}
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            idx = len(edges)
            edges.append((i, j))
            edge_map[(i, j)] = idx
            edge_map[(j, i)] = idx

    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # 3 cycles
                    cycles.append((edge_map[(a, b)], edge_map[(b, c)], edge_map[(c, d)], edge_map[(d, a)]))
                    cycles.append((edge_map[(a, b)], edge_map[(b, d)], edge_map[(d, c)], edge_map[(c, a)]))
                    cycles.append((edge_map[(a, c)], edge_map[(c, b)], edge_map[(b, d)], edge_map[(d, a)]))

    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    def is_rainbow(colors, cyc):
        c1 = colors[cyc[0]]
        c2 = colors[cyc[1]]
        c3 = colors[cyc[2]]
        c4 = colors[cyc[3]]
        return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

    # Initialize
    colors = [random.randint(0, 2) for _ in range(NUM_EDGES)]
    best_colors = colors[:]
    best_num_colors = len(set(best_colors))
    
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    while time.time() < deadline:
        e_idx = random.choice(edge_order)
        new_color = random.randint(0, 100)
        
        if new_color == colors[e_idx]:
            continue
            
        old_color = colors[e_idx]
        colors[e_idx] = new_color
        
        valid = True
        violating_cycle = None
        for cyc_idx in edge_to_cycles[e_idx]:
            if is_rainbow(colors, cycles[cyc_idx]):
                valid = False
                violating_cycle = cycles[cyc_idx]
                break
        
        if valid:
            num_used = len(set(colors))
            if num_used > best_num_colors:
                best_num_colors = num_used
                best_colors = colors[:]
        else:
            # Try to fix
            other_edges = [e for e in violating_cycle if e != e_idx]
            fixed = False
            for e_fix in other_edges:
                target_colors = set()
                for e in violating_cycle:
                    if e != e_fix:
                        target_colors.add(colors[e])
                
                for tc in target_colors:
                    old_fix = colors[e_fix]
                    colors[e_fix] = tc
                    
                    ok = True
                    for cyc_idx2 in edge_to_cycles[e_fix]:
                        if is_rainbow(colors, cycles[cyc_idx2]):
                            ok = False
                            break
                    
                    if ok:
                        fixed = True
                        break
                    else:
                        colors[e_fix] = old_fix
                
                if fixed:
                    break
            
            if fixed:
                num_used = len(set(colors))
                if num_used > best_num_colors:
                    best_num_colors = num_used
                    best_colors = colors[:]
            else:
                colors[e_idx] = old_color

    return best_colors
