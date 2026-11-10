import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    random.seed(42)
    colors = list(range(NUM_EDGES))
    
    # Precompute all 3 cycles for each 4-vertex subset
    cycles_info = []
    for vertices in itertools.combinations(range(N), 4):
        a, b, c, d = vertices
        e1 = [edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a)]
        e2 = [edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a)]
        e3 = [edge_index(a, d), edge_index(d, c), edge_index(c, b), edge_index(b, a)]
        cycles_info.append(e1)
        cycles_info.append(e2)
        cycles_info.append(e3)
        
    # Precompute which cycles contain each edge
    cycles_containing_edge = [[] for _ in range(NUM_EDGES)]
    for i, cycle in enumerate(cycles_info):
        for e in cycle:
            cycles_containing_edge[e].append(i)

    def is_valid_global():
        for cycle in cycles_info:
            c1 = colors[cycle[0]]
            c2 = colors[cycle[1]]
            c3 = colors[cycle[2]]
            c4 = colors[cycle[3]]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True

    def is_valid_local(e_idx):
        for c_idx in cycles_containing_edge[e_idx]:
            cycle = cycles_info[c_idx]
            c1 = colors[cycle[0]]
            c2 = colors[cycle[1]]
            c3 = colors[cycle[2]]
            c4 = colors[cycle[3]]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True

    # Phase 1: Repair to a valid state
    max_iter = 10000
    for _ in range(max_iter):
        if is_valid_global():
            break
        # Find a violation
        for cycle in cycles_info:
            c1 = colors[cycle[0]]
            c2 = colors[cycle[1]]
            c3 = colors[cycle[2]]
            c4 = colors[cycle[3]]
            if len({c1, c2, c3, c4}) == 4:
                # Fix: change one edge to match another
                idx_to_change = random.randint(0, 3)
                idx_target = random.randint(0, 3)
                while idx_target == idx_to_change:
                    idx_target = random.randint(0, 3)
                colors[cycle[idx_to_change]] = colors[cycle[idx_target]]
                break

    # Phase 2: Recover distinct colors
    used_colors = set(colors)
    available_colors = list(set(range(120)) - used_colors)
    random.shuffle(available_colors)
    
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    for e_idx in edge_order:
        if not available_colors:
            break
        old_color = colors[e_idx]
        assigned = False
        # Try to assign an available color
        for i in range(len(available_colors)):
            c = available_colors[i]
            colors[e_idx] = c
            if is_valid_local(e_idx):
                used_colors.add(c)
                available_colors.pop(i)
                assigned = True
                break
        if not assigned:
            colors[e_idx] = old_color
            
    return colors
