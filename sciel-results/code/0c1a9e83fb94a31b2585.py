N = 16

def construct():
    import time
    import random
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i*(i+1)//2 + (j - i - 1)
    
    # Precompute cycles and edge-to-cycle mapping
    cycles = []
    for a, b, c, d in combinations(range(N), 4):
        cycles.append([edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)])
        cycles.append([edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)])
        cycles.append([edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)])
    
    # Map each edge to list of cycle indices containing it
    edge_to_cycles = [[] for _ in range(len(cycles[0])*0 + N*(N-1)//2)]
    for ci, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    
    # Start with min-vertex coloring
    colors = [0] * (N*(N-1)//2)
    for i in range(N):
        for j in range(i+1, N):
            colors[edge_idx(i,j)] = i
    
    def count_colors(c):
        return len(set(c))
    
    def check_edge_valid(e, c):
        """Check if changing edge e makes any cycle invalid."""
        for ci in edge_to_cycles[e]:
            cyc = cycles[ci]
            cs = set(c[x] for x in cyc)
            if len(cs) == 4:
                return False
        return True
    
    best = colors[:]
    best_score = count_colors(best)
    
    random.seed(42)
    
    # Track which colors are used
    used_colors = set(best)
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        e = random.randint(0, N*(N-1)//2 - 1)
        old = colors[e]
        
        # Prefer unused colors to increase count
        if len(used_colors) < 120:
            unused = [c for c in range(120) if c not in used_colors]
            new_color = random.choice(unused)
        else:
            new_color = random.randint(0, 119)
        
        if new_color == old:
            continue
        
        colors[e] = new_color
        if check_edge_valid(e, colors):
            if new_color not in used_colors:
                used_colors.add(new_color)
            if old not in colors:  # check if old color still used elsewhere
                # Need to verify - this is O(n), but n=120
                pass
            # Recalculate used colors is expensive; just track best
            new_score = len(set(colors))
            if new_score > best_score:
                best = colors[:]
                best_score = new_score
                used_colors = set(best)
        else:
            colors[e] = old
    
    return best
