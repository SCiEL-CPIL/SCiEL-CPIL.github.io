"""Few colors without any bicolored cycle - local search for 3-coloring."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
M = len(EDGES)

# Build adjacency
ADJ = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGES):
    ADJ[u].append((v, i))
    ADJ[v].append((u, i))

def _has_bicolored_cycle_full(colors):
    """Check all color pairs for bicolored cycles using union-find."""
    used = set()
    for u in range(N):
        for v, ei in ADJ[u]:
            if u < v and colors[u] != colors[v]:
                used.add((colors[u], colors[v]))
    for (ca, cb) in used:
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u in range(N):
            cu = colors[u]
            if cu not in (ca, cb):
                continue
            for v, ei in ADJ[u]:
                if u < v and colors[v] in (ca, cb):
                    if not union(u, v):
                        return True
    return False

def _count_bicolored_cycles(colors):
    """Count number of bicolored cycles (0 means valid)."""
    used = set()
    for u in range(N):
        for v, ei in ADJ[u]:
            if u < v and colors[u] != colors[v]:
                used.add((colors[u], colors[v]))
    total = 0
    for (ca, cb) in used:
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        cnt = 0
        for u in range(N):
            cu = colors[u]
            if cu not in (ca, cb):
                continue
            for v, ei in ADJ[u]:
                if u < v and colors[v] in (ca, cb):
                    if not union(u, v):
                        cnt += 1
        total += cnt
    return total

def _greedy_init():
    """Greedy coloring with 3 colors, trying to avoid bicolored cycles."""
    colors = [None] * N
    for u in range(N):
        # Try colors 0,1,2 in order
        for c in range(3):
            # Check adjacency constraint
            ok = True
            for v, ei in ADJ[u]:
                if colors[v] is not None and colors[v] == c:
                    ok = False
                    break
            if not ok:
                continue
            # Tentatively assign and check if it creates a bicolored cycle
            colors[u] = c
            # Quick check: for each color pair involving c, see if adding u creates a cycle
            # Only need to check pairs (c, other) where other is a neighbor's color
            bad = False
            for other in range(3):
                if other == c:
                    continue
                # Check if in the subgraph of colors {c, other}, adding u's edges creates a cycle
                # u connects to neighbors with color 'other'
                # If two such neighbors are already connected in the {c,other} subgraph, it's a cycle
                neighbors_other = [v for v, ei in ADJ[u] if colors[v] == other]
                if len(neighbors_other) >= 2:
                    # Check if any two are connected via a path in {c, other}
                    # BFS from first
                    start = neighbors_other[0]
                    visited = set()
                    stack = [start]
                    visited.add(start)
                    for target in neighbors_other[1:]:
                        if target in visited:
                            bad = True
                            break
                        if stack:
                            stack.append(target)
                            visited.add(target)
                            # Actually just check if target reachable from start in {c,other} subgraph
                            # Simpler: BFS
                    # Let me do proper BFS
                    visited = set([start])
                    queue = [start]
                    found = False
                    while queue and not found:
                        node = queue.pop(0)
                        for nv, _ in ADJ[node]:
                            if colors[nv] in (c, other) and nv not in visited:
                                if nv in neighbors_other[1:]:
                                    found = True
                                    break
                                visited.add(nv)
                                queue.append(nv)
                    if found:
                        bad = True
                        break
            if not bad:
                return colors[u] if False else None  # placeholder
            colors[u] = None
    # Fallback: simple greedy
    colors = [0] * N
    for u in range(1, N):
        used_colors = set(colors[v] for v, _ in ADJ[u] if v < u)
        for c in range(3):
            if c not in used_colors:
                colors[u] = c
                break
    return colors

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    # Start with a valid 3-coloring (greedy)
    colors = [0] * N
    for u in range(1, N):
        used_colors = set()
        for v, _ in ADJ[u]:
            if v < u:
                used_colors.add(colors[v])
        for c in range(3):
            if c not in used_colors:
                colors[u] = c
                break
    
    # Check if already valid
    if _count_bicolored_cycles(colors) == 0:
        return [int(c) for c in colors]
    
    best = colors[:]
    best_cyc = _count_bicolored_cycles(best)
    
    rng = random.Random(42)
    
    # Local search: hill climbing with random restarts
    while time.time() < deadline:
        # Try to reduce cycle count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Random order of vertices
            order = list(range(N))
            rng.shuffle(order)
            for u in order:
                old_c = colors[u]
                for new_c in range(3):
                    if new_c == old_c:
                        continue
                    # Check adjacency
                    ok = True
                    for v, _ in ADJ[u]:
                        if colors[v] == new_c:
                            ok = False
                            break
                    if not ok:
                        continue
                    colors[u] = new_c
                    cyc = _count_bicolored_cycles(colors)
                    if cyc < best_cyc:
                        best_cyc = cyc
                        best = colors[:]
                        improved = True
                        if cyc == 0:
                            return [int(c) for c in colors]
                        break
                    colors[u] = old_c
                # If no improvement at u, try a swap with a neighbor
                for v, _ in ADJ[u]:
                    if colors[u] == colors[v]:
                        continue
                    # Swap colors of u and v
                    cu, cv = colors[u], colors[v]
                    colors[u], colors[v] = cv, cu
                    cyc = _count_bicolored_cycles(colors)
                    if cyc < best_cyc:
                        best_cyc = cyc
                        best = colors[:]
                        improved = True
                        if cyc == 0:
                            return [int(c) for c in colors]
                        break
                    colors[u], colors[v] = cu, cv
                    if improved:
                        break
                if improved:
                    break
        
        # If stuck, random restart
        if time.time() >= deadline:
            break
        # Perturb: random reassignment of a few vertices
        for _ in range(5):
            u = rng.randint(0, N - 1)
            c = rng.randint(0, 2)
            # Only if it doesn't violate adjacency
            ok = True
            for v, _ in ADJ[u]:
                if colors[v] == c:
                    ok = False
                    break
            if ok:
                colors[u] = c
        # Reset to best if perturbation made things worse
        if _count_bicolored_cycles(colors) > best_cyc:
            colors = best[:]
    
    # Return best found
    if best_cyc == 0:
        return [int(c) for c in best]
    
    # If still not valid, try more aggressive search
    # Try all 3^k assignments for small subsets? No, too expensive.
    # Instead, try systematic repair
    while time.time() < deadline:
        # Find a bicolored cycle and try to break it
        if _count_bicolored_cycles(colors) == 0:
            return [int(c) for c in colors]
        
        # Find which color pair has a cycle
        used = set()
        for u in range(N):
            for v, ei in ADJ[u]:
                if u < v and colors[u] != colors[v]:
                    used.add((colors[u], colors[v]))
        
        broken = False
        for (ca, cb) in used:
            # Find a cycle in {ca, cb} subgraph
            parent = list(range(N))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True
            
            cycle_edge = None
            for u in range(N):
                cu = colors[u]
                if cu not in (ca, cb):
                    continue
                for v, ei in ADJ[u]:
                    if u < v and colors[v] in (ca, cb):
                        if not union(u, v):
                            cycle_edge = (u, v)
                            break
                if cycle_edge:
                    break
            
            if cycle_edge:
                u, v = cycle_edge
                # Try to recolor u or v to break the cycle
                for target in [u, v]:
                    for new_c in range(3):
                        if new_c == colors[target]:
                            continue
                        ok = True
                        for nb, _ in ADJ[target]:
                            if colors[nb] == new_c:
                                ok = False
                                break
                        if not ok:
                            continue
                        old = colors[target]
                        colors[target] = new_c
                        if _count_bicolored_cycles(colors) < _count_bicolored_cycles(best):
                            best = colors[:]
                            broken = True
                            break
                        colors[target] = old
                    if broken:
                        break
            if broken:
                break
        
        if not broken:
            # Random perturbation
            for _ in range(3):
                u = rng.randint(0, N - 1)
                c = rng.randint(0, 2)
                ok = True
                for v, _ in ADJ[u]:
                    if colors[v] == c:
                        ok = False
                        break
                if ok:
                    colors[u] = c
    
    # Final check
    if _count_bicolored_cycles(best) == 0:
        return [int(c) for c in best]
    
    # If we couldn't find a valid 3-coloring, return best effort
    # But we need validity... try 4 colors as fallback
    # Actually, let's try a simple valid coloring with more colors
    colors = [0] * N
    for u in range(1, N):
        used = set(colors[v] for v, _ in ADJ[u] if v < u)
        for c in range(4):
            if c not in used:
                colors[u] = c
                break
    
    # Verify and fix if needed
    if _count_bicolored_cycles(colors) == 0:
        return [int(c) for c in colors]
    
    # Fallback: use the original 50-color assignment (always valid)
    return list(range(N))
# EVOLVE-BLOCK-END
