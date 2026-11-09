"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

# Build edge index for quick lookup
EDGE_SET = set()
for u, v in EDGES:
    if u < v:
        EDGE_SET.add((u, v))
    else:
        EDGE_SET.add((v, u))

def has_bicolored_cycle(coloring, k):
    """Check if any pair of colors induces a cycle using union-find."""
    # For each pair of colors, collect edges where both endpoints have those colors
    color_vertices = [[] for _ in range(k)]
    for i, c in enumerate(coloring):
        if c >= 0:
            color_vertices[c].append(i)
    
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            if not color_vertices[c1] or not color_vertices[c2]:
                continue
            # Union-find for vertices of color c1 or c2
            parent = list(range(N))
            rank = [0] * N
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # cycle detected
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            # Check edges between vertices of color c1 and c2
            v1_set = set(color_vertices[c1])
            v2_set = set(color_vertices[c2])
            for u in color_vertices[c1]:
                for v in ADJ[u]:
                    if v in v2_set:
                        if not union(u, v):
                            return True
            # Also check edges within same color? No - bicolored means exactly two colors
            # But we also need to check edges where both endpoints are in {c1, c2}
            # Actually the induced subgraph on {c1, c2} includes all edges between vertices
            # that have color c1 or c2. Let me redo this properly.
            
            # Reset
            parent = list(range(N))
            rank = [0] * N
            def find2(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union2(x, y):
                rx, ry = find2(x), find2(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            # All edges where both endpoints have color c1 or c2
            for u, v in EDGES:
                cu, cv = coloring[u], coloring[v]
                if cu == c1 or cu == c2:
                    if cv == c1 or cv == c2:
                        if not union2(u, v):
                            return True
    return False

def solve_k_colors(k, deadline):
    """Try to find a valid coloring with at most k colors using backtracking."""
    coloring = [-1] * N
    used_colors = set()
    
    # Order vertices by degree (highest first) for better pruning
    order = sorted(range(N), key=lambda x: len(ADJ[x]), reverse=True)
    # Actually, let's use a fixed order but with MRV (minimum remaining values)
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        
        v = order[idx]
        # Determine available colors
        neighbor_colors = set()
        for nb in ADJ[v]:
            if coloring[nb] >= 0:
                neighbor_colors.add(coloring[nb])
        
        for c in range(k):
            if c in neighbor_colors:
                continue
            coloring[v] = c
            used_colors.add(c)
            
            # Quick check: does adding this vertex create a bicolored cycle?
            # Only check pairs involving color c
            valid = True
            for c2 in used_colors:
                if c2 == c:
                    continue
                # Check if edge (v, nb) where nb has color c2 creates a cycle
                # in the subgraph induced by colors {c, c2}
                # We need to check if v and nb are already connected in the 
                # {c, c2}-induced subgraph
                # For efficiency, just do a full check on affected pairs
                pass  # Full check is expensive; do it at the end or with incremental check
            
            # Incremental cycle check for pairs involving c
            valid = True
            for c2 in used_colors:
                if c2 == c:
                    continue
                # Check if adding v (color c) creates a cycle in {c, c2} subgraph
                # Find all neighbors of v with color c2
                for nb in ADJ[v]:
                    if coloring[nb] == c2:
                        # Check if nb is connected to v in the {c,c2} subgraph 
                        # (excluding the edge v-nb we're adding)
                        # BFS/DFS from nb in {c,c2} subgraph, see if we reach v
                        visited = set()
                        stack = [nb]
                        visited.add(nb)
                        while stack:
                            cur = stack.pop()
                            for nb2 in ADJ[cur]:
                                if nb2 == v:
                                    # Found path from nb to v not using direct edge? 
                                    # Actually if we reach v, there's a cycle
                                    valid = False
                                    break
                                if nb2 in visited:
                                    continue
                                if coloring[nb2] == c or coloring[nb2] == c2:
                                    visited.add(nb2)
                                    stack.append(nb2)
                            if not valid:
                                break
                        if not valid:
                            break
                if not valid:
                    break
            
            if valid:
                if backtrack(idx + 1):
                    return True
            
            coloring[v] = -1
            used_colors.discard(c)
        
        return False
    
    if backtrack(0):
        return coloring
    return None

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Try k=3, 4, 5, ...
    for k in range(3, 50):
        if time.time() > deadline:
            break
        result = solve_k_colors(k, deadline)
        if result is not None:
            # Verify
            used = set(result)
            # Check proper coloring
            for u, v in EDGES:
                if result[u] == result[v]:
                    break
            else:
                # Check bicolored forest
                if not has_bicolored_cycle(result, k):
                    return [int(c) for c in result]
    
    # Fallback: use 50 colors (always valid)
    return list(range(N))

# EVOLVE-BLOCK-END
