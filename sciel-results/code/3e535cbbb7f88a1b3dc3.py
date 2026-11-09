import time
import random

N = 16

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i*(i+1)//2 + (j - i - 1)

def construct():
    deadline = time.time() + 55
    n_edges = N*(N-1)//2
    colors = [0] * n_edges
    
    # 1. Initialize with a valid Gallai coloring (15 colors)
    # Recursive bipartition
    def color_gallai(vertices, color_offset):
        if len(vertices) <= 1:
            return color_offset
        mid = len(vertices) // 2
        part1 = vertices[:mid]
        part2 = vertices[mid:]
        
        # Color internal edges of part1
        c1 = color_gallai(part1, color_offset)
        c2 = color_gallai(part2, c1) # Note: c1 is the next available offset? No.
        
        # Let's do it iteratively or properly recursively
        # A standard Gallai coloring of K_n uses n-1 colors.
        # We can just hardcode a simple structure for 16 vertices.
        # Split 16 into 8 and 8.
        # Color edges between 0..7 and 8..15 with color 14.
        # Color edges within 0..7 with colors 0..6.
        # Color edges within 8..15 with colors 7..13.
        
        pass

    # Simple deterministic initialization
    # Split into two sets A={0..7}, B={8..15}
    # Edges in A: use colors 0..6 (Gallai on 8)
    # Edges in B: use colors 7..13 (Gallai on 8)
    # Edges A-B: use color 14
    
    def init_gallai_8(vertices, base_color):
        # Recursive for 8 vertices
        if len(vertices) == 1:
            return base_color
        mid = len(vertices) // 2
        v1 = vertices[:mid]
        v2 = vertices[mid:]
        
        # Color cross edges with base_color
        for u in v1:
            for v in v2:
                colors[_edge_idx(u, v)] = base_color
        
        # Recurse
        init_gallai_8(v1, base_color + 1)
        init_gallai_8(v2, base_color + 1) # Wait, this reuses colors?
        # Gallai coloring needs distinct colors for distinct "cuts" to maximize count.
        # Actually, for n=8, we need 7 colors.
        # If we use base_color for the top cut, and base_color+1 for the next, etc.
        
        # Let's just use a known valid pattern for K8.
        # Or simpler:
        # Just use min(i,j) logic? No, that's not Gallai.
        
        # I will use the recursive split but ensure colors are unique.
        # For n=16, max 15 colors.
        # Let's assign colors 0..14 based on the binary tree depth/node.
        
        pass

    # Hardcoded efficient initialization:
    # Use the property that coloring edge (i,j) with floor(log2(i xor j)) or similar?
    # No, let's just use the recursive split with unique color IDs.
    
    color_counter = 0
    def build_gallai(vertices):
        nonlocal color_counter
        if len(vertices) <= 1:
            return
        mid = len(vertices) // 2
        v1 = vertices[:mid]
        v2 = vertices[mid:]
        
        c = color_counter
        color_counter += 1
        
        for u in v1:
            for v in v2:
                colors[_edge_idx(u, v)] = c
        
        build_gallai(v1)
        build_gallai(v2)
        
    build_gallai(list(range(16)))
    # This uses 15 colors (0..14).
    
    # Precompute cycles for each edge
    # For edge e=(u,v), find all pairs (w,x) such that w,x != u,v.
    # Cycles: u-w-x-v and u-x-w-v
    edge_cycles = [[] for _ in range(n_edges)]
    edges_list = []
    for i in range(N):
        for j in range(i+1, N):
            edges_list.append((i, j))
    
    for idx, (u, v) in enumerate(edges_list):
        others = [x for x in range(N) if x != u and x != v]
        for w_idx in range(len(others)):
            for x_idx in range(w_idx+1, len(others)):
                w = others[w_idx]
                x = others[x_idx]
                # Cycle 1: u-w-x-v
                e1 = _edge_idx(u, w)
                e2 = _edge_idx(w, x)
                e3 = _edge_idx(x, v)
                edge_cycles[idx].append((e1, e2, e3))
                # Cycle 2: u-x-w-v
                e4 = _edge_idx(u, x)
                e5 = _edge_idx(x, w)
                e6 = _edge_idx(w, v)
                edge_cycles[idx].append((e4, e5, e6))

    best_colors = list(colors)
    best_distinct = len(set(colors))
    
    # Local Search
    # Optimization: Check validity by iterating cycles
    def is_valid_change(edge_idx, new_color):
        for e1, e2, e3 in edge_cycles[edge_idx]:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            # If c1, c2, c3 are distinct, and new_color is different from all, then invalid
            if c1 != c2 and c1 != c3 and c2 != c3:
                if new_color != c1 and new_color != c2 and new_color != c3:
                    return False
        return True

    # Track used colors
    used_set = set(colors)
    
    while time.time() < deadline:
        e = random.randint(0, n_edges - 1)
        # Try to pick an unused color first
        if len(used_set) < 120:
            # Find an unused color
            # Random 0..119
            c = random.randint(0, 119)
            if c in used_set:
                # If we want unused, and we picked used, we could just use it, or retry.
                # Let's bias 80% to unused
                if random.random() < 0.8:
                    # Pick from unused
                    # Simple: generate until unused?
                    # Or maintain list?
                    # Let's just check a few randoms
                    found = False
                    for _ in range(5):
                        c = random.randint(0, 119)
                        if c not in used_set:
                            found = True
                            break
                    if not found:
                        c = random.choice(list(used_set))
                else:
                    c = random.choice(list(used_set))
        else:
            c = random.choice(list(used_set))
            
        if c == colors[e]:
            continue
            
        if is_valid_change(e, c):
            old_c = colors[e]
            colors[e] = c
            
            # Update used_set
            if old_c not in colors: # Check if old color still present?
                # This is O(120). Acceptable.
                # Better: keep counts
                pass
            # Let's maintain counts for efficiency
            # I'll refactor to use counts in the final code below.

    return best_colors

# REFACTORING FOR EFFICIENCY AND CORRECTNESS IN FINAL OUTPUT
