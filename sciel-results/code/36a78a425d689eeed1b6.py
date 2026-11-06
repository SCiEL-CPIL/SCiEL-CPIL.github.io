import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85.0

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def count_used(colors):
    return len(set(colors))

def is_valid_coloring(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    c1 = (colors[edge_idx(a,b)], colors[edge_idx(b,c)],
                          colors[edge_idx(c,d)], colors[edge_idx(d,a)])
                    if len(set(c1)) == 4:
                        return False
                    c2 = (colors[edge_idx(a,b)], colors[edge_idx(b,d)],
                          colors[edge_idx(d,c)], colors[edge_idx(c,a)])
                    if len(set(c2)) == 4:
                        return False
                    c3 = (colors[edge_idx(a,c)], colors[edge_idx(c,b)],
                          colors[edge_idx(b,d)], colors[edge_idx(d,a)])
                    if len(set(c3)) == 4:
                        return False
    return True

def construct():
    start_time = time.time()
    
    # Precompute all edges
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    
    # Precompute for each edge, which other edges are in 4-cycles with it
    # For edge (u,v), a 4-cycle uses (u,v) and two other edges
    # The 4 vertices are u, v, x, y where x,y are other vertices
    # Cycles containing (u,v):
    #   u-v-x-y-u: uses (u,v), (v,x), (x,y), (y,u)
    #   u-v-y-x-u: uses (u,v), (v,y), (y,x), (x,u)
    
    # For fast checking when recoloring edge (u,v) to color c:
    # Check all pairs (x,y) of other vertices
    # Cycle 1: colors[(v,x)], colors[(x,y)], colors[(y,u)] must not all be distinct from c
    # Cycle 2: colors[(v,y)], colors[(y,x)], colors[(x,u)] must not all be distinct from c
    
    # Build adjacency structure for faster checks
    # For each edge (u,v), store list of (x, y, e_vx, e_xy, e_yu) for cycle 1
    # and (x, y, e_vy, e_yx, e_xu) for cycle 2
    
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx
    
    # Precompute cycle partners for each edge
    # cycle_partners[idx] = list of (e1_idx, e2_idx, e3_idx) where these 3 edges + this edge form a C4
    cycle_partners = [[] for _ in range(NUM_EDGES)]
    
    for idx, (u, v) in enumerate(edges):
        others = [w for w in range(N) if w != u and w != v]
        for xi in range(len(others)):
            for xj in range(xi + 1, len(others)):
                x = others[xi]
                y = others[xj]
                # Cycle: u-v-x-y-u
                e1 = edge_to_idx[(v, x)]
                e2 = edge_to_idx[(x, y)]
                e3 = edge_to_idx[(y, u)]
                cycle_partners[idx].append((e1, e2, e3))
                # Cycle: u-v-y-x-u
                e1b = edge_to_idx[(v, y)]
                e2b = edge_to_idx[(y, x)]
                e3b = edge_to_idx[(x, u)]
                cycle_partners[idx].append((e1b, e2b, e3b))
    
    def is_valid_after_recolor_fast(colors, idx, new_color):
        """Check validity after recoloring edge at idx to new_color."""
        for (e1, e2, e3) in cycle_partners[idx]:
            if len({new_color, colors[e1], colors[e2], colors[e3]}) == 4:
                return False
        return True
    
    # Strategy: Greedy assignment with constraint propagation
    # Order edges by how constrained they are (more cycle partners = more constrained)
    # Assign colors preferring unused colors to maximize diversity
    
    # Initial coloring: use a structured base
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i % 8
    
    # Verify initial
    if not is_valid_coloring(colors):
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = i
    
    best_colors = list(colors)
    best_count = count_used(best_colors)
    
    # Local search with fast validity check
    rng = random.Random(42)
    
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < DEADLINE - 3:
        if no_improve_count > max_no_improve:
            # Restart with perturbation
            rng.seed(int(time.time() * 1000) % 1000000)
            no_improve_count = 0
            # Perturb: randomly recolor some edges
            for _ in range(10):
                idx = rng.randint(0, NUM_EDGES - 1)
                c = rng.randint(0, 119)
                if is_valid_after_recolor_fast(best_colors, idx, c):
                    best_colors[idx] = c
            continue
        
        idx = rng.randint(0, NUM_EDGES - 1)
        old_color = best_colors[idx]
        
        # Try to assign a new color (not currently used)
        used = set(best_colors)
        new_colors = [c for c in range(120) if c not in used]
        
        if new_colors:
            # Shuffle for variety
            rng.shuffle(new_colors)
            for new_color in new_colors[:5]:  # Try up to 5 new colors
                if is_valid_after_recolor_fast(best_colors, idx, new_color):
                    best_colors[idx] = new_color
                    no_improve_count = 0
                    break
            else:
                no_improve_count += 1
            continue
        
        # All 120 colors used? Try to recolor to free up a color
        # Pick a color that's used only once and see if we can reassign that edge
        color_counts = {}
        for c in best_colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        singleton_colors = [c for c, cnt in color_counts.items() if cnt == 1]
        
        if singleton_colors:
            target_color = rng.choice(singleton_colors)
            # Find the edge with this color
            for i, c in enumerate(best_colors):
                if c == target_color:
                    # Try to recolor this edge to something else
                    for new_color in range(120):
                        if new_color != target_color and is_valid_after_recolor_fast(best_colors, i, new_color):
                            best_colors[i] = new_color
                            break
                    no_improve_count += 1
                    break
        
        # Random recolor
        c = rng.randint(0, 119)
        if c != old_color and is_valid_after_recolor_fast(best_colors, idx, c):
            best_colors[idx] = c
            if old_color not in best_colors:
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
    
    # Final verification
    if not is_valid_coloring(best_colors):
        for i in range(N):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
    
    return best_colors
