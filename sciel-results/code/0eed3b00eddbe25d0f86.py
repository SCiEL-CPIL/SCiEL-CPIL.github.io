import time
import random

N = 16
DEADLINE = 50.0  # safety margin before 60s kill

# Precomputed optimal colors g(n) for n=1..16
# These values are derived from the maximum number of colors in K_n 
# without a rainbow C4, using a recursive splitting strategy.
G = [0, 0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]

# Precomputed best split k for n=4..16
SPLIT = {
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 6,
    14: 7,
    15: 6,
    16: 7
}

def construct():
    """
    Constructs a Gallai coloring of K_16, then attempts to increase the number
    of distinct colors via local search (edge recoloring and small swaps) while
    maintaining the no-rainbow-C4 constraint.
    """
    start_time = time.time()

    # --- Build initial Gallai coloring ---
    colors = [[0] * N for _ in range(N)]
    next_color = [0]

    def assign_colors(vertices):
        n = len(vertices)
        if n <= 1:
            return next_color[0]
        if n == 2:
            u, v = vertices
            c = next_color[0]
            colors[u][v] = c
            colors[v][u] = c
            next_color[0] += 1
            return next_color[0]
        if n == 3:
            u, v, w = vertices
            c1 = next_color[0]
            c2 = next_color[0] + 1
            c3 = next_color[0] + 2
            colors[u][v] = c1; colors[v][u] = c1
            colors[v][w] = c2; colors[w][v] = c2
            colors[u][w] = c3; colors[w][u] = c3
            next_color[0] += 3
            return next_color[0]
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        cross_color = next_color[0]
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        next_color[0] += 1
        assign_colors(v1)
        assign_colors(v2)
        return next_color[0]

    assign_colors(list(range(N)))

    # --- Precompute all 4-vertex subsets and their 3 four-cycles ---
    # A 4-cycle on vertices {a,b,c,d} in K4 has 3 possible cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d) — i.e., the 3 perfect-matchings complement
    # Actually in K4, the three 4-cycles are the three ways to arrange 4 vertices
    # in a cycle up to rotation/reflection. They are:
    # cycle1: a-b-c-d-a  (edges ab, bc, cd, da)
    # cycle2: a-b-d-c-a  (edges ab, bd, dc, ca)
    # cycle3: a-c-b-d-a  (edges ac, cb, bd, da)
    all_quads = []
    edge_to_cycles = { (i, j): [] for i in range(N) for j in range(i+1, N) }
    edge_to_cycles_rev = { (j, i): [] for i in range(N) for j in range(i+1, N) } # Not strictly needed if we normalize
    
    # We will store cycles as lists of edge tuples (min, max) for easy lookup?
    # No, colors are symmetric, so colors[u][v] == colors[v][u].
    # We just need to check the set of colors on the 4 edges.
    
    quad_index = 0
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on {a,b,c,d}:
                    # C1: a-b-c-d-a
                    # C2: a-b-d-c-a
                    # C3: a-c-b-d-a
                    cycles = [
                        [(a, b), (b, c), (c, d), (d, a)],
                        [(a, b), (b, d), (d, c), (c, a)],
                        [(a, c), (c, b), (b, d), (d, a)],
                    ]
                    all_quads.append(cycles)
                    for cyc in cycles:
                        for (u, v) in cyc:
                            # Normalize edge to (min, max) for consistent mapping if needed, 
                            # but our colors matrix is symmetric so we can just use the tuple as is for lookup in edge_to_cycles?
                            # The edge (u,v) in the cycle might be (2,0). 
                            # Let's map all edges to (min, max) for the index.
                            e_key = (u, v) if u < v else (v, u)
                            edge_to_cycles[e_key].append(quad_index)
                    quad_index += 1

    def is_valid(colors):
        """Check no 4-cycle has four distinct colors."""
        for cycles in all_quads:
            for cyc in cycles:
                cs = set()
                for (u, v) in cyc:
                    cs.add(colors[u][v])
                    if len(cs) == 4:
                        return False
        return True

    def count_colors(colors):
        seen = set()
        for i in range(N):
            for j in range(i + 1, N):
                seen.add(colors[i][j])
        return len(seen)

    best_colors = [row[:] for row in colors]
    best_count = count_colors(best_colors)

    # --- Local search: try recoloring individual edges ---
    rng = random.Random(42)
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Precompute a list of all possible colors? 
    # We only care about colors 0..119.
    
    def try_recolor(colors, edge, new_color):
        """Try recoloring `edge` to `new_color`; return True if valid."""
        u, v = edge
        old_color = colors[u][v]
        if old_color == new_color:
            return False
        
        # Get the indices of the 4-cycles containing this edge
        e_key = (u, v) if u < v else (v, u)
        cycle_indices = edge_to_cycles[e_key]
        
        # Tentatively recolor
        colors[u][v] = new_color
        colors[v][u] = new_color
        
        # Check validity for only the affected cycles
        for q_idx in cycle_indices:
            cycles = all_quads[q_idx]
            # The edge (u,v) is in this quad. 
            # We need to check if any of the 3 cycles in this quad is a rainbow.
            # Note: The quad might contain the edge (u,v) in one or more of its 3 cycles?
            # Yes, in a K4, each edge is in exactly 2 of the 3 4-cycles?
            # Let's verify.
            # Vertices a,b,c,d. Edge ab.
            # C1: a-b-c-d-a (uses ab)
            # C2: a-b-d-c-a (uses ab)
            # C3: a-c-b-d-a (does not use ab, uses ac, cb, bd, da)
            # So edge ab is in C1 and C2.
            # So for each quad index, we should check the specific cycles that contain the edge.
            # But it's faster to just check all 3 cycles in the quad if the edge is in the quad.
            # Since we only added the quad index if the edge is in *any* cycle of that quad,
            # and the edge is in the quad's vertex set, we just check the 3 cycles.
            
            for cyc in cycles:
                # Optimization: if the edge (u,v) is not in this specific cycle, we can skip?
                # But checking set of 4 colors is fast.
                cs = set()
                for (x, y) in cyc:
                    cs.add(colors[x][y])
                    if len(cs) == 4:
                        colors[u][v] = old_color
                        colors[v][u] = old_color
                        return False
        return True

    # Strategy: 
    # 1. Identify "candidate" edges for recoloring. 
    #    Edges that are part of a large monochromatic component are good candidates to break.
    # 2. Try to recolor them to a color that is *not* currently used, or a color that helps.
    #    Actually, trying to introduce a *new* color is the goal.
    
    # Let's try a deterministic pass first, then random.
    # Iterate through edges. For each edge, try all colors 0..119.
    # This is 120 * 120 = 14400 checks. Very fast.
    # We can do this in multiple passes.
    
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        # Try to increase color count by recoloring edges to unused colors
        # Or just any valid recoloring that might allow further improvements?
        # Simple hill climbing: accept any valid move that increases color count.
        # If no such move, accept any valid move? (Random walk)
        # Let's stick to "increase color count" or "keep same count but change structure".
        
        # To find "increase count" moves, we must recolor an edge to a color 
        # that is not currently present in the graph? 
        # No, recoloring to an existing color doesn't increase count.
        # So we must try colors that are *not* in `current_colors`.
        
        current_colors = set()
        for i in range(N):
            for j in range(i+1, N):
                current_colors.add(colors[i][j])
                
        unused_colors = [c for c in range(120) if c not in current_colors]
        
        if not unused_colors:
            break # Can't increase count if all 120 colors are used? Unlikely for N=16.
            
        # Shuffle edges to avoid bias
        edges_to_try = all_edges[:]
        rng.shuffle(edges_to_try)
        
        for edge in edges_to_try:
            if time.time() - start_time > DEADLINE:
                break
                
            # Try each unused color
            # Optimization: if we find one that works and increases count, take it and restart loop?
            # Or take the first one found?
            # Let's try all and pick the one that allows the most future moves? 
            # Too complex. Just pick the first valid one.
            
            for new_color in unused_colors:
                if try_recolor(colors, edge, new_color):
                    # It was valid. 
                    # Does it increase count? 
                    # Since new_color was unused, yes, count increases by 1.
                    new_count = count_colors(colors)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = [row[:] for row in colors]
                    improved = True
                    # After a successful recoloring, the set of unused colors changes.
                    # And the validity of other edges changes.
                    # Restart the edge iteration?
                    # For now, break out of color loop and continue to next edge?
                    # Or break out of edge loop and restart whole while?
                    # Restarting whole while is safer to re-evaluate `unused_colors`.
                    break 
            if improved:
                # If we improved, break out of edge loop to restart while
                # This is a "first-improvement" hill climbing strategy.
                break

    # Final check and return
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            result.append(best_colors[i][j])
    return result

if __name__ == "__main__":
    # Simple self-check
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Number of colors used: {len(set(res))}")
    print(f"First 10 colors: {res[:10]}")
    print(f"Time: {t1 - t0:.2f}s")
