import time
import random
import itertools

N = 16
DEADLINE = 75.0  # safety margin before 90s kill

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

    # --- Precompute neighbors for faster validity checks ---
    # For each edge (u,v), we need to check all pairs (w,x) disjoint from u,v.
    # The cycles are:
    # 1. u-v-w-x-u  (edges uv, vw, wx, xu)
    # 2. u-v-x-w-u  (edges uv, vx, xw, wu)
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Precompute list of (w, x) pairs for each edge (u, v)
    # w, x must be distinct and not u, v
    edge_to_quads = {}
    for (u, v) in all_edges:
        others = [i for i in range(N) if i != u and i != v]
        pairs = []
        for w in others:
            for x in others:
                if w != x:
                    pairs.append((w, x))
        edge_to_quads[(u, v)] = pairs

    def is_valid_recolor(colors, u, v, new_color):
        """Check if recoloring (u,v) to new_color is valid."""
        old_color = colors[u][v]
        if old_color == new_color:
            return True
        
        # Temporarily set new color
        colors[u][v] = new_color
        colors[v][u] = new_color
        
        valid = True
        for (w, x) in edge_to_quads[(u, v)]:
            # Cycle 1: u-v-w-x-u
            # Edges: (u,v), (v,w), (w,x), (x,u)
            c1 = colors[u][v]
            c2 = colors[v][w]
            c3 = colors[w][x]
            c4 = colors[x][u]
            if len({c1, c2, c3, c4}) == 4:
                valid = False
                break
            
            # Cycle 2: u-v-x-w-u
            # Edges: (u,v), (v,x), (x,w), (w,u)
            c1 = colors[u][v]
            c2 = colors[v][x]
            c3 = colors[x][w]
            c4 = colors[w][u]
            if len({c1, c2, c3, c4}) == 4:
                valid = False
                break
                
        if not valid:
            colors[u][v] = old_color
            colors[v][u] = old_color
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
    
    # Current color counts to help choose new colors
    color_counts = {}
    for i in range(N):
        for j in range(i + 1, N):
            c = colors[i][j]
            color_counts[c] = color_counts.get(c, 0) + 1

    # --- Local Search ---
    rng = random.Random(42)
    max_iter = 50000
    no_improve_count = 0
    max_no_improve = 5000
    
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE:
            break
            
        # Pick a random edge
        u, v = rng.choice(all_edges)
        
        # Strategy: 
        # 1. Try to introduce a new color (not in current set)
        # 2. Try to recolor to an existing color (to free up a slot or merge)
        # 3. Try to recolor to a random color
        
        current_colors = set(color_counts.keys())
        
        # 60% chance to try a new color
        if rng.random() < 0.6:
            # Pick a random new color from 0..119 not in current_colors
            # If all are used, pick random
            available_new = [c for c in range(120) if c not in current_colors]
            if available_new:
                new_color = rng.choice(available_new)
            else:
                new_color = rng.randint(0, 119)
        else:
            # Pick a random existing color
            new_color = rng.choice(list(current_colors))
            
        if is_valid_recolor(colors, u, v, new_color):
            old_color = colors[u][v] # This is the new color now, need to revert logic or track old
            # Wait, is_valid_recolor reverts on failure. On success, it keeps the new color.
            # We need to update color_counts.
            # We need the OLD color. Let's fetch it before calling? 
            # No, is_valid_recolor changes colors[u][v].
            # Let's refactor is_valid_recolor to return the old color or just handle it outside.
            
            # Actually, let's just re-fetch the old color by reverting? No.
            # Let's modify the loop to get old_color first.
            
            # Since is_valid_recolor modifies colors, we lost old_color if we didn't save it.
            # Let's save it before calling.
            
            # Re-do this step properly in the code below.
            pass

    # The logic above is slightly flawed in the draft. Let's write the clean version.
    
    # Reset colors to initial
    colors = [row[:] for row in best_colors] # best_colors is initial
    color_counts = {}
    for i in range(N):
        for j in range(i + 1, N):
            c = colors[i][j]
            color_counts[c] = color_counts.get(c, 0) + 1
            
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE:
            break
            
        u, v = rng.choice(all_edges)
        old_color = colors[u][v]
        
        current_colors = set(color_counts.keys())
        
        if rng.random() < 0.6:
            available_new = [c for c in range(120) if c not in current_colors]
            if available_new:
                new_color = rng.choice(available_new)
            else:
                new_color = rng.randint(0, 119)
        else:
            new_color = rng.choice(list(current_colors))
            
        if new_color == old_color:
            continue
            
        if is_valid_recolor(colors, u, v, new_color):
            # Success
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                del color_counts[old_color]
            color_counts[new_color] = color_counts.get(new_color, 0) + 1
            
            new_count = len(color_counts)
            if new_count > best_count:
                best_count = new_count
                best_colors = [row[:] for row in colors]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Restart from best
            colors = [row[:] for row in best_colors]
            color_counts = {}
            for i in range(N):
                for j in range(i + 1, N):
                    c = colors[i][j]
                    color_counts[c] = color_counts.get(c, 0) + 1
            no_improve_count = 0

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
