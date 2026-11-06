import time
from itertools import combinations
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the lexicographic index of edge (i,j) where 0 <= i < j < N."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def idx_to_edge(idx):
    """Convert lexicographic edge index back to (i, j)."""
    # Binary search or direct computation
    # Row i starts at index: i*(N-1) - i*(i-1)//2
    # For a given idx, find i such that start_i <= idx < start_{i+1}
    lo, hi = 0, N - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        start_mid = mid * (N - 1) - mid * (mid - 1) // 2
        if start_mid <= idx:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    start_i = i * (N - 1) - i * (i - 1) // 2
    pos = idx - start_i
    j = i + 1 + pos
    return i, j

def get_4cycles():
    """Precompute all 4-cycles (3 per 4-vertex subset)."""
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three possible 4-cycles on {a,b,c,d}:
        # Cycle 1: a-b-c-d-a (edges: ab, bc, cd, da)
        # Cycle 2: a-b-d-c-a (edges: ab, bd, dc, ca)
        # Cycle 3: a-c-b-d-a (edges: ac, cb, bd, da)
        cycles.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
        cycles.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
        cycles.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))
    return cycles

def build_initial():
    """Build the initial Gallai-decomposition coloring (19 colors)."""
    dp = [0] * (N + 1)
    split = [0] * (N + 1)
    dp[1] = 0
    dp[2] = 1
    dp[3] = 2
    dp[4] = 4
    for n in range(5, N + 1):
        best_val = 0
        best_k = 1
        for k in range(1, n):
            val = dp[k] + dp[n - k] + 1
            if val > best_val:
                best_val = val
                best_k = k
        dp[n] = best_val
        split[n] = best_k
        
    colors = [0] * NUM_EDGES
    
    def assign_edge(u, v, c):
        colors[edge_idx(u, v)] = c
        
    def build_block(vertices, start_color):
        n = len(vertices)
        if n == 1:
            return start_color
        if n == 2:
            u, v = vertices
            assign_edge(u, v, start_color)
            return start_color + 1
        if n == 3:
            u, v, w = vertices
            assign_edge(u, v, start_color)
            assign_edge(u, w, start_color + 1)
            assign_edge(v, w, start_color + 2)
            return start_color + 3
        if n == 4:
            u, v, w, x = vertices
            A = start_color
            B = start_color + 1
            C = start_color + 2
            D = start_color + 3
            assign_edge(u, v, A)
            assign_edge(w, x, A)
            assign_edge(u, w, B)
            assign_edge(v, x, B)
            assign_edge(u, x, C)
            assign_edge(v, w, D)
            return start_color + 4
        k = split[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        next_color = build_block(v1, start_color)
        next_color = build_block(v2, next_color)
        cross_color = next_color
        for u in v1:
            for v in v2:
                assign_edge(u, v, cross_color)
        return cross_color + 1
        
    build_block(list(range(N)), 0)
    return colors

def check_validity(colors, cycles):
    """Check if no 4-cycle has four distinct colors."""
    for e1, e2, e3, e4 in cycles:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            return False
    return True

def local_search(initial_colors, cycles, time_limit=55.0):
    """Local search to try to increase color count."""
    start_time = time.time()
    colors = list(initial_colors)
    best_colors = list(initial_colors)
    best_count = len(set(colors))
    
    # Precompute for each edge, which cycles it participates in
    edge_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3, e4) in enumerate(cycles):
        for e in (e1, e2, e3, e4):
            edge_cycles[e].append(ci)
    
    # Count how many times each color is used
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    
    # For each edge, precompute the set of colors appearing in each cycle it's in
    # This helps us quickly check if recoloring an edge would create a rainbow
    
    random.seed(42)
    max_iterations = 500000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random edge to try recoloring
        edge = random.randint(0, NUM_EDGES - 1)
        old_color = colors[edge]
        
        # Try to assign a new color (one not currently used, or a less common color)
        # Strategy: try to introduce a new color if possible
        used_colors = set(colors)
        available = [c for c in range(NUM_EDGES) if c not in used_colors]
        
        if available:
            new_color = random.choice(available)
        else:
            # Try to consolidate: pick a color that appears few times
            # This might allow other edges to use new colors later
            new_color = random.choice(list(color_counts.keys()))
            if new_color == old_color:
                continue
        
        # Check if recoloring this edge maintains validity
        valid = True
        for ci in edge_cycles[edge]:
            e1, e2, e3, e4 = cycles[ci]
            # Get the other 3 edges' colors
            other_colors = []
            for e in (e1, e2, e3, e4):
                if e == edge:
                    other_colors.append(new_color)
                else:
                    other_colors.append(colors[e])
            if len(set(other_colors)) == 4:
                valid = False
                break
        
        if valid:
            colors[edge] = new_color
            # Update color counts
            color_counts[old_color] -= 1
            if color_counts[old_color] == 0:
                del color_counts[old_color]
            color_counts[new_color] = color_counts.get(new_color, 0) + 1
            
            current_count = len(color_counts)
            if current_count > best_count:
                best_count = current_count
                best_colors = list(colors)
        
        # Occasionally try to consolidate colors to free up new color slots
        if iteration % 1000 == 0 and color_counts:
            # Find a color used only once and try to merge it
            single_use = [c for c, cnt in color_counts.items() if cnt == 1]
            if single_use:
                c_to_merge = random.choice(single_use)
                # Find an edge with this color
                for e in range(NUM_EDGES):
                    if colors[e] == c_to_merge:
                        # Try to change it to another existing color
                        other_colors = [c for c in color_counts if c != c_to_merge]
                        if other_colors:
                            target_color = random.choice(other_colors)
                            valid = True
                            for ci in edge_cycles[e]:
                                e1, e2, e3, e4 = cycles[ci]
                                other_cs = []
                                for ee in (e1, e2, e3, e4):
                                    if ee == e:
                                        other_cs.append(target_color)
                                    else:
                                        other_cs.append(colors[ee])
                                if len(set(other_cs)) == 4:
                                    valid = False
                                    break
                            if valid:
                                colors[e] = target_color
                                color_counts[c_to_merge] -= 1
                                if color_counts[c_to_merge] == 0:
                                    del color_counts[c_to_merge]
                                color_counts[target_color] = color_counts.get(target_color, 0) + 1
                        break
    
    return best_colors, best_count

def construct():
    """
    Construct a coloring of K_16 edges with no rainbow 4-cycle.
    Starts with Gallai decomposition, then uses local search to maximize colors.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    # Build initial coloring
    initial = build_initial()
    cycles = get_4cycles()
    
    # Verify initial is valid
    if not check_validity(initial, cycles):
        # Fallback: return initial as-is (shouldn't happen)
        return initial
    
    # Local search
    best_colors, best_count = local_search(initial, cycles, time_limit=deadline - (time.time() - start_time))
    
    # Final verification
    if not check_validity(best_colors, cycles):
        # Return initial if local search produced invalid result
        return initial
    
    return best_colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
