import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the lexicographic index of edge (i,j) where 0 <= i < j < N."""
    if i > j:
        i, j = j, i
    # Row i has edges (i, i+1), (i, i+2), ..., (i, N-1)
    # Number of edges before row i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Position within row i: j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def idx_to_edge(idx):
    """Convert lexicographic edge index back to (i, j)."""
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

def build_cycle_data():
    """Precompute all 4-cycles and which edges they contain.
    Returns list of (edges_tuple, vertex_set) for all 3 cycles on each 4-vertex subset."""
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on 4 vertices:
        # 1. a-b-c-d-a: edges (a,b), (b,c), (c,d), (d,a)
        # 2. a-b-d-c-a: edges (a,b), (b,d), (d,c), (c,a)
        # 3. a-c-b-d-a: edges (a,c), (c,b), (b,d), (d,a)
        cycles.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)))
        cycles.append((edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)))
        cycles.append((edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)))
    return cycles

def check_validity(colors, cycles):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for e1, e2, e3, e4 in cycles:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    """
    Construct a coloring of K_16 edges with no rainbow 4-cycle.
    Maximizes the number of distinct colors used.
    
    Strategy: 
    1. Start from a Gallai-like recursive partition (score 20 baseline).
    2. Use a modular arithmetic construction (c(i,j) = (i + 3j) % 19) as an alternative start.
    3. Apply a randomized local search (simulated annealing / hill climbing) to introduce new colors.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin before 90s kill
    
    cycles = build_cycle_data()
    
    # Precompute: for each edge, which cycles contain it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3, e4) in enumerate(cycles):
        for e in (e1, e2, e3, e4):
            edge_to_cycles[e].append(ci)

    # --- Strategy 1: Gallai Decomposition ---
    def build_gallai():
        dp = [0] * (N + 1)
        split = [0] * (N + 1)
        dp[1] = 0
        dp[2] = 1
        dp[3] = 3
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
            idx = edge_idx(u, v)
            colors[idx] = c
            
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
            
        all_vertices = list(range(N))
        build_block(all_vertices, 0)
        return colors

    # --- Strategy 2: Modular Arithmetic ---
    def build_modular(mod=19):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_idx(i, j)
                # Tuned linear combination
                colors[idx] = (i + 3 * j) % mod
        return colors

    # Generate candidate initial colorings
    candidates = []
    
    # Add Gallai
    c_gallai = build_gallai()
    if check_validity(c_gallai, cycles):
        candidates.append(c_gallai)
        
    # Add Modular variants
    for mod in [17, 19, 23]:
        for a in [1, 2, 3]:
            for b in [1, 2, 3]:
                colors = [0] * NUM_EDGES
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        idx = edge_idx(i, j)
                        colors[idx] = (a * i + b * j) % mod
                if check_validity(colors, cycles):
                    candidates.append(colors)
                if time.time() > deadline - 5:
                    break

    # Fallback to transitive if no valid candidates (should not happen)
    if not candidates:
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = i
        return colors

    # Select the best candidate (most colors)
    best_colors = max(candidates, key=count_colors)
    best_count = count_colors(best_colors)
    
    # Local Search: Try to improve by introducing new colors
    # We will use a simple hill-climbing / random restart approach.
    # We maintain a "best" coloring and try to mutate it.
    
    # Precompute available colors pool
    all_possible_colors = list(range(120))
    
    # We'll run multiple iterations of local search
    # Each iteration: pick a random edge, try to assign a new color (not currently used)
    # If valid, accept. If it increases count, keep.
    # To allow escaping local minima, we might accept non-improving valid moves with some probability,
    # but for simplicity and safety, let's stick to strict improvement or neutral moves that enable future improvement.
    # Actually, strict improvement on count is hard. Let's allow any valid move, but track best.
    # Better: Try to assign a NEW color (one not in the current set) to a random edge.
    # If valid, we gain a color.
    
    random.seed(42)
    
    current_colors = best_colors[:]
    current_count = best_count
    
    # We will try to increase the number of colors by assigning unused colors to edges.
    # This is a "coloring" problem where we want to maximize the palette size.
    
    # Optimization: Precompute the set of used colors for fast lookup
    used_colors = set(current_colors)
    
    # List of edges
    all_edges = list(range(NUM_EDGES))
    
    # We'll do a loop until time is up or no more improvements are possible
    # Since we only add new colors, the count is monotonically non-decreasing if we only accept valid new-color assignments.
    # However, sometimes we might need to change an existing color to free up a constraint to allow a new color elsewhere.
    # Let's stick to the simpler model first: just try to assign unused colors to random edges.
    
    attempts = 0
    max_attempts = 100000
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        attempts += 1
        
        # Determine unused colors
        unused_colors = [c for c in all_possible_colors if c not in used_colors]
        if not unused_colors:
            break
            
        # Pick a random edge
        e = random.choice(all_edges)
        # Pick a random unused color
        new_color = random.choice(unused_colors)
        
        old_color = current_colors[e]
        
        # Try the change
        current_colors[e] = new_color
        
        # Check validity only for cycles containing edge e
        valid = True
        for ci in edge_to_cycles[e]:
            e1, e2, e3, e4 = cycles[ci]
            if len({current_colors[e1], current_colors[e2], current_colors[e3], current_colors[e4]}) == 4:
                valid = False
                break
        
        if valid:
            # Update used colors
            # We need to check if old_color is still used elsewhere
            # For efficiency, we can just recalculate used_colors if count changes, or maintain a counter.
            # Given N=16, 120 edges, recalculating set(current_colors) is O(120) which is fine.
            new_used = set(current_colors)
            new_count = len(new_used)
            
            if new_count > best_count:
                best_colors = current_colors[:]
                best_count = new_count
                used_colors = new_used
            else:
                # If count didn't increase, revert to keep the "best" state consistent for further attempts?
                # Or keep the current state?
                # If we keep the current state, we might get stuck in a local optimum of "same count but different structure".
                # However, since we are picking random unused colors, and we only accept if valid,
                # the state changes even if count doesn't increase.
                # Let's keep the current state to explore the space of valid colorings with the same count,
                # hoping to find a configuration that allows a NEW color to be added later.
                used_colors = new_used
        else:
            # Revert
            current_colors[e] = old_color

    # Final check
    if not check_validity(best_colors, cycles):
        # Fallback to transitive
        for i in range(N):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
    
    return best_colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
    # Quick validity check
    cycles = build_cycle_data()
    print(f"Valid: {check_validity(result, cycles)}")
