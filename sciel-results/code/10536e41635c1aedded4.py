import time
import itertools
from collections import defaultdict

N = 16
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (i - 1) // 2 + (j - 1)

def _is_valid(colors, n):
    """Check if no 4-cycle has 4 distinct edge colors."""
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Three 4-cycles on vertices a,b,c,d:
        # Cycle 1: a-b-c-d-a
        c1 = {colors[a][b], colors[b][c], colors[c][d], colors[d][a]}
        if len(c1) == 4:
            return False
        # Cycle 2: a-b-d-c-a
        c2 = {colors[a][b], colors[b][d], colors[d][c], colors[c][a]}
        if len(c2) == 4:
            return False
        # Cycle 3: a-c-b-d-a
        c3 = {colors[a][c], colors[c][b], colors[b][d], colors[d][a]}
        if len(c3) == 4:
            return False
    return True

def _count_colors(colors, n):
    """Count distinct colors used."""
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            seen.add(colors[i][j])
    return len(seen)

def _generate_structural_coloring(n, split_strategy):
    """
    Generate a coloring using hierarchical clustering.
    split_strategy: dict mapping size -> list of part sizes (multi-way split)
    Returns (colors_matrix, num_colors_used)
    """
    colors = [[-1] * n for _ in range(n)]
    
    def solve(vertices, current_color):
        n_v = len(vertices)
        if n_v <= 1:
            return current_color
        if n_v == 2:
            u, v = vertices
            colors[u][v] = current_color
            colors[v][u] = current_color
            return current_color + 1
        if n_v == 3:
            u, v, w = vertices
            colors[u][v] = current_color
            colors[v][u] = current_color
            colors[v][w] = current_color + 1
            colors[w][v] = current_color + 1
            colors[u][w] = current_color + 2
            colors[w][u] = current_color + 2
            return current_color + 3
        
        # Get split strategy for this size
        parts_sizes = split_strategy.get(n_v, [n_v // 2, n_v - n_v // 2])
        
        # Split vertices into parts
        parts = []
        idx = 0
        for size in parts_sizes:
            parts.append(vertices[idx:idx+size])
            idx += size
        
        # Assign cross-edge colors: each pair of parts gets a unique color
        num_parts = len(parts)
        cross_colors = {}
        color_offset = current_color
        for pi in range(num_parts):
            for pj in range(pi+1, num_parts):
                cross_colors[(pi, pj)] = color_offset
                color_offset += 1
        
        # Assign colors to cross edges
        for pi in range(num_parts):
            for pj in range(pi+1, num_parts):
                c = cross_colors[(pi, pj)]
                for u in parts[pi]:
                    for v in parts[pj]:
                        colors[u][v] = c
                        colors[v][u] = c
        
        # Recursively solve each part
        next_color = color_offset
        for part in parts:
            next_color = solve(part, next_color)
        
        return next_color
    
    vertices = list(range(n))
    final_color = solve(vertices, 0)
    num_colors = final_color
    return colors, num_colors

def construct():
    start_time = time.time()
    
    # Precompute G values via dynamic programming with multi-way splits
    # G[n] = minimum number of colors needed for n vertices
    # For a split into parts of sizes s1, s2, ..., sk:
    #   cost = (k choose 2) + sum(G[si])
    # We want to minimize this over all partitions
    
    # Initialize
    G = [0] * (N + 1)
    G[0] = 0
    G[1] = 0
    G[2] = 1
    G[3] = 3
    
    # Best split strategy for each n
    best_split = {1: [1], 2: [2], 3: [3]}
    
    # Compute G[n] for n=4..N
    for n in range(4, N + 1):
        best_cost = float('inf')
        best_parts = None
        
        # Try all possible partitions of n into parts of size >= 2
        # (parts of size 1 can be merged with others, but let's be general)
        # Actually, parts of size 1 are allowed but suboptimal usually
        # Let's try partitions where each part has size >= 2, plus handle size 1
        
        # Generate all integer partitions of n with parts >= 1
        # But we only care about the multiset of part sizes
        # For efficiency, try small number of parts (2 to 6)
        
        max_parts = min(n, 6)
        for k in range(2, max_parts + 1):
            # Try all compositions of n into k positive integers
            # This is expensive, so limit to reasonable ranges
            # For small n, we can enumerate
            if n <= 12:
                # Enumerate all compositions
                for comp in _compositions(n, k):
                    # Filter: each part should be >= 1, and we need G[part] defined
                    if any(p > N for p in comp):
                        continue
                    cost = k * (k - 1) // 2 + sum(G[p] for p in comp)
                    if cost < best_cost:
                        best_cost = cost
                        best_parts = comp
            else:
                # For larger n, try balanced splits and some variations
                base = n // k
                rem = n % k
                # Balanced: rem parts of size base+1, k-rem parts of size base
                comp = [base + 1] * rem + [base] * (k - rem)
                if all(p >= 1 and p <= N for p in comp):
                    cost = k * (k - 1) // 2 + sum(G[p] for p in comp)
                    if cost < best_cost:
                        best_cost = cost
                        best_parts = comp
                
                # Also try some unbalanced splits
                for i in range(2, min(k, n // 2 + 1)):
                    part1 = i
                    part2 = n - i
                    if part2 >= k - 1:  # need to split part2 into k-1 parts
                        # Try splitting part2 into k-1 parts
                        base2 = part2 // (k - 1)
                        rem2 = part2 % (k - 1)
                        comp2 = [base2 + 1] * rem2 + [base2] * ((k - 1) - rem2)
                        if all(p >= 1 for p in comp2):
                            full_comp = [part1] + comp2
                            cost = k * (k - 1) // 2 + sum(G[p] for p in full_comp)
                            if cost < best_cost:
                                best_cost = cost
                                best_parts = full_comp
        
        if best_parts is None:
            # Fallback: binary split
            k = n // 2
            best_parts = [k, n - k]
            best_cost = 1 + G[k] + G[n - k]
        
        G[n] = best_cost
        best_split[n] = best_parts
    
    # Now construct the coloring using the optimal split strategy
    split_strategy = {}
    for n in range(1, N + 1):
        split_strategy[n] = best_split[n]
    
    colors, num_colors = _generate_structural_coloring(N, split_strategy)
    
    # Verify validity
    if not _is_valid(colors, N):
        # Fallback to simple binary split
        simple_split = {n: [n // 2, n - n // 2] for n in range(4, N + 1)}
        simple_split[1] = [1]
        simple_split[2] = [2]
        simple_split[3] = [3]
        colors, num_colors = _generate_structural_coloring(N, simple_split)
    
    # Flatten to list of 120 integers
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(colors[i][j])
    
    return result

def _compositions(n, k):
    """Generate all compositions of n into k positive parts."""
    if k == 1:
        yield (n,)
        return
    for i in range(1, n - k + 2):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest

# EVOLVE-BLOCK-END
