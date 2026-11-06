"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# Precompute edge index for O(1) lookup
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each 4-subset, the three 4-cycles
# A 4-cycle is defined by 4 vertices in order. For {a,b,c,d} with a<b<c<d:
# Cycle 1: a-b-c-d -> edges (a,b),(b,c),(c,d),(d,a)
# Cycle 2: a-b-d-c -> edges (a,b),(b,d),(d,c),(c,a)
# Cycle 3: a-c-b-d -> edges (a,c),(c,b),(b,d),(d,a)

def build_cycles():
    """Return list of 4-cycles, each as a list of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d
                    e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c
                    e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d
                    e3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
                    cycles.append(e3)
    return cycles

# Precompute for each edge, which cycles it participates in
def build_edge_cycle_map(cycles):
    edge_to_cycles = [[] for _ in range(N*(N-1)//2)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

# Precompute for each cycle, the other 3 edges given one edge
def build_cycle_other_edges(cycles):
    cycle_other = {}
    for ci, cycle in enumerate(cycles):
        for pos, e in enumerate(cycle):
            others = [cycle[k] for k in range(4) if k != pos]
            cycle_other[(ci, pos)] = others
    return cycle_other

# Build all precomputation
ALL_CYCLES = build_cycles()
EDGE_TO_CYCLES = build_edge_cycle_map(ALL_CYCLES)
NUM_CYCLES = len(ALL_CYCLES)

def check_rainbow(cycle, colors):
    """Check if a 4-cycle has 4 distinct colors."""
    c0 = colors[cycle[0]]
    c1 = colors[cycle[1]]
    c2 = colors[cycle[2]]
    c3 = colors[cycle[3]]
    return len(set([c0, c1, c2, c3])) == 4

def count_violations(colors):
    """Count number of rainbow 4-cycles."""
    count = 0
    for cycle in ALL_CYCLES:
        if check_rainbow(cycle, colors):
            count += 1
    return count

def check_edge_change(colors, edge, new_color):
    """Check if changing edge to new_color creates any new rainbow cycle.
    Returns (valid, new_violation_count_delta) or just validity."""
    old_color = colors[edge]
    if old_color == new_color:
        return True, 0
    
    # Only need to check cycles containing this edge
    delta = 0
    for ci in EDGE_TO_CYCLES[edge]:
        cycle = ALL_CYCLES[ci]
        # Find position of edge in cycle
        pos = -1
        for k in range(4):
            if cycle[k] == edge:
                pos = k
                break
        # Check before change
        was_rainbow = check_rainbow(cycle, colors)
        # Check after change (simulate)
        other_colors = []
        for k in range(4):
            if k != pos:
                other_colors.append(colors[cycle[k]])
        after_rainbow = len(set(other_colors + [new_color])) == 4
        
        if was_rainbow and not after_rainbow:
            delta -= 1
        elif not was_rainbow and after_rainbow:
            delta += 1
    return delta <= 0, delta

def gallai_construct():
    """Construct a valid Gallai coloring using recursive partition."""
    colors = [0] * (N*(N-1)//2)
    
    def color_subtree(vertices, color_offset, num_colors):
        if len(vertices) <= 1:
            return
        # Partition into 2-3 parts
        n = len(vertices)
        if n == 2:
            i, j = vertices[0], vertices[1]
            colors[edge_idx(i, j)] = color_offset
            return
        if n == 3:
            a, b, c = vertices
            # Use 2 colors for triangle (Gallai)
            colors[edge_idx(a, b)] = color_offset
            colors[edge_idx(b, c)] = color_offset
            colors[edge_idx(a, c)] = color_offset + 1 if num_colors > 1 else color_offset
            return
        # Recursive: split into two groups
        mid = n // 2
        group1 = vertices[:mid]
        group2 = vertices[mid:]
        
        # Color within group1 with colors [color_offset, color_offset + k1)
        # Color within group2 with colors [color_offset, color_offset + k2)
        # Cross edges: use one color
        k1 = min(num_colors, 1)  # Use fewer colors in subgroups to save budget
        k2 = min(num_colors - k1 if num_colors > k1 else 1, 1)
        
        # Actually, let's use a simpler approach:
        # Within each group, use a single color
        # Between groups, use a different color
        cross_color = color_offset
        inner1_color = color_offset + 1 if num_colors > 1 else color_offset
        inner2_color = color_offset + 2 if num_colors > 2 else (color_offset + 1 if num_colors > 1 else color_offset)
        
        for idx_a in range(len(group1)):
            for idx_b in range(idx_a + 1, len(group1)):
                colors[edge_idx(group1[idx_a], group1[idx_b])] = inner1_color
        for idx_a in range(len(group2)):
            for idx_b in range(idx_a + 1, len(group2)):
                colors[edge_idx(group2[idx_a], group2[idx_b])] = inner2_color
        for a in group1:
            for b in group2:
                colors[edge_idx(a, b)] = cross_color
    
    color_subtree(list(range(N)), 0, 19)
    return colors

# Better Gallai construction: use the known 19-color scheme
def gallai_19():
    """
    Recursive partition:
    Level 0: {0..7} | {8..15} with cross color 0
    Level 1: {0..3}|{4..7} cross=1, {8..11}|{12..15} cross=2
    Level 2: pairs within each group of 4 get distinct colors
    This gives: 1 (top cross) + 2 (mid cross) + 8 (bottom, 2 per group of 4) + ... 
    Let me think more carefully.
    
    Actually for a Gallai coloring with maximum colors:
    - Partition V into A, B. All cross edges same color c0.
    - Recursively color A and B with disjoint color sets.
    
    For n=16: split into 8+8
    - Cross: 1 color
    - Each 8: split into 4+4
      - Cross: 1 color each (2 total for this level)
      - Each 4: split into 2+2
        - Cross: 1 color each (4 total)
        - Each 2: 1 edge, 1 color (8 total)
    Total: 1 + 2 + 4 + 8 = 15 colors
    
    To get 19, we can be smarter at the 4-vertex level:
    For 4 vertices, a Gallai coloring can use up to 3 colors (not 2).
    A 4-clique Gallai coloring: partition into 2+2, cross uses 1 color,
    each pair uses 1 color. That's 3 colors per group of 4.
    
    So: 1 + 2 + 4*3 + 0 = 1 + 2 + 12 = 15... hmm.
    
    Wait, for 4 vertices split as 2+2:
    - Cross edges (4 of them): 1 color
    - Edge in first pair: 1 color
    - Edge in second pair: 1 color
    Total: 3 colors per group of 4.
    
    16 = 8+8: 1 color
    8 = 4+4: 1 color each, so 2 colors at this level
    4 = 2+2: 3 colors each, so 4*3 = 12 colors at this level
    Total: 1 + 2 + 12 = 15
    
    To get 19, we need a different partition. What if we don't always split evenly?
    
    Alternative: 16 = 7+9
    7 = 3+4: 
    3-vertex Gallai: 2 colors
    4-vertex Gallai: 3 colors
    Cross: 1 color
    So 7-vertex: 1 + 2 + 3 = 6 colors
    9 = 4+5:
    4: 3 colors
    5 = 2+3: 1 + 1 + 2 = 4 colors
    Cross: 1 color
    So 9-vertex: 1 + 3 + 4 = 8 colors
    Total: 1 + 6 + 8 = 15... still 15.
    
    Hmm, the maximum number of colors in a Gallai coloring of K_n is known.
    For n=16, it might be 15. But the task says 19 was achieved.
    
    Let me reconsider. Maybe the 19-color solution isn't a pure Gallai coloring but still avoids rainbow 4-cycles.
    """
    # Let me just use a simple valid construction and then improve with local search
    colors = [0] * (N*(N-1)//2)
    
    # Simple hierarchical: 8+8, then 4+4 in each, then 2+2 in each
    # Level 0: cross color 0
    for a in range(8):
        for b in range(8, 16):
            colors[edge_idx(a, b)] = 0
    
    # Level 1: within 0-7, split 0-3 and 4-7, cross color 1
    for a in range(4):
        for b in range(4, 8):
            colors[edge_idx(a, b)] = 1
    # Within 8-15, split 8-11 and 12-15, cross color 2
    for a in range(8, 12):
        for b in range(12, 16):
            colors[edge_idx(a, b)] = 2
    
    # Level 2: within each group of 4, split into 2+2
    # Group 0-3: 0-1 and 2-3, cross color 3
    for a in range(2):
        for b in range(2, 4):
            colors[edge_idx(a, b)] = 3
    # Group 4-7: 4-5 and 6-7, cross color 4
    for a in range(4, 6):
        for b in range(6, 8):
            colors[edge_idx(a, b)] = 4
    # Group 8-11: 8-9 and 10-11, cross color 5
    for a in range(8, 10):
        for b in range(10, 12):
            colors[edge_idx(a, b)] = 5
    # Group 12-15: 12-13 and 14-15, cross color 6
    for a in range(12, 14):
        for b in range(14, 16):
            colors[edge_idx(a, b)] = 6
    
    # Level 3: within each pair
    # (0,1): color 7
    colors[edge_idx(0, 1)] = 7
    # (2,3): color 8
    colors[edge_idx(2, 3)] = 8
    # (4,5): color 9
    colors[edge_idx(4, 5)] = 9
    # (6,7): color 10
    colors[edge_idx(6, 7)] = 10
    # (8,9): color 11
    colors[edge_idx(8, 9)] = 11
    # (10,11): color 12
    colors[edge_idx(10, 11)] = 12
    # (12,13): color 13
    colors[edge_idx(12, 13)] = 13
    # (14,15): color 14
    colors[edge_idx(14, 15)] = 14
    
    return colors

def local_search(colors, max_time=45.0):
    """Improve coloring by trying to introduce new colors."""
    start_time = time.time()
    current_colors = list(colors)
    current_num_colors = len(set(current_colors))
    best_colors = list(current_colors)
    best_num_colors = current_num_colors
    
    # Track which colors are "used" and which are "available"
    all_possible_colors = set(range(120))
    
    # Precompute: for each edge, list of cycles it's in (already have EDGE_TO_CYCLES)
    
    # Strategy: try to change an edge to a new color (not currently used)
    # If it doesn't create a rainbow cycle, accept it.
    
    max_iterations = 500000
    iterations = 0
    
    while time.time() - start_time < max_time and iterations < max_iterations:
        iterations += 1
        if time.time() - start_time > max_time - 5:
            break
        
        # Get current used colors
        used_colors = set(current_colors)
        available = all_possible_colors - used_colors
        
        if not available:
            break
        
        # Try random edges and random available colors
        found_improvement = False
        for _ in range(20):  # Try 20 random changes per iteration
            if time.time() - start_time > max_time - 5:
                break
            edge = random.randint(0, N*(N-1)//2 - 1)
            new_color = random.choice(list(available))
            
            # Check if this change is valid
            old_color = current_colors[edge]
            valid = True
            for ci in EDGE_TO_CYCLES[edge]:
                cycle = ALL_CYCLES[ci]
                # Find position
                pos = -1
                for k in range(4):
                    if cycle[k] == edge:
                        pos = k
                        break
                # Get other 3 colors
                others = set()
                for k in range(4):
                    if k != pos:
                        others.add(current_colors[cycle[k]])
                # After change, the 4 colors are others + {new_color}
                if len(others | {new_color}) == 4:
                    valid = False
                    break
            
            if valid:
                current_colors[edge] = new_color
                found_improvement = True
                break
        
        if found_improvement:
            new_num = len(set(current_colors))
            if new_num > best_num_colors:
                best_num_colors = new_num
                best_colors = list(current_colors)
        else:
            # No improvement found in this batch; try swapping colors between edges
            # or just continue
            pass
    
    return best_colors, best_num_colors

def construct():
    deadline = 55.0  # Safety margin before 60s
    start = time.time()
    
    # Start with Gallai coloring (15 colors)
    colors = gallai_19()
    num_colors = len(set(colors))
    
    # Verify it's valid
    assert count_violations(colors) == 0, "Initial coloring has violations!"
    
    # Local search to improve
    if time.time() - start < deadline - 10:
        colors, num_colors = local_search(colors, max_time=deadline - (time.time() - start) - 5)
    
    # Final verification
    if count_violations(colors) != 0:
        # Fallback to the valid Gallai coloring
        colors = gallai_19()
    
    return colors

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
