import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
ALL_COLORS = set(range(120))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute edge indices
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    # Precompute 4-cycles
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        c1 = [e12, e23, e34, e14]
        cycles.append(c1)
        for e in c1:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c2 = [e12, e24, e34, e13]
        cycles.append(c2)
        for e in c2:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c3 = [e13, e23, e24, e14]
        cycles.append(c3)
        for e in c3:
            edge_to_cycles[e].append(len(cycles) - 1)

    def count_violations(colors):
        """Count number of 4-cycles with 4 distinct colors."""
        count = 0
        for cyc in cycles:
            c0 = colors[cyc[0]]
            c1 = colors[cyc[1]]
            c2 = colors[cyc[2]]
            c3 = colors[cyc[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                count += 1
        return count

    def is_valid(colors):
        return count_violations(colors) == 0

    def count_colors(colors):
        return len(set(colors))

    def energy(colors):
        """Energy = violations * 1000 - colors_used. Lower is better."""
        v = count_violations(colors)
        c = count_colors(colors)
        return v * 1000 - c

    random.seed(42)
    
    # Structural construction: partition into 4 groups of 4
    # Within each group: proper 3-edge-coloring of K4
    # Between each pair of groups: 2 colors
    def structural_initial():
        colors = [0] * NUM_EDGES
        groups = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
        vertex_group = [0]*N
        for gi, g in enumerate(groups):
            for v in g:
                vertex_group[v] = gi
        
        # Within-group coloring: K4 has 6 edges, use 3 colors (proper edge coloring)
        # K4 edges: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        # Proper 3-edge-coloring: 
        # (0,1)->0, (0,2)->1, (0,3)->2, (1,2)->2, (1,3)->1, (2,3)->0
        k4_coloring = {}
        k4_coloring[(0,1)] = 0
        k4_coloring[(0,2)] = 1
        k4_coloring[(0,3)] = 2
        k4_coloring[(1,2)] = 2
        k4_coloring[(1,3)] = 1
        k4_coloring[(2,3)] = 0
        
        # Between-group: for groups gi < gj, use 2 colors
        # Color base: 12 + 2*(gi*gj - gi - gj) ... let's just use a counter
        between_color = 12
        group_pair_colors = {}
        for gi in range(4):
            for gj in range(gi+1, 4):
                group_pair_colors[(gi,gj)] = [between_color, between_color+1]
                between_color += 2
        
        for i in range(N):
            for j in range(i+1, N):
                eidx = edge_to_idx[(i,j)]
                gi = vertex_group[i]
                gj = vertex_group[j]
                if gi == gj:
                    # Within group: map local indices
                    # Find positions of i and j in the group
                    g = groups[gi]
                    pos_i = g.index(i)
                    pos_j = g.index(j)
                    local_pair = (pos_i, pos_j) if pos_i < pos_j else (pos_j, pos_i)
                    colors[eidx] = gi * 3 + k4_coloring[local_pair]
                else:
                    # Between groups
                    pair_key = (gi, gj) if gi < gj else (gj, gi)
                    # Use position within groups to pick which of the 2 colors
                    g_i = groups[gi]
                    g_j = groups[gj]
                    pos_in_i = g_i.index(i) if gi < gj else g_j.index(j)
                    # Alternate based on position
                    color_idx = pos_in_i % 2
                    colors[eidx] = group_pair_colors[pair_key][color_idx]
        
        return colors

    # Try structural initial
    best_score = 0
    best_colors = [0] * NUM_EDGES
    best_energy = float('inf')
    
    colors_struct = structural_initial()
    v = count_violations(colors_struct)
    c = count_colors(colors_struct)
    e = v * 1000 - c
    print(f"Structural init: violations={v}, colors={c}, energy={e}")
    
    if v == 0:
        best_score = c
        best_colors = colors_struct[:]
        best_energy = e
    else:
        # Fallback to all-zero
        best_score = 1
        best_colors = [0] * NUM_EDGES
        best_energy = -1

    # Local search to improve: try to increase colors while maintaining validity
    current = best_colors[:]
    current_energy = best_energy
    current_score = best_score
    
    # Track which colors are used
    def get_used_colors(colors):
        return set(colors)
    
    used_colors = get_used_colors(current)
    
    # Precompute: for each edge, the set of colors that would create a violation if assigned
    # This is dynamic, so we'll compute on the fly but optimize
    
    def would_create_violation(colors, e_idx, new_c):
        """Check if assigning new_c to edge e_idx creates any new violation."""
        for cyc_idx in edge_to_cycles[e_idx]:
            cyc = cycles[cyc_idx]
            other_edges = [cyc[i] for i in range(4) if cyc[i] != e_idx]
            c1 = colors[other_edges[0]]
            c2 = colors[other_edges[1]]
            c3 = colors[other_edges[2]]
            # Check if {new_c, c1, c2, c3} has 4 distinct colors
            if len({new_c, c1, c2, c3}) == 4:
                return True
        return False
    
    # Local search: try to change edges to new colors to increase color count
    max_iterations = 200000
    iteration = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 3 and iteration < max_iterations and no_improve < max_no_improve:
        iteration += 1
        
        # Pick a random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_c = current[e_idx]
        
        # Try to assign a new color (not currently used) to this edge
        if random.random() < 0.8:
            # Try a new color
            unused = [c for c in range(120) if c not in used_colors]
            if not unused:
                break
            new_c = random.choice(unused)
        else:
            # Try any color
            new_c = random.randint(0, 119)
        
        if new_c == old_c:
            continue
        
        # Check if this change is valid (doesn't create violations)
        if would_create_violation(current, e_idx, new_c):
            no_improve += 1
            continue
        
        # Accept the change
        current[e_idx] = new_c
        used_colors = get_used_colors(current)
        new_score = len(used_colors)
        
        if new_score > best_score:
            best_score = new_score
            best_colors = current[:]
            best_energy = -new_score  # No violations
            no_improve = 0
        else:
            no_improve += 1
    
    # Final verification
    final_v = count_violations(best_colors)
    final_c = count_colors(best_colors)
    if final_v == 0:
        return best_colors
    else:
        # Return all-zero as fallback
        return [0] * NUM_EDGES
