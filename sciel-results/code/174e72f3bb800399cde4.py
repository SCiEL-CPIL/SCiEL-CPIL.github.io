import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

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
    
    # Precompute 4-cycles and for each edge, the "other 3 edges" in each cycle it belongs to
    # edge_to_cycle_others[e] is a list of tuples (e1, e2, e3)
    edge_to_cycle_others = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        # Cycle 1: v1-v2-v3-v4-v1 (Edges: e12, e23, e34, e14)
        edge_to_cycle_others[e12].append((e23, e34, e14))
        edge_to_cycle_others[e23].append((e12, e34, e14))
        edge_to_cycle_others[e34].append((e12, e23, e14))
        edge_to_cycle_others[e14].append((e12, e23, e34))
        
        # Cycle 2: v1-v2-v4-v3-v1 (Edges: e12, e24, e34, e13)
        edge_to_cycle_others[e12].append((e24, e34, e13))
        edge_to_cycle_others[e24].append((e12, e34, e13))
        edge_to_cycle_others[e34].append((e12, e24, e13))
        edge_to_cycle_others[e13].append((e12, e24, e34))
        
        # Cycle 3: v1-v3-v2-v4-v1 (Edges: e13, e23, e24, e14)
        edge_to_cycle_others[e13].append((e23, e24, e14))
        edge_to_cycle_others[e23].append((e13, e24, e14))
        edge_to_cycle_others[e24].append((e13, e23, e14))
        edge_to_cycle_others[e14].append((e13, e23, e24))

    def is_valid_color(e_idx, c, colors):
        """Check if color c is valid for edge e_idx given current colors."""
        for o1, o2, o3 in edge_to_cycle_others[e_idx]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            # If c1, c2, c3 are distinct, c must be one of them
            if c1 != c2 and c1 != c3 and c2 != c3:
                if c != c1 and c != c2 and c != c3:
                    return False
        return True

    def can_take_new_color(e_idx, colors):
        """Check if edge e_idx can be assigned a brand new color (not currently used anywhere)."""
        # A new color is not in {c1, c2, c3}, so the condition "c not in S" is always true for new colors.
        # Thus, we just need |S| < 3 for all cycles.
        for o1, o2, o3 in edge_to_cycle_others[e_idx]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            if c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    # Initialize with min(i,j) coloring
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_to_idx[(i, j)]] = i
            
    # Count distinct colors
    used_colors = set(colors)
    next_new_color = max(used_colors) + 1 if used_colors else 0
    
    # We want to maximize distinct colors.
    # Strategy: 
    # 1. Try to find an edge that can take a new color.
    # 2. If found, assign it.
    # 3. If not found, do a local search move to "loosen" constraints.
    
    random.seed(42)
    
    while time.time() < deadline - 5:
        found_new = False
        # Check all edges for potential new color
        # We can shuffle the order of edges to break symmetry
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for e in edge_order:
            if can_take_new_color(e, colors):
                # Assign new color
                if next_new_color < MAX_COLOR:
                    colors[e] = next_new_color
                    used_colors.add(next_new_color)
                    next_new_color += 1
                    found_new = True
                    break
            # If we run out of colors, we stop adding new ones
            if next_new_color >= MAX_COLOR:
                break
                
        if not found_new:
            # No edge can take a new color. 
            # Perform a local move to try to create flexibility.
            # Pick a random edge and change its color to a valid existing color.
            # This might change the "distinctness" of other cycles, allowing another edge to take a new color.
            
            # To be efficient, we can pick an edge that is "critical" in many cycles,
            # or just random. Let's try random first.
            e = random.randint(0, NUM_EDGES - 1)
            
            # Find a valid color for e that is different from current
            current_c = colors[e]
            # Try random colors from used_colors first, then unused
            # Actually, changing to a new color is handled by the loop above.
            # Here we just shuffle existing colors.
            
            # Get a list of valid colors for e
            # We can sample a few
            candidate_colors = list(used_colors)
            random.shuffle(candidate_colors)
            
            changed = False
            for c in candidate_colors:
                if c == current_c:
                    continue
                if is_valid_color(e, c, colors):
                    colors[e] = c
                    changed = True
                    break
            
            # If we couldn't change to an existing color, try a random unused color?
            # No, that's what the "new color" loop is for.
            # If we can't even shuffle, we might be stuck.
            # But min(i,j) is very rigid.
            # Let's try to change an edge to a color that is *less* restrictive.
            # Actually, just random valid moves should suffice to explore the space.
            
            if not changed:
                # Try harder: look for any valid color in 0..119
                # This is expensive, so do it rarely
                if random.random() < 0.01:
                    for c in range(MAX_COLOR):
                        if c == current_c:
                            continue
                        if is_valid_color(e, c, colors):
                            colors[e] = c
                            if c not in used_colors:
                                # This would have been caught by can_take_new_color if it was truly new and flexible?
                                # No, can_take_new_color checks if *any* new color works.
                                # If is_valid_color(e, c, colors) is true, and c is new, then can_take_new_color should have been true?
                                # Not necessarily. can_take_new_color requires |S|<3 for ALL cycles.
                                # is_valid_color(e, c, colors) allows c to be in S if |S|==3.
                                # If c is new, it's not in S. So if |S|==3, is_valid_color fails.
                                # So if c is new and valid, then |S|<3 for all cycles.
                                # So can_take_new_color would have returned True.
                                # So this case shouldn't happen if we checked can_take_new_color correctly.
                                pass
                            changed = True
                            break

    return colors
