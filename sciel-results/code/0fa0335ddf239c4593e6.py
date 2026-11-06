import time
import random
import itertools

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    E = len(all_edges)  # 190
    
    # Map edge to index
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i

    # Precompute all 4-cycles
    # A 4-cycle is a sequence of 4 distinct vertices a, b, c, d such that
    # edges (a,b), (b,c), (c,d), (d,a) exist.
    # We represent a cycle by the tuple of 4 edge indices: (e_ab, e_bc, e_cd, e_da)
    # We need to ensure we don't double count. 
    # A set of 4 vertices {a,b,c,d} has 3 distinct 4-cycles (up to rotation/reflection):
    # 1. a-b-c-d-a
    # 2. a-b-d-c-a
    # 3. a-c-b-d-a
    # Note: a-c-d-b-a is the same as a-b-d-c-a reversed.
    
    cycles = []
    edge_to_cycles = [[] for _ in range(E)]
    
    vertices = list(range(n))
    
    # Iterate over all combinations of 4 vertices
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        # The three distinct cycles on these 4 vertices
        # Cycle 1: a-b-c-d-a
        e1 = edge_to_idx[(a, b)]
        e2 = edge_to_idx[(b, c)]
        e3 = edge_to_idx[(c, d)]
        e4 = edge_to_idx[(d, a)]
        cycle1 = (e1, e2, e3, e4)
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_to_idx[(a, b)]
        e2 = edge_to_idx[(b, d)]
        e3 = edge_to_idx[(d, c)]
        e4 = edge_to_idx[(c, a)]
        cycle2 = (e1, e2, e3, e4)
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_to_idx[(a, c)]
        e2 = edge_to_idx[(c, b)]
        e3 = edge_to_idx[(b, d)]
        e4 = edge_to_idx[(d, a)]
        cycle3 = (e1, e2, e3, e4)
        
        for cyc in [cycle1, cycle2, cycle3]:
            cycles.append(cyc)
            for e_idx in cyc:
                edge_to_cycles[e_idx].append(len(cycles) - 1)
                
    num_cycles = len(cycles)
    
    # State variables
    best_edge_count = 0
    best_edge_set = None
    best_colors = None

    # Greedy construction with random restarts and backtracking-ish heuristics
    # We will use a strategy of adding edges one by one.
    # For each edge, we check if adding it with any color violates any 4-cycle.
    # A 4-cycle is violated if it becomes monochromatic, or has < 4 distinct colors?
    # No, "rainbow" means 4 distinct colors.
    # So if a 4-cycle has 3 edges already colored, and they are not all distinct, it's already broken?
    # No, we must ensure that EVERY 4-cycle in the final graph is rainbow.
    # This implies that if we add an edge and complete a 4-cycle, that cycle MUST be rainbow.
    # If a 4-cycle is not yet complete, we just need to ensure that the partial coloring
    # doesn't make it IMPOSSIBLE to be rainbow in the future.
    # A partial coloring of a 4-cycle is "viable" if the existing colors are distinct.
    # If two existing edges in a 4-cycle have the same color, that cycle can never be rainbow.
    # So, when adding an edge (u,v) with color c:
    # 1. For every 4-cycle containing (u,v):
    #    - If the other 3 edges are present:
    #      - Their colors must be distinct from each other AND from c.
    #    - If 2 other edges are present:
    #      - Their colors must be distinct from each other AND from c.
    #      - (The 4th edge is not present yet, but its color will be chosen later.
    #       We just need the 3 present colors to be distinct so that the 4th can be different.)
    #    - If 1 other edge is present:
    #      - Its color must be different from c.
    #    - If 0 other edges are present:
    #      - No constraint.
    
    # This check is local to the edge being added.
    
    # Optimization:
    # Instead of checking all cycles, we can precompute for each edge, the list of cycles it belongs to.
    # When adding edge e with color c:
    #   For each cycle in edge_to_cycles[e]:
    #     Let other_edges = [e1, e2, e3] in the cycle excluding e.
    #     Count how many of other_edges are in the current graph.
    #     If count == 3:
    #        Check if colors of those 3 are distinct and != c.
    #     If count == 2:
    #        Check if colors of those 2 are distinct and != c.
    #     If count == 1:
    #        Check if color of that 1 != c.
    
    # To make this faster, we can maintain for each cycle a count of colored edges and a set of colors used.
    # But updating these for every edge addition is O(degree of edge in hypergraph of cycles).
    # Each edge is in ~ O(n^2) cycles?
    # Number of 4-cycles containing a specific edge (u,v):
    # Choose 2 other vertices w, x from remaining 18.
    # Cycles: u-v-w-x-u, u-v-x-w-u.
    # So for each pair (w,x), there are 2 cycles.
    # Number of pairs = C(18,2) = 153.
    # So each edge is in ~306 cycles.
    # Total cycles = 190 * 306 / 4 = 14535. (Matches C(20,4)*3 = 4845*3 = 14535).
    
    # So for each edge addition, we check ~306 cycles.
    # For each cycle, we check up to 3 other edges.
    # This is fast enough.
    
    # We will try multiple random restarts.
    # In each restart, we shuffle the order of edges.
    # We greedily add edges if there exists a valid color.
    # To improve density, we can try to pick colors that are "most constrained" or "least constrained"?
    # Usually, picking a random valid color works well with restarts.
    # However, we can also try to prioritize edges that are in many cycles (high degree in line graph of cycles).
    # Or just random order.
    
    # Let's implement the greedy algorithm with the cycle check.
    
    def is_valid_addition(edge_idx, color, edge_color_map, edge_present):
        """
        Check if adding edge `edge_idx` with `color` is valid.
        edge_color_map: dict edge_idx -> color
        edge_present: set of edge_idx present
        """
        for cyc_idx in edge_to_cycles[edge_idx]:
            cycle = cycles[cyc_idx]
            # Find the other 3 edges
            # cycle is (e0, e1, e2, e3). One of them is edge_idx.
            # We need to check the other 3.
            colors_in_cycle = []
            count_present = 0
            for e in cycle:
                if e == edge_idx:
                    continue
                if e in edge_present:
                    count_present += 1
                    colors_in_cycle.append(edge_color_map[e])
            
            if count_present == 3:
                # All 3 others are present.
                # The cycle will be complete.
                # All 4 colors must be distinct.
                if len(set(colors_in_cycle + [color])) != 4:
                    return False
            elif count_present == 2:
                # 2 others present.
                # The 3 colors (2 existing + new) must be distinct.
                # The 4th edge is not present, so we can always pick a 4th color later?
                # Yes, as long as the 3 present are distinct.
                if len(set(colors_in_cycle + [color])) != 3:
                    return False
            elif count_present == 1:
                # 1 other present.
                # The 2 colors must be distinct.
                if color in colors_in_cycle:
                    return False
            # count_present == 0: no constraint
            
        return True

    max_restarts = 5000
    restart = 0
    
    while restart < max_restarts:
        if time.time() > deadline - 2:
            break
            
        restart += 1
        
        # Random order of edges
        edge_order = list(range(E))
        random.shuffle(edge_order)
        
        edge_present = set()
        edge_color_map = {}
        
        for e_idx in edge_order:
            # Find a valid color
            valid_colors = []
            for c in range(num_colors):
                if is_valid_addition(e_idx, c, edge_color_map, edge_present):
                    valid_colors.append(c)
            
            if valid_colors:
                # Pick a random valid color
                # Heuristic: pick the color that appears least often in the graph so far?
                # Or just random.
                c = random.choice(valid_colors)
                edge_present.add(e_idx)
                edge_color_map[e_idx] = c
        
        if len(edge_present) > best_edge_count:
            best_edge_count = len(edge_present)
            best_edge_set = set(edge_present)
            best_colors = dict(edge_color_map)
            
            # Early stopping if we reach a very high density?
            # Max edges is 190.
            # If we get close to 190, we can stop.
            if best_edge_count >= 180:
                break

    # Convert result to required format
    if best_edge_set is None:
        return []
        
    result = []
    for e_idx in best_edge_set:
        u, v = all_edges[e_idx]
        c = best_colors[e_idx]
        result.append([u, v, c])
        
    return result

# EVOLVE-BLOCK-START
# (The function above is the implementation)
# EVOLVE-BLOCK-END
