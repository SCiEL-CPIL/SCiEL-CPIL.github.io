import random
import sys
import time
from collections import defaultdict

N = 16
NUM_COLORS = 120
sys.setrecursionlimit(10000)

def construct():
    # Generate all edges in lexicographic order
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    
    num_edges = len(edges)
    
    # Create edge index lookup
    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
    
    # Pre-compute all 4-cycles
    # A 4-cycle is defined by 4 vertices {a, b, c, d} and a specific ordering (a,b,c,d)
    # The edges in the cycle are (a,b), (b,c), (c,d), (d,a)
    # For each 4-cycle, we store the 4 edge indices
    # We need to check all three cycle structures on every 4-vertex subset:
    # 1. (a,b,c,d) -> edges (a,b), (b,c), (c,d), (d,a)
    # 2. (a,b,d,c) -> edges (a,b), (b,d), (d,c), (c,a)
    # 3. (a,c,b,d) -> edges (a,c), (c,b), (b,d), (d,a)
    
    # Actually, let's think about this more carefully.
    # Given 4 vertices {v1, v2, v3, v4}, the three possible 4-cycles are:
    # Cycle 1: v1-v2-v3-v4-v1 (edges: v1v2, v2v3, v3v4, v4v1)
    # Cycle 2: v1-v2-v4-v3-v1 (edges: v1v2, v2v4, v4v3, v3v1)
    # Cycle 3: v1-v3-v2-v4-v1 (edges: v1v3, v3v2, v2v4, v4v1)
    
    # For each 4-cycle, we have 4 edges. The constraint is that these 4 edges must NOT have 4 distinct colors.
    # This means at least two of the 4 edges must share a color.
    
    # Let's pre-compute all 4-cycles and their edge indices
    four_cycles = []  # List of tuples of 4 edge indices
    
    for v1 in range(N):
        for v2 in range(v1 + 1, N):
            for v3 in range(v2 + 1, N):
                for v4 in range(v3 + 1, N):
                    # Cycle 1: v1-v2-v3-v4-v1
                    try:
                        e1 = edge_idx[(v1, v2)]
                        e2 = edge_idx[(v2, v3)]
                        e3 = edge_idx[(v3, v4)]
                        e4 = edge_idx[(v4, v1)]
                        four_cycles.append((e1, e2, e3, e4))
                    except KeyError:
                        pass
                    
                    # Cycle 2: v1-v2-v4-v3-v1
                    try:
                        e1 = edge_idx[(v1, v2)]
                        e2 = edge_idx[(v2, v4)]
                        e3 = edge_idx[(v4, v3)]
                        e4 = edge_idx[(v3, v1)]
                        four_cycles.append((e1, e2, e3, e4))
                    except KeyError:
                        pass
                    
                    # Cycle 3: v1-v3-v2-v4-v1
                    try:
                        e1 = edge_idx[(v1, v3)]
                        e2 = edge_idx[(v3, v2)]
                        e3 = edge_idx[(v2, v4)]
                        e4 = edge_idx[(v4, v1)]
                        four_cycles.append((e1, e2, e3, e4))
                    except KeyError:
                        pass
    
    # For each edge, pre-compute which 4-cycles it belongs to
    # edge_to_cycles[e_idx] = list of cycle indices
    edge_to_cycles = defaultdict(list)
    for cycle_idx, cycle in enumerate(four_cycles):
        for e in cycle:
            edge_to_cycles[e].append(cycle_idx)
    
    # Pre-compute for each 4-cycle, the other 3 edges for each edge in the cycle
    # For a cycle (e1, e2, e3, e4), for edge e1, the other edges are e2, e3, e4
    # When we assign a color to e1, we need to check if the constraint is violated
    # Constraint: not all 4 edges have distinct colors
    # This is violated only if all 4 edges have distinct colors
    # So when assigning color c to edge e, we need to ensure that among the 4 edges in each cycle containing e,
    # not all 4 have distinct colors
    
    # Let's use a greedy approach with backtracking
    # We'll try to assign colors to edges one by one, trying to maximize the number of distinct colors
    
    best_count = 0
    best_assignment = [0] * num_edges
    
    # Greedy heuristic: for each edge, try colors in an order that favors new colors
    # But we need to check constraints
    
    def is_valid_partial_assignment(colors, assigned_edges):
        """Check if the partial assignment is valid (no 4-cycle has 4 distinct colors)"""
        for cycle in four_cycles:
            e1, e2, e3, e4 = cycle
            # If any edge in the cycle is unassigned, skip
            if e1 not in assigned_edges or e2 not in assigned_edges or e3 not in assigned_edges or e4 not in assigned_edges:
                continue
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            c4 = colors[e4]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True
    
    def get_valid_colors_for_edge(e_idx, colors, assigned_edges):
        """Get valid colors for edge e_idx given current partial assignment"""
        valid = set(range(NUM_COLORS))
        
        # Check all 4-cycles containing this edge
        for cycle_idx in edge_to_cycles[e_idx]:
            cycle = four_cycles[cycle_idx]
            # Find the other 3 edges
            other_edges = [e for e in cycle if e != e_idx]
            
            # If any other edge is unassigned, this cycle doesn't constrain us yet
            if any(e not in assigned_edges for e in other_edges):
                continue
            
            c1, c2, c3 = colors[other_edges[0]], colors[other_edges[1]], colors[other_edges[2]]
            
            # The constraint is that the 4 colors (c_new, c1, c2, c3) must NOT be all distinct
            # So c_new must be in {c1, c2, c3}
            valid.intersection_update({c1, c2, c3})
        
        return valid
    
    # Use randomized greedy with restarts
    start_time = time.time()
    time_limit = 55  # Leave 5 seconds for safety
    
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(attempt * 104729 + 12345)
        
        colors = [-1] * num_edges
        assigned_edges = set()
        used_colors = set()
        
        # Shuffle edge order for diversity
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        success = True
        for e_idx in edge_order:
            valid_colors = get_valid_colors_for_edge(e_idx, colors, assigned_edges)
            
            if not valid_colors:
                success = False
                break
            
            # Prefer new colors to maximize count
            new_colors = [c for c in valid_colors if c not in used_colors]
            old_colors = [c for c in valid_colors if c in used_colors]
            
            if new_colors:
                c = random.choice(new_colors)
            else:
                c = random.choice(old_colors)
            
            colors[e_idx] = c
            assigned_edges.add(e_idx)
            used_colors.add(c)
        
        if success:
            count = len(used_colors)
            if count > best_count:
                best_count = count
                best_assignment = colors[:]
        
        if best_count >= 110:
            break
    
    # Verify the best assignment
    if best_assignment is not None:
        assigned_edges = set(range(num_edges))
        if is_valid_partial_assignment(best_assignment, assigned_edges):
            return best_assignment
    
    # Fallback: all zeros (always valid)
    return [0] * num_edges
