"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def construct():
    n = 20
    num_colors = 20
    
    # Precompute all edges
    edge_list = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_list.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx += 1
    m = len(edge_list) # 190 edges

    # Precompute all 4-cycles
    cycles = []
    edge_to_cycles = defaultdict(list)
    
    vertices = range(n)
    for quad in itertools.combinations(vertices, 4):
        a, b, c, d = quad
        # The 3 distinct cycles (edge sets) on these 4 vertices
        e1 = [(a,b), (b,c), (c,d), (d,a)]
        e2 = [(a,b), (b,d), (d,c), (c,a)]
        e3 = [(a,c), (c,b), (b,d), (d,a)]
        
        for edges in [e1, e2, e3]:
            cyc_idx_set = []
            for e in edges:
                u, v = e
                if u > v:
                    u, v = v, u
                cyc_idx_set.append(edge_to_idx[(u, v)])
            
            cyc_idx = tuple(sorted(cyc_idx_set))
            cycles.append(cyc_idx)
            for e_idx in cyc_idx:
                edge_to_cycles[e_idx].append(len(cycles) - 1)

    num_cycles = len(cycles)
    
    # State variables
    present = [1] * m  # Start with all edges present
    color = [random.randint(0, num_colors - 1) for _ in range(m)]
    
    # Data structures for fast violation tracking
    present_count = [0] * num_cycles
    color_counts = [[0] * num_colors for _ in range(num_cycles)]
    
    # Initialize counts
    for i in range(m):
        if present[i]:
            c = color[i]
            for cyc_idx in edge_to_cycles[i]:
                present_count[cyc_idx] += 1
                color_counts[cyc_idx][c] += 1

    def is_violated(cyc_idx):
        if present_count[cyc_idx] < 4:
            return False
        counts = color_counts[cyc_idx]
        sum_sq = 0
        for cnt in counts:
            if cnt > 0:
                sum_sq += cnt * cnt
        return sum_sq > 4

    violated_cycles = set()
    for i in range(num_cycles):
        if is_violated(i):
            violated_cycles.add(i)
            
    best_score = sum(present) - 1000 * len(violated_cycles)
    best_present = present[:]
    best_color = color[:]
    
    start_time = time.time()
    time_limit = 110.0 # Leave buffer
    
    # Local Search Parameters
    # We use a WalkSAT-like approach.
    # Move: Change color of a random edge, or remove a random edge if many violations.
    
    while time.time() - start_time < time_limit:
        if not violated_cycles:
            # Valid coloring found for current set of edges
            # Try to add an edge? 
            # Since we started with all edges, we can't add more.
            # So we are done if we started with all edges and found a valid coloring.
            # But we know all edges is impossible.
            # So this branch is likely never hit if we start with 190 edges.
            # Unless we removed edges and now want to add back?
            # For simplicity, if we have a valid coloring, we return it.
            # But we want to maximize edges.
            # If we started with 190 and removed some, we have a valid subgraph.
            # We could try to re-add removed edges, but that's complex.
            # Given the time limit, we just return the current valid state.
            break
        
        # Pick a random violated cycle
        cyc_idx = random.choice(list(violated_cycles))
        edges_in_cyc = cycles[cyc_idx]
        
        # Strategy:
        # 1. Try to change the color of one of the edges in the cycle to fix it.
        # 2. If that fails (or with low probability), remove an edge.
        
        fixed = False
        # Try to fix by recoloring
        for e_idx in edges_in_cyc:
            if not present[e_idx]:
                continue
                
            other_colors = []
            for other_e in edges_in_cyc:
                if other_e != e_idx and present[other_e]:
                    other_colors.append(color[other_e])
            
            # Check if other 3 are distinct
            if len(set(other_colors)) < 3:
                continue
                
            available = [c for c in range(num_colors) if c not in other_colors]
            if available:
                old_color = color[e_idx]
                new_color = random.choice(available)
                
                # Apply move
                color[e_idx] = new_color
                fixed = True
                
                # Update data structures
                for c_idx in edge_to_cycles[e_idx]:
                    color_counts[c_idx][old_color] -= 1
                    color_counts[c_idx][new_color] += 1
                    
                    if c_idx in violated_cycles:
                        if not is_violated(c_idx):
                            violated_cycles.remove(c_idx)
                    else:
                        if is_violated(c_idx):
                            violated_cycles.add(c_idx)
                break
        
        if not fixed:
            # Recoloring didn't work or wasn't possible. Remove an edge.
            # Pick the edge in the cycle that is part of the most other violations?
            # Or just random.
            e_idx = random.choice([e for e in edges_in_cyc if present[e]])
            present[e_idx] = 0
            old_color = color[e_idx]
            
            for c_idx in edge_to_cycles[e_idx]:
                present_count[c_idx] -= 1
                color_counts[c_idx][old_color] -= 1
                
                if c_idx in violated_cycles:
                    if not is_violated(c_idx):
                        violated_cycles.remove(c_idx)
                else:
                    if is_violated(c_idx):
                        violated_cycles.add(c_idx)
            
        # Update best
        current_score = sum(present) - 1000 * len(violated_cycles)
        if current_score > best_score:
            best_score = current_score
            best_present = present[:]
            best_color = color[:]

    # Extract result
    result = []
    for i in range(m):
        if best_present[i]:
            u, v = edge_list[i]
            c = best_color[i]
            result.append([u, v, c])
            
    return result

# EVOLVE-BLOCK-END
