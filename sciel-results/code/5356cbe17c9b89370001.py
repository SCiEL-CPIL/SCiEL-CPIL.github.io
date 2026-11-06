import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0  # 120 seconds limit
    
    n = 20
    # Vertices 0..9 are L, 10..19 are R
    L = list(range(10))
    R = list(range(10, 20))
    
    # Initialize with all edges of K_{10,10}
    # Edge is (u, v) with u < v
    # Color function for cross edges
    def get_color(u, v):
        # Map to 0..9
        u_idx = u if u < 10 else u - 10
        v_idx = v if v < 10 else v - 10
        
        # Determine which is L and which is R
        if u < 10 and v >= 10:
            l_idx, r_idx = u_idx, v_idx
        elif v < 10 and u >= 10:
            l_idx, r_idx = v_idx, u_idx
        else:
            # Internal edge, handle separately or return -1
            return -1
            
        return (r_idx - l_idx) % 10

    # List of all potential cross edges
    all_cross_edges = []
    for u in L:
        for v in R:
            all_cross_edges.append((u, v))
            
    # Current set of edges (frozenset of (u,v) or set of tuples)
    current_edges = set(all_cross_edges)
    
    # Precompute all 4-cycles in K_{10,10}
    # A 4-cycle is defined by l1, l2 in L and r1, r2 in R
    # Cycle: l1 - r1 - l2 - r2 - l1
    # Edges: (l1, r1), (r1, l2), (l2, r2), (r2, l1)
    # Note: (r1, l2) is same as (l2, r1)
    
    cycles = []
    for i, l1 in enumerate(L):
        for l2 in L[i+1:]:
            for j, r1 in enumerate(R):
                for r2 in R[j+1:]:
                    # Edges
                    e1 = (l1, r1)
                    e2 = (l2, r1) # (r1, l2)
                    e3 = (l2, r2)
                    e4 = (l1, r2) # (r2, l1)
                    
                    # Check if all 4 edges exist in current graph?
                    # We will check dynamically.
                    cycles.append([e1, e2, e3, e4])
                    
    # To speed up, we can maintain a list of "bad" cycles
    # A cycle is bad if any of its edges are missing? No, bad if colors conflict.
    # Colors are fixed. So a cycle is "structurally bad" if it exists and colors conflict.
    # If an edge is removed, the cycle disappears (no longer a subgraph).
    # So we only need to consider cycles where all 4 edges are present.
    
    # Let's identify which cycles are "color-bad" (conflict)
    # and which are "color-good".
    # If a cycle is color-good, we don't care if it exists.
    # If a cycle is color-bad, we must remove at least one edge.
    
    bad_cycles = []
    for cyc in cycles:
        c1 = get_color(cyc[0][0], cyc[0][1])
        c2 = get_color(cyc[1][0], cyc[1][1])
        c3 = get_color(cyc[2][0], cyc[2][1])
        c4 = get_color(cyc[3][0], cyc[3][1])
        
        # Check distinctness
        if len({c1, c2, c3, c4}) == 4:
            continue # Good cycle
        
        bad_cycles.append(cyc)
        
    # We need to remove edges to break all bad_cycles.
    # Greedy: while bad_cycles is not empty:
    #   Pick a random bad cycle (or one with max edge frequency)
    #   Remove an edge from it
    #   Update bad_cycles: remove any cycle containing the removed edge
    
    # To be efficient, we can maintain a list of active bad cycles.
    # And a mapping from edge to list of cycle indices containing it.
    
    edge_to_cycles = {}
    for idx, cyc in enumerate(bad_cycles):
        for e in cyc:
            if e not in edge_to_cycles:
                edge_to_cycles[e] = []
            edge_to_cycles[e].append(idx)
            
    active_bad_cycles = list(range(len(bad_cycles)))
    
    # Greedy removal
    while active_bad_cycles and (time.time() - start_time) < deadline - 5.0:
        # Pick a cycle
        # Heuristic: Pick the cycle with an edge that has the highest degree in the conflict graph?
        # Or just random.
        # Let's try picking the cycle with the smallest number of available edges?
        # Or just random to avoid bias.
        
        # Let's pick a random active bad cycle
        idx = random.choice(active_bad_cycles)
        cyc = bad_cycles[idx]
        
        # Find edges in this cycle that are still present
        present_edges = [e for e in cyc if e in current_edges]
        
        if not present_edges:
            # This cycle is already broken
            active_bad_cycles.remove(idx)
            continue
            
        # Pick an edge to remove
        # Heuristic: Remove the edge that appears in the most other active bad cycles
        max_count = -1
        chosen_edge = None
        for e in present_edges:
            # Count how many active bad cycles contain e
            # This is expensive if we do it naively.
            # We can approximate or just pick random.
            # Let's pick random for simplicity and speed.
            pass
            
        # Let's stick to random edge from present_edges
        e_to_remove = random.choice(present_edges)
        
        # Remove it
        current_edges.remove(e_to_remove)
        
        # Update active_bad_cycles
        # Any cycle containing e_to_remove is now broken (or at least, we don't need to worry about it for this specific conflict?
        # No, if we remove an edge, the cycle is gone. So it's no longer a 4-cycle in the graph.
        # So we can mark all cycles containing e_to_remove as inactive.
        
        cycles_to_remove = set(edge_to_cycles.get(e_to_remove, []))
        
        # Remove these from active list
        # This is O(N) if we use list.
        # Let's use a set for active cycles?
        # active_bad_cycles is a list.
        # Let's just filter.
        
        new_active = []
        for c_idx in active_bad_cycles:
            if c_idx not in cycles_to_remove:
                new_active.append(c_idx)
            # If c_idx is in cycles_to_remove, it's broken.
            
        active_bad_cycles = new_active
        
    # Construct result
    result = []
    for u, v in current_edges:
        c = get_color(u, v)
        result.append([u, v, c])
        
    return result
