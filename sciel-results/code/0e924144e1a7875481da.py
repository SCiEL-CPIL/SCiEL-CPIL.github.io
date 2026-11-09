import random
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    n = 20
    num_colors = 20
    deadline = time.time() + 110  # Safety margin
    
    # 1. Generate all 4-cycles in K20
    # A 4-cycle is a set of 4 vertices. In K_n, for each set of 4 vertices, 
    # there are 3 distinct 4-cycle subgraphs.
    # We represent a 4-cycle by its 4 edges.
    # Edges are represented as (min(u,v), max(u,v))
    
    all_edges = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            edge_to_idx[(u, v)] = len(all_edges) - 1
            
    E = len(all_edges) # 190
    
    # Precompute 4-cycles
    # For each combination of 4 vertices, there are 3 cycles.
    # Let vertices be a, b, c, d.
    # Cycle 1: (a,b), (b,c), (c,d), (d,a)
    # Cycle 2: (a,b), (b,d), (d,c), (c,a)
    # Cycle 3: (a,c), (c,b), (b,d), (d,a)
    
    cycles = [] # List of lists of edge indices
    edge_to_cycles = defaultdict(list) # edge index -> list of cycle indices
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # 3 cycles
                    # 1: a-b-c-d-a
                    e1 = edge_to_idx[(a, b)]
                    e2 = edge_to_idx[(b, c)]
                    e3 = edge_to_idx[(c, d)]
                    e4 = edge_to_idx[(a, d)]
                    cid = len(cycles)
                    cycles.append([e1, e2, e3, e4])
                    edge_to_cycles[e1].append(cid)
                    edge_to_cycles[e2].append(cid)
                    edge_to_cycles[e3].append(cid)
                    edge_to_cycles[e4].append(cid)
                    
                    # 2: a-b-d-c-a
                    e1 = edge_to_idx[(a, b)]
                    e2 = edge_to_idx[(b, d)]
                    e3 = edge_to_idx[(c, d)]
                    e4 = edge_to_idx[(a, c)]
                    cid = len(cycles)
                    cycles.append([e1, e2, e3, e4])
                    edge_to_cycles[e1].append(cid)
                    edge_to_cycles[e2].append(cid)
                    edge_to_cycles[e3].append(cid)
                    edge_to_cycles[e4].append(cid)
                    
                    # 3: a-c-b-d-a
                    e1 = edge_to_idx[(a, c)]
                    e2 = edge_to_idx[(b, c)]
                    e3 = edge_to_idx[(b, d)]
                    e4 = edge_to_idx[(a, d)]
                    cid = len(cycles)
                    cycles.append([e1, e2, e3, e4])
                    edge_to_cycles[e1].append(cid)
                    edge_to_cycles[e2].append(cid)
                    edge_to_cycles[e3].append(cid)
                    edge_to_cycles[e4].append(cid)
    
    C = len(cycles)
    
    # Initialize colors
    colors = [random.randint(0, num_colors - 1) for _ in range(E)]
    
    # Count bad cycles
    # bad_count[cid] = 1 if cycle cid has duplicate colors, 0 otherwise
    # We just need the total number of bad cycles.
    # To update efficiently, we need to know if a specific cycle is bad.
    # A cycle is bad if any two edges have the same color.
    
    def is_bad_cycle(cid):
        e_idx = cycles[cid]
        c0 = colors[e_idx[0]]
        c1 = colors[e_idx[1]]
        c2 = colors[e_idx[2]]
        c3 = colors[e_idx[3]]
        # Check for duplicates
        if c0 == c1 or c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3 or c2 == c3:
            return 1
        return 0

    bad_counts = [0] * C
    total_bad = 0
    for cid in range(C):
        bad_counts[cid] = is_bad_cycle(cid)
        total_bad += bad_counts[cid]
        
    best_colors = colors[:]
    best_total_bad = total_bad
    best_edges = E
    
    if total_bad == 0:
        # Already perfect
        return [[all_edges[i][0], all_edges[i][1], colors[i]] for i in range(E)]
        
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    
    current_bad = total_bad
    current_colors = colors[:]
    
    while T > T_min and time.time() < deadline - 10:
        # Pick random edge
        e_idx = random.randint(0, E - 1)
        old_color = current_colors[e_idx]
        new_color = random.randint(0, num_colors - 1)
        
        if new_color == old_color:
            continue
            
        # Calculate delta
        delta = 0
        affected_cycles = edge_to_cycles[e_idx]
        
        # Store old state of affected cycles to revert if rejected
        # We can just re-evaluate them, or store old bad status.
        # Storing is faster.
        old_statuses = {}
        for cid in affected_cycles:
            old_statuses[cid] = bad_counts[cid]
            
        # Apply change temporarily
        current_colors[e_idx] = new_color
        
        for cid in affected_cycles:
            new_status = is_bad_cycle(cid)
            if new_status != bad_counts[cid]:
                delta += (new_status - bad_counts[cid])
                bad_counts[cid] = new_status
                current_bad += (new_status - bad_counts[cid] - (new_status - old_statuses[cid])) # Wait, simpler:
                # Actually, let's just recalculate current_bad properly.
                
        # My delta calculation above is messy. Let's do it cleanly.
        # Revert and do it properly.
        
        # REVERT
        current_colors[e_idx] = old_color
        for cid in affected_cycles:
            bad_counts[cid] = old_statuses[cid]
            
        # REAPPLY and CALCULATE DELTA
        current_colors[e_idx] = new_color
        new_current_bad = current_bad
        for cid in affected_cycles:
            new_status = is_bad_cycle(cid)
            if new_status != bad_counts[cid]:
                new_current_bad += (new_status - bad_counts[cid])
                bad_counts[cid] = new_status
                # Note: we are updating bad_counts in place for the "new" state
                # but we need to revert if rejected.
        
        delta = new_current_bad - current_bad
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_bad = new_current_bad
            # bad_counts is already updated to new state
            if current_bad < best_total_bad:
                best_total_bad = current_bad
                best_colors = current_colors[:]
                best_edges = E
                if current_bad == 0:
                    break
        else:
            # Reject
            # Revert bad_counts
            for cid in affected_cycles:
                bad_counts[cid] = old_statuses[cid]
            current_colors[e_idx] = old_color
            
        T *= alpha
        
    # Check if we found a perfect coloring
    if best_total_bad == 0:
        return [[all_edges[i][0], all_edges[i][1], best_colors[i]] for i in range(E)]
        
    # If not, we have a partial solution with some bad cycles.
    # We need to remove edges to make it valid.
    # Greedy removal:
    # We use the best_colors found.
    colors = best_colors[:]
    
    # Re-initialize bad counts for the best solution
    bad_counts = [0] * C
    total_bad = 0
    for cid in range(C):
        bad_counts[cid] = is_bad_cycle(cid)
        total_bad += bad_counts[cid]
        
    active_edges = [True] * E
    
    # If we are close to deadline, just return what we have? 
    # No, we must return a VALID graph.
    # So we must repair.
    
    # To speed up repair, we can identify edges in many bad cycles.
    # edge_bad_count[e] = number of bad cycles containing e
    edge_bad_count = [0] * E
    for cid in range(C):
        if bad_counts[cid] == 1:
            for e_idx in cycles[cid]:
                edge_bad_count[e_idx] += 1
                
    # Remove edges with highest bad count
    # Repeat until total_bad == 0
    # This might be slow if we do it one by one.
    # But 190 edges is small.
    
    # We can batch remove? No, removing one edge changes badness of many cycles.
    
    while total_bad > 0 and time.time() < deadline - 2:
        # Find active edge with max bad count
        max_bad = -1
        max_edge = -1
        for i in range(E):
            if active_edges[i] and edge_bad_count[i] > max_bad:
                max_bad = edge_bad_count[i]
                max_edge = i
                
        if max_edge == -1:
            # No active edges? Should not happen if graph is simple and we start with K20
            break
            
        # Remove max_edge
        active_edges[max_edge] = False
        
        # Update bad counts for cycles containing max_edge
        for cid in edge_to_cycles[max_edge]:
            if bad_counts[cid] == 1:
                # It was bad. Is it still bad?
                # We need to check if the remaining 3 edges have distinct colors.
                # Actually, we just re-evaluate the cycle.
                new_status = is_bad_cycle(cid)
                if new_status != bad_counts[cid]:
                    total_bad += (new_status - bad_counts[cid])
                    bad_counts[cid] = new_status
                    # Update edge_bad_counts for edges in this cycle
                    # This is getting complex to maintain incrementally.
                    # Given small size, maybe just recalculate edge_bad_count from scratch?
                    # 14535 cycles * 4 edges = ~60k ops. Very fast.
        
        # Recalculate edge_bad_count from scratch for simplicity and correctness
        # This is O(C*4) ~ 60k ops. 
        # If we do this for each edge removal (up to 190 times), it's ~12M ops.
        # 12M ops in Python might take a second or two. 
        # Deadline is 110s. This is safe.
        
        edge_bad_count = [0] * E
        for cid in range(C):
            if bad_counts[cid] == 1:
                for e_idx in cycles[cid]:
                    if active_edges[e_idx]:
                        edge_bad_count[e_idx] += 1

    # Construct result
    result = []
    for i in range(E):
        if active_edges[i]:
            u, v = all_edges[i]
            c = colors[i]
            result.append([u, v, c])
            
    return result

# EVOLVE-BLOCK-END
