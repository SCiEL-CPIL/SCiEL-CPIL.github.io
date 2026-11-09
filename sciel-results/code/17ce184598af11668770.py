import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = 110  # Safety margin
    
    # Map edges to indices
    edges = []
    edge_index = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = len(edges)
            edge_index[(v, u)] = len(edges)
            edges.append((u, v))
    
    m = len(edges) # 190
    
    # Pre-compute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d
    # Edges: (a,b), (b,c), (c,d), (d,a)
    # We need to consider all permutations that form a simple cycle.
    # Standard form: a-b-c-d-a
    # To avoid duplicates, we can fix a < b < c < d? No, the cycle structure matters.
    # Let's just iterate all combinations of 4 vertices and the 3 possible pairings.
    # For vertices {a,b,c,d}, the 3 distinct 4-cycles are:
    # 1: a-b-c-d-a
    # 2: a-b-d-c-a
    # 3: a-c-b-d-a
    # Note: a-b-c-d-a is the same as d-c-b-a-d etc.
    
    cycles = []
    verts = range(n)
    for combo in itertools.combinations(verts, 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d-a
        try:
            e1 = edge_index[(a, b)]
            e2 = edge_index[(b, c)]
            e3 = edge_index[(c, d)]
            e4 = edge_index[(d, a)]
            cycles.append((e1, e2, e3, e4))
        except KeyError:
            pass
        # Cycle 2: a-b-d-c-a
        try:
            e1 = edge_index[(a, b)]
            e2 = edge_index[(b, d)]
            e3 = edge_index[(d, c)]
            e4 = edge_index[(c, a)]
            cycles.append((e1, e2, e3, e4))
        except KeyError:
            pass
        # Cycle 3: a-c-b-d-a
        try:
            e1 = edge_index[(a, c)]
            e2 = edge_index[(c, b)]
            e3 = edge_index[(b, d)]
            e4 = edge_index[(d, a)]
            cycles.append((e1, e2, e3, e4))
        except KeyError:
            pass
            
    num_cycles = len(cycles)
    
    def count_bad_cycles(colours, present):
        bad = 0
        for cyc in cycles:
            if not (present[cyc[0]] and present[cyc[1]] and present[cyc[2]] and present[cyc[3]]):
                continue
            c0 = colours[cyc[0]]
            c1 = colours[cyc[1]]
            c2 = colours[cyc[2]]
            c3 = colours[cyc[3]]
            if len({c0, c1, c2, c3}) < 4:
                bad += 1
        return bad

    def get_conflict_counts(colours, present):
        conflicts = [0] * m
        for cyc in cycles:
            if not (present[cyc[0]] and present[cyc[1]] and present[cyc[2]] and present[cyc[3]]):
                continue
            c0 = colours[cyc[0]]
            c1 = colours[cyc[1]]
            c2 = colours[cyc[2]]
            c3 = colours[cyc[3]]
            if len({c0, c1, c2, c3}) < 4:
                conflicts[cyc[0]] += 1
                conflicts[cyc[1]] += 1
                conflicts[cyc[2]] += 1
                conflicts[cyc[3]] += 1
        return conflicts

    # State
    present = [True] * m
    colours = [random.randint(0, 19) for _ in range(m)]
    
    best_valid_edges = 0
    best_result = None
    
    # Phase 1: Try to find valid colouring for K20
    # Simulated Annealing
    T = 10.0
    cooling = 0.9995
    
    current_bad = count_bad_cycles(colours, present)
    
    # If we start with 0 bad, we are done
    if current_bad == 0:
        return [[edges[i][0], edges[i][1], colours[i]] for i in range(m)]
        
    while time.time() < deadline:
        # Pick a random edge to change colour
        idx = random.randint(0, m - 1)
        if not present[idx]:
            continue
            
        old_colour = colours[idx]
        new_colour = random.randint(0, 19)
        if old_colour == new_colour:
            continue
            
        # Calculate change in bad cycles
        # Only cycles containing idx are affected
        delta_bad = 0
        # Optimization: Pre-compute which cycles contain which edge?
        # For now, just recount? 14k cycles is fast enough for a few thousand steps.
        # But we need delta.
        
        # Let's just re-evaluate the affected cycles.
        # We need a mapping edge_index -> list of cycle indices containing it.
        # Let's build it once.
        
    # Build edge-to-cycles map
    edge_to_cycles = [[] for _ in range(m)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)
            
    def get_delta_bad(colours, present, edge_idx, new_colour):
        delta = 0
        for cyc_idx in edge_to_cycles[edge_idx]:
            cyc = cycles[cyc_idx]
            # Check if all edges in cycle are present
            if not (present[cyc[0]] and present[cyc[1]] and present[cyc[2]] and present[cyc[3]]):
                continue
                
            # Get current colours
            c_list = [colours[e] for e in cyc]
            # Current state
            is_bad_now = len(set(c_list)) < 4
            
            # Update colour for this edge in c_list
            pos = cyc.index(edge_idx)
            c_list[pos] = new_colour
            is_bad_new = len(set(c_list)) < 4
            
            if is_bad_now and not is_bad_new:
                delta -= 1
            elif not is_bad_now and is_bad_new:
                delta += 1
                
        return delta

    # Restart SA loop with delta optimization
    T = 5.0
    cooling = 0.999
    current_bad = count_bad_cycles(colours, present)
    best_bad = current_bad
    best_colours = colours.copy()
    
    # We will also track the best valid solution if we find one
    found_valid = False
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        idx = random.randint(0, m - 1)
        if not present[idx]:
            continue
            
        new_colour = random.randint(0, 19)
        if colours[idx] == new_colour:
            continue
            
        delta = get_delta_bad(colours, present, idx, new_colour)
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            colours[idx] = new_colour
            current_bad += delta
            if current_bad == 0:
                # Found a valid K20!
                return [[edges[i][0], edges[i][1], colours[i]] for i in range(m)]
            if current_bad < best_bad:
                best_bad = current_bad
                best_colours = colours.copy()
                
        T *= cooling
        
    # Phase 2: If K20 not valid, remove edges
    # We use the best colouring found so far
    colours = best_colours
    current_bad = best_bad
    
    # If still bad, we must remove edges
    # Identify edges with high conflict
    while time.time() < deadline and current_bad > 0:
        conflicts = get_conflict_counts(colours, present)
        # Find the edge with max conflicts among present edges
        max_conf = -1
        max_idx = -1
        for i in range(m):
            if present[i] and conflicts[i] > max_conf:
                max_conf = conflicts[i]
                max_idx = i
                
        if max_idx == -1 or max_conf == 0:
            # No conflicts? Should be 0 bad. If bad>0 but max_conf==0, something is wrong or cycles are not covered?
            # Actually if bad>0, there must be conflicts.
            break
            
        # Remove this edge
        present[max_idx] = False
        current_bad = count_bad_cycles(colours, present)
        
        if current_bad == 0:
            # Valid!
            res = []
            for i in range(m):
                if present[i]:
                    res.append([edges[i][0], edges[i][1], colours[i]])
            return res
            
        # Try to improve colouring with SA on the reduced graph
        # Short SA burst
        T = 2.0
        for _ in range(1000):
            if time.time() > deadline:
                break
            idx = random.randint(0, m - 1)
            if not present[idx]:
                continue
            new_colour = random.randint(0, 19)
            if colours[idx] == new_colour:
                continue
            delta = get_delta_bad(colours, present, idx, new_colour)
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                colours[idx] = new_colour
                current_bad += delta
                if current_bad == 0:
                    res = []
                    for i in range(m):
                        if present[i]:
                            res.append([edges[i][0], edges[i][1], colours[i]])
                    return res
            T *= 0.999

    # If we get here, return the best valid solution found or a fallback
    # We should have returned if we found one.
    # If not, we might have an invalid graph.
    # We must return a valid graph.
    # Let's do a final greedy check:
    # If current_bad > 0, we need to remove more edges until valid.
    # This might be slow, but we have time.
    
    while time.time() < deadline and current_bad > 0:
        conflicts = get_conflict_counts(colours, present)
        max_conf = -1
        max_idx = -1
        for i in range(m):
            if present[i] and conflicts[i] > max_conf:
                max_conf = conflicts[i]
                max_idx = i
        if max_idx == -1:
            break
        present[max_idx] = False
        current_bad = count_bad_cycles(colours, present)
        
    # Final result
    res = []
    for i in range(m):
        if present[i]:
            res.append([edges[i][0], edges[i][1], colours[i]])
            
    # Verify validity one last time
    if count_bad_cycles(colours, present) != 0:
        # Fallback: Star graph
        return [[0, v, 0] for v in range(1, 20)]
        
    return res
