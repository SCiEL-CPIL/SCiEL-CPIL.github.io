<<<<<<< SEARCH
import time
import random
import itertools

# Precomputation for n=20
N = 20
# Generate all edges
_all_edges = []
_edge_index = {}
for u in range(N):
    for v in range(u + 1, N):
        idx = len(_all_edges)
        _all_edges.append((u, v))
        _edge_index[(u, v)] = idx
        _edge_index[(v, u)] = idx

E = len(_all_edges) # 190

# Generate all 4-cycles
# A 4-cycle is defined by 4 distinct vertices.
# For 4 vertices {a,b,c,d}, there are 3 distinct 4-cycles (up to rotation/reflection).
# 1: a-b-c-d-a
# 2: a-b-d-c-a
# 3: a-c-b-d-a
# We need to map these to edge indices.

_cycles = []
_edge_to_cycles = [[] for _ in range(E)]

def get_edge_idx(u, v):
    return _edge_index[(u, v)]

for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Cycle 1: a-b-c-d-a
    e1 = get_edge_idx(a, b)
    e2 = get_edge_idx(b, c)
    e3 = get_edge_idx(c, d)
    e4 = get_edge_idx(d, a)
    cid = len(_cycles)
    _cycles.append((e1, e2, e3, e4))
    _edge_to_cycles[e1].append(cid)
    _edge_to_cycles[e2].append(cid)
    _edge_to_cycles[e3].append(cid)
    _edge_to_cycles[e4].append(cid)
    
    # Cycle 2: a-b-d-c-a
    e1 = get_edge_idx(a, b)
    e2 = get_edge_idx(b, d)
    e3 = get_edge_idx(d, c)
    e4 = get_edge_idx(c, a)
    cid = len(_cycles)
    _cycles.append((e1, e2, e3, e4))
    _edge_to_cycles[e1].append(cid)
    _edge_to_cycles[e2].append(cid)
    _edge_to_cycles[e3].append(cid)
    _edge_to_cycles[e4].append(cid)
    
    # Cycle 3: a-c-b-d-a
    e1 = get_edge_idx(a, c)
    e2 = get_edge_idx(c, b)
    e3 = get_edge_idx(b, d)
    e4 = get_edge_idx(d, a)
    cid = len(_cycles)
    _cycles.append((e1, e2, e3, e4))
    _edge_to_cycles[e1].append(cid)
    _edge_to_cycles[e2].append(cid)
    _edge_to_cycles[e3].append(cid)
    _edge_to_cycles[e4].append(cid)

C = len(_cycles)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initialize colors randomly
    colors = [random.randint(0, 19) for _ in range(E)]
    
    # is_bad[cid] = True if the 4 edges in cycle cid do not have distinct colors
    is_bad = [False] * C
    bad_count = 0
    
    def check_cycle(cid):
        e1, e2, e3, e4 = _cycles[cid]
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        # Check if all distinct
        if len({c1, c2, c3, c4}) == 4:
            return False
        return True

    # Initial computation
    for cid in range(C):
        if check_cycle(cid):
            is_bad[cid] = True
            bad_count += 1
            
    # Simulated Annealing
    T = 1.0
    cooling = 0.99999
    min_T = 0.01
    
    # If already perfect, return
    if bad_count == 0:
        return [[u, v, colors[i]] for i, (u, v) in enumerate(_all_edges)]
        
    while time.time() < deadline - 5:  # Leave 5s for cleanup
        e_idx = random.randint(0, E - 1)
        old_c = colors[e_idx]
        
        # Pick a new color different from old_c
        new_c = random.randint(0, 19)
        while new_c == old_c:
            new_c = random.randint(0, 19)
            
        # Calculate delta
        delta = 0
        for cid in _edge_to_cycles[e_idx]:
            was_bad = is_bad[cid]
            # Simulate change
            e1, e2, e3, e4 = _cycles[cid]
            c1 = colors[e1] if e1 != e_idx else new_c
            c2 = colors[e2] if e2 != e_idx else new_c
            c3 = colors[e3] if e3 != e_idx else new_c
            c4 = colors[e4] if e4 != e_idx else new_c
            
            is_bad_now = (len({c1, c2, c3, c4}) < 4)
            
            if was_bad and not is_bad_now:
                delta -= 1
            elif not was_bad and is_bad_now:
                delta += 1
            
        # Acceptance criterion
        if delta < 0:
            accept = True
        else:
            # Avoid division by zero or overflow
            if T < 1e-9:
                accept = False
            else:
                accept = random.random() < pow(2.718281828, -delta / T)
                
        if accept:
            colors[e_idx] = new_c
            # Update is_bad and bad_count
            for cid in _edge_to_cycles[e_idx]:
                new_state = check_cycle(cid)
                if new_state != is_bad[cid]:
                    if new_state:
                        bad_count += 1
                    else:
                        bad_count -= 1
                    is_bad[cid] = new_state
                    
            if bad_count == 0:
                break
                
        T *= cooling
        if T < min_T:
            T = min_T

    # If bad_count is 0, we are done
    if bad_count == 0:
        return [[u, v, colors[i]] for i, (u, v) in enumerate(_all_edges)]
        
    # Greedy Edge Removal
    # We need to remove edges to eliminate all bad cycles.
    # We maintain a set of active edges.
    active_edges = set(range(E))
    
    # List of bad cycle IDs
    bad_cycles = [cid for cid in range(C) if is_bad[cid]]
    
    # Map edge -> list of bad cycles it is in
    # This needs to be updated as we remove edges/cycles
    # To keep it simple and fast, we just re-scan bad_cycles each time if the list is small.
    # If the list is large, we might be in trouble, but SA should have reduced it.
    
    while bad_cycles and time.time() < deadline - 1:
        # Count frequency of each edge in bad_cycles
        edge_counts = {}
        for cid in bad_cycles:
            for e in _cycles[cid]:
                if e in active_edges:
                    edge_counts[e] = edge_counts.get(e, 0) + 1
                else:
                    # If an edge is already removed, the cycle is technically broken?
                    # No, a cycle exists only if all 4 edges exist.
                    # So if we remove an edge, all cycles containing it are no longer "cycles" in the graph.
                    # Thus, we should remove them from bad_cycles.
                    pass
        
        if not edge_counts:
            # All bad cycles contain at least one removed edge?
            # This shouldn't happen if we manage bad_cycles correctly.
            # But if we just removed an edge, we should have filtered bad_cycles.
            break
            
        # Find edge with max count
        max_count = -1
        best_edge = -1
        for e, count in edge_counts.items():
            if count > max_count:
                max_count = count
                best_edge = e
                
        if best_edge == -1:
            break
            
        # Remove best_edge
        active_edges.remove(best_edge)
        
        # Update bad_cycles: remove any cycle containing best_edge
        new_bad_cycles = []
        for cid in bad_cycles:
            if best_edge in _cycles[cid]:
                # Cycle is broken
                pass
            else:
                new_bad_cycles.append(cid)
        bad_cycles = new_bad_cycles

    # Construct result from active_edges
    result = []
    for e in active_edges:
        u, v = _all_edges[e]
        result.append([u, v, colors[e]])
        
    return result

# EVOLVE-BLOCK-END
=======
import time
import random
import itertools
import sys

# Precomputation for n=20
N = 20
# Generate all edges
_all_edges = []
_edge_index = {}
for u in range(N):
    for v in range(u + 1, N):
        idx = len(_all_edges)
        _all_edges.append((u, v))
        _edge_index[(u, v)] = idx
        _edge_index[(v, u)] = idx

E = len(_all_edges) # 190

# Generate all 4-cycles
_cycles = []
_edge_to_cycles = [[] for _ in range(E)]

def get_edge_idx(u, v):
    return _edge_index[(u, v)]

for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Cycle 1: a-b-c-d-a
    e1 = get_edge_idx(a, b)
    e2 = get_edge_idx(b, c)
    e3 = get_edge_idx(c, d)
    e4 = get_edge_idx(d, a)
    cid = len(_cycles)
    _cycles.append((e1, e2, e3, e4))
    _edge_to_cycles[e1].append(cid)
    _edge_to_cycles[e2].append(cid)
    _edge_to_cycles[e3].append(cid)
    _edge_to_cycles[e4].append(cid)
    
    # Cycle 2: a-b-d-c-a
    e1 = get_edge_idx(a, b)
    e2 = get_edge_idx(b, d)
    e3 = get_edge_idx(d, c)
    e4 = get_edge_idx(c, a)
    cid = len(_cycles)
    _cycles.append((e1, e2, e3, e4))
    _edge_to_cycles[e1].append(cid)
    _edge_to_cycles[e2].append(cid)
    _edge_to_cycles[e3].append(cid)
    _edge_to_cycles[e4].append(cid)
    
    # Cycle 3: a-c-b-d-a
    e1 = get_edge_idx(a, c)
    e2 = get_edge_idx(c, b)
    e3 = get_edge_idx(b, d)
    e4 = get_edge_idx(d, a)
    cid = len(_cycles)
    _cycles.append((e1, e2, e3, e4))
    _edge_to_cycles[e1].append(cid)
    _edge_to_cycles[e2].append(cid)
    _edge_to_cycles[e3].append(cid)
    _edge_to_cycles[e4].append(cid)

C = len(_cycles)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit
    
    # Strategy: Start with a dense random subgraph, then use CSP backtracking
    # to find valid colorings. If we can't color the full graph, we remove edges.
    
    # Generate multiple candidate edge sets (subgraphs) and try to color them
    best_result = None
    best_edge_count = 0
    
    # Try different densities of subgraphs
    # We'll use a random edge selection approach with CSP coloring
    
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        # Create a random subgraph by including edges with some probability
        # Start with high density, decrease if coloring fails
        # Use a structured approach: include all edges, then remove problematic ones
        
        # Start with all edges
        active_edges = set(range(E))
        
        # Try to color using CSP with backtracking
        colors = [-1] * E  # -1 means uncolored
        is_bad = [False] * C
        bad_count = 0
        
        # Order edges by degree (number of cycles they participate in) for better pruning
        edge_order = sorted(range(E), key=lambda e: len(_edge_to_cycles[e]), reverse=True)
        
        # Assign colors greedily with backtracking
        # We'll use a simplified approach: assign colors in order, backtrack if stuck
        
        def try_coloring(active_edge_set, max_backtracks=100000):
            nonlocal colors, is_bad, bad_count
            
            # Reset state
            colors = [-1] * E
            is_bad = [False] * C
            bad_count = 0
            
            # For each active edge, determine which colors are forbidden by already-colored neighbors
            # In the line graph, two edges are adjacent if they share a vertex
            # But our constraint is on 4-cycles, not direct adjacency
            
            # Simpler approach: random assignment with local search
            # Assign random colors to all active edges
            for e in active_edge_set:
                colors[e] = random.randint(0, 19)
            
            # Count bad cycles
            for cid in range(C):
                e1, e2, e3, e4 = _cycles[cid]
                if e1 in active_edge_set and e2 in active_edge_set and e3 in active_edge_set and e4 in active_edge_set:
                    c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                    if len({c1, c2, c3, c4}) < 4:
                        is_bad[cid] = True
                        bad_count += 1
            
            if bad_count == 0:
                return True
            
            # Local search: try to fix bad cycles by changing colors
            # Use simulated annealing on the bad cycles
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            for step in range(50000):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random bad cycle
                bad_cids = [cid for cid in range(C) if is_bad[cid]]
                if not bad_cids:
                    return True
                
                cid = random.choice(bad_cids)
                e1, e2, e3, e4 = _cycles[cid]
                
                # Try changing one of the four edges' colors
                e_idx = random.choice([e1, e2, e3, e4])
                old_c = colors[e_idx]
                new_c = random.randint(0, 19)
                while new_c == old_c:
                    new_c = random.randint(0, 19)
                
                # Calculate delta
                delta = 0
                for cid2 in _edge_to_cycles[e_idx]:
                    e_a, e_b, e_c, e_d = _cycles[cid2]
                    if e_a not in active_edge_set or e_b not in active_edge_set or e_c not in active_edge_set or e_d not in active_edge_set:
                        continue
                    
                    was_bad = is_bad[cid2]
                    c_a = colors[e_a] if e_a != e_idx else new_c
                    c_b = colors[e_b] if e_b != e_idx else new_c
                    c_c = colors[e_c] if e_c != e_idx else new_c
                    c_d = colors[e_d] if e_d != e_idx else new_c
                    
                    is_bad_now = (len({c_a, c_b, c_c, c_d}) < 4)
                    
                    if was_bad and not is_bad_now:
                        delta -= 1
                    elif not was_bad and is_bad_now:
                        delta += 1
                
                # Accept
                if delta < 0 or (T > min_T and random.random() < pow(2.71828, -delta / T)):
                    colors[e_idx] = new_c
                    for cid2 in _edge_to_cycles[e_idx]:
                        e_a, e_b, e_c, e_d = _cycles[cid2]
                        if e_a not in active_edge_set or e_b not in active_edge_set or e_c not in active_edge_set or e_d not in active_edge_set:
                            continue
                        
                        c_a, c_b, c_c, c_d = colors[e_a], colors[e_b], colors[e_c], colors[e_d]
                        new_state = (len({c_a, c_b, c_c, c_d}) < 4)
                        
                        if new_state != is_bad[cid2]:
                            if new_state:
                                bad_count += 1
                            else:
                                bad_count -= 1
                            is_bad[cid2] = new_state
                    
                    if bad_count == 0:
                        return True
                    
                T *= cooling
                if T < min_T:
                    T = min_T
            
            return False
        
        if try_coloring(active_edges):
            # Success! We have a valid coloring of all 190 edges
            result = [[u, v, colors[i]] for i, (u, v) in enumerate(_all_edges)]
            return result
        
        # If coloring failed, we need to remove edges to make it colorable
        # Use greedy removal: remove edges that are in the most bad cycles
        
        # Re-run to get bad cycles
        for e in active_edges:
            colors[e] = random.randint(0, 19)
        
        is_bad = [False] * C
        bad_count = 0
        for cid in range(C):
            e1, e2, e3, e4 = _cycles[cid]
            if e1 in active_edges and e2 in active_edges and e3 in active_edges and e4 in active_edges:
                c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                if len({c1, c2, c3, c4}) < 4:
                    is_bad[cid] = True
                    bad_count += 1
        
        bad_cycles = [cid for cid in range(C) if is_bad[cid]]
        
        # Remove edges from bad cycles
        while bad_cycles and time.time() < deadline - 2:
            edge_counts = {}
            for cid in bad_cycles:
                for e in _cycles[cid]:
                    if e in active_edges:
                        edge_counts[e] = edge_counts.get(e, 0) + 1
            
            if not edge_counts:
                break
            
            max_count = -1
            best_edge = -1
            for e, count in edge_counts.items():
                if count > max_count:
                    max_count = count
                    best_edge = e
            
            if best_edge == -1:
                break
            
            active_edges.remove(best_edge)
            
            new_bad_cycles = []
            for cid in bad_cycles:
                if best_edge in _cycles[cid]:
                    pass
                else:
                    new_bad_cycles.append(cid)
            bad_cycles = new_bad_cycles
        
        # Now try to color the remaining edges
        remaining_edges = set(active_edges)
        if not remaining_edges:
            continue
        
        # Color the remaining edges
        for e in remaining_edges:
            colors[e] = random.randint(0, 19)
        
        # Verify
        valid = True
        for cid in range(C):
            e1, e2, e3, e4 = _cycles[cid]
            if e1 in remaining_edges and e2 in remaining_edges and e3 in remaining_edges and e4 in remaining_edges:
                c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
                if len({c1, c2, c3, c4}) < 4:
                    valid = False
                    break
        
        if valid:
            edge_count = len(remaining_edges)
            if edge_count > best_edge_count:
                best_edge_count = edge_count
                best_result = [[u, v, colors[e]] for e in remaining_edges for (u, v) in [_all_edges[e]]]
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a small valid graph
    # Use a simple construction: take edges from a bipartite graph
    # Bipartite graphs have no odd cycles, but can have 4-cycles
    # Let's just return a few edges that don't form 4-cycles
    
    # Simple fallback: star graph (no cycles at all)
    result = []
    for v in range(1, N):
        result.append([0, v, v % 20])
    
    return result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
