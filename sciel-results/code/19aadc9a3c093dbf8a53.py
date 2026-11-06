import time
import random
import sys

N = 20
C = 20
DEADLINE = 115.0

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _generate_all_4cycles():
    """
    Generate all 4-cycles as frozensets of 4 edge keys.
    Returns list of cycles and a mapping from edge to list of cycle indices.
    """
    from itertools import combinations
    verts = list(range(N))
    cycles = []
    edge_to_cycles = {}
    
    # Initialize edge_to_cycles for all possible edges
    all_possible_edges = []
    for u in range(N):
        for v in range(u+1, N):
            e = (u, v)
            all_possible_edges.append(e)
            edge_to_cycles[e] = []
            
    for a, b, c, d in combinations(verts, 4):
        # 3 distinct 4-cycles on 4 vertices
        # 1: a-b-c-d-a
        e1 = _edge_key(a, b)
        e2 = _edge_key(b, c)
        e3 = _edge_key(c, d)
        e4 = _edge_key(d, a)
        c1 = frozenset([e1, e2, e3, e4])
        if len(c1) == 4:
            idx = len(cycles)
            cycles.append(c1)
            edge_to_cycles[e1].append(idx)
            edge_to_cycles[e2].append(idx)
            edge_to_cycles[e3].append(idx)
            edge_to_cycles[e4].append(idx)
            
        # 2: a-b-d-c-a
        e1 = _edge_key(a, b)
        e2 = _edge_key(b, d)
        e3 = _edge_key(d, c)
        e4 = _edge_key(c, a)
        c2 = frozenset([e1, e2, e3, e4])
        if len(c2) == 4:
            idx = len(cycles)
            cycles.append(c2)
            edge_to_cycles[e1].append(idx)
            edge_to_cycles[e2].append(idx)
            edge_to_cycles[e3].append(idx)
            edge_to_cycles[e4].append(idx)
            
        # 3: a-c-b-d-a
        e1 = _edge_key(a, c)
        e2 = _edge_key(c, b)
        e3 = _edge_key(b, d)
        e4 = _edge_key(d, a)
        c3 = frozenset([e1, e2, e3, e4])
        if len(c3) == 4:
            idx = len(cycles)
            cycles.append(c3)
            edge_to_cycles[e1].append(idx)
            edge_to_cycles[e2].append(idx)
            edge_to_cycles[e3].append(idx)
            edge_to_cycles[e4].append(idx)
            
    return cycles, edge_to_cycles, all_possible_edges

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    cycles, edge_to_cycles, all_possible_edges = _generate_all_4cycles()
    
    best_adj = {}
    best_count = 0
    
    # Helper to check if a specific cycle is valid given current adj
    def is_cycle_valid(cyc_idx):
        cyc = cycles[cyc_idx]
        cols = []
        for e in cyc:
            if e not in adj:
                return True
            cols.append(adj[e])
        if len(cols) < 4:
            return True
        return len(set(cols)) == 4
        
    # Helper to check if graph is fully valid
    def is_graph_valid():
        for i in range(len(cycles)):
            if not is_cycle_valid(i):
                return False
        return True

    # Restart logic
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() - t0 > DEADLINE:
            break
            
        # Initialize with random dense graph
        # Increase density slightly with restarts?
        num_init = 110 + restart * 5
        if num_init > 150: num_init = 150
        
        adj = {}
        init_edges = rng.sample(all_possible_edges, num_init)
        for e in init_edges:
            adj[e] = rng.randint(0, C-1)
            
        # Main Search Loop for this restart
        max_steps = 20000
        step = 0
        last_full_check = 0
        steps_since_improvement = 0
        
        while time.time() - t0 < DEADLINE and step < max_steps:
            step += 1
            
            # Every 20 steps, do a full check and update best
            if step - last_full_check >= 20:
                if is_graph_valid():
                    if len(adj) > best_count:
                        best_adj = dict(adj)
                        best_count = len(adj)
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 20
                last_full_check = step
                
            # Early stop if stuck too long
            if steps_since_improvement > 500:
                break
                
            # Pick a random cycle to inspect
            cyc_idx = rng.randint(0, len(cycles)-1)
            
            if is_cycle_valid(cyc_idx):
                # This cycle is OK. Try to add an edge.
                u = rng.randint(0, N-1)
                v = rng.randint(0, N-1)
                if u == v: continue
                e_new = _edge_key(u, v)
                
                if e_new in adj:
                    continue
                    
                # Try to find a color for e_new
                found_color = False
                colors = list(range(C))
                rng.shuffle(colors)
                
                for c in colors:
                    adj[e_new] = c
                    ok = True
                    for idx in edge_to_cycles[e_new]:
                        if not is_cycle_valid(idx):
                            ok = False
                            break
                    if ok:
                        found_color = True
                        break
                    else:
                        del adj[e_new]
                        
                if not found_color:
                    if e_new in adj:
                        del adj[e_new]
            else:
                # Cycle is violated. Try to fix.
                cyc = cycles[cyc_idx]
                e_fix = rng.choice(list(cyc))
                
                other_cols = set()
                for e in cyc:
                    if e != e_fix:
                        other_cols.add(adj[e])
                        
                success = False
                colors = list(range(C))
                rng.shuffle(colors)
                
                for c in colors:
                    if c in other_cols:
                        continue
                        
                    old_c = adj[e_fix]
                    adj[e_fix] = c
                    
                    ok = True
                    for idx in edge_to_cycles[e_fix]:
                        if not is_cycle_valid(idx):
                            ok = False
                            break
                            
                    if ok:
                        success = True
                        break
                    else:
                        adj[e_fix] = old_c
                        
                if not success:
                    del adj[e_fix]

        # Final check for this restart
        if is_graph_valid():
            if len(adj) > best_count:
                best_adj = dict(adj)
                best_count = len(adj)

    return [[u, v, best_adj[(u, v)]] for (u, v) in best_adj]

