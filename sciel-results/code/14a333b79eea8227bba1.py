import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    edges = []
    edge_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx += 1
            
    def get_e_idx(u, v):
        if u > v: u, v = v, u
        return edge_to_idx[(u, v)]
        
    cycles = []
    affected_cycles = [[] for _ in range(NUM_EDGES)]
    
    for v_tuple in itertools.combinations(range(N), 4):
        a, b, c, d = v_tuple
        # 3 cycles
        # 1: a-b-c-d
        e1, e2, e3, e4 = get_e_idx(a,b), get_e_idx(b,c), get_e_idx(c,d), get_e_idx(d,a)
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        for e in (e1, e2, e3, e4):
            affected_cycles[e].append(cid)
            
        # 2: a-b-d-c
        e1, e2, e3, e4 = get_e_idx(a,b), get_e_idx(b,d), get_e_idx(d,c), get_e_idx(c,a)
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        for e in (e1, e2, e3, e4):
            affected_cycles[e].append(cid)
            
        # 3: a-c-b-d
        e1, e2, e3, e4 = get_e_idx(a,c), get_e_idx(c,b), get_e_idx(b,d), get_e_idx(d,a)
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        for e in (e1, e2, e3, e4):
            affected_cycles[e].append(cid)

    colors = [0] * NUM_EDGES
    color_counts = [0] * NUM_COLORS
    color_counts[0] = NUM_EDGES
    num_distinct = 1
    
    cycle_colors = []
    for _ in cycles:
        cycle_colors.append((0, 0, 0, 0))
        
    bad_count = 0
    best_colors = colors[:]
    best_distinct = 1
    
    unused_colors = [i for i in range(1, NUM_COLORS)]
    random.shuffle(unused_colors)
    
    while time.time() < deadline:
        e_idx = random.randint(0, NUM_EDGES - 1)
        current_color = colors[e_idx]
        
        # Try to add a new color
        if unused_colors:
            # Pick a random unused color
            new_color = unused_colors[random.randint(0, len(unused_colors)-1)]
            
            # Check validity
            is_valid = True
            for cid in affected_cycles[e_idx]:
                c_edges = cycles[cid]
                # Build new tuple
                # c_edges is (e0, e1, e2, e3)
                # We need to replace the one that is e_idx
                # Instead of searching, we can just build the tuple of colors
                v0 = colors[c_edges[0]]
                v1 = colors[c_edges[1]]
                v2 = colors[c_edges[2]]
                v3 = colors[c_edges[3]]
                
                # Find which one is e_idx
                # This is a bit slow.
                # Better: Precompute the index of e_idx in the cycle tuple?
                # No, e_idx is fixed for the edge, but cycles change.
                # Wait, `affected_cycles[e_idx]` gives the cycles.
                # For a specific cycle `cid`, the position of `e_idx` is fixed.
                # We can precompute `pos_in_cycle[cid]`?
                # No, `cid` is the index in `cycles`.
                # Let's precompute `cycle_pos[e_idx]`? No.
                # Let's precompute for each cycle `cid`, a dict or list of positions?
                # Or just:
                # if c_edges[0] == e_idx: new_v0 = new_color ...
                
                if c_edges[0] == e_idx:
                    new_v0 = new_color
                else:
                    new_v0 = v0
                if c_edges[1] == e_idx:
                    new_v1 = new_color
                else:
                    new_v1 = v1
                if c_edges[2] == e_idx:
                    new_v2 = new_color
                else:
                    new_v2 = v2
                if c_edges[3] == e_idx:
                    new_v3 = new_color
                else:
                    new_v3 = v3
                    
                if len({new_v0, new_v1, new_v2, new_v3}) == 4:
                    is_valid = False
                    break
                    
            if is_valid:
                # Accept
                colors[e_idx] = new_color
                color_counts[current_color] -= 1
                color_counts[new_color] += 1
                num_distinct += 1 # Since new_color was unused
                
                # Update cycle_colors
                for cid in affected_cycles[e_idx]:
                    c_edges = cycles[cid]
                    v0 = colors[c_edges[0]]
                    v1 = colors[c_edges[1]]
                    v2 = colors[c_edges[2]]
                    v3 = colors[c_edges[3]]
                    cycle_colors[cid] = (v0, v1, v2, v3)
                    
                # Update unused
                if color_counts[current_color] == 0:
                    unused_colors.append(current_color)
                unused_colors.remove(new_color)
                
                if num_distinct > best_distinct:
                    best_distinct = num_distinct
                    best_colors = colors[:]
                    
        else:
            # No unused colors, we have maxed out or stuck
            # Try to swap?
            # For now, break or try random moves
            pass

    return best_colors
