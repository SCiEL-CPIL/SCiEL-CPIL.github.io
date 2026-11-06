import random
import time

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Pre-computation
    edges_list = []
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            edges_list.append((u, v))
            idx += 1
            
    num_edges = len(edges_list)
    
    cycles = []
    edge_to_cycles = [[] for _ in range(num_edges)]
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    verts = [i, j, k, l]
                    # 3 cycles
                    # 0-1-2-3-0
                    e1 = edge_index[(verts[0], verts[1])]
                    e2 = edge_index[(verts[1], verts[2])]
                    e3 = edge_index[(verts[2], verts[3])]
                    e4 = edge_index[(verts[3], verts[0])]
                    c_idx = len(cycles)
                    cycles.append((e1, e2, e3, e4))
                    for e in (e1, e2, e3, e4):
                        edge_to_cycles[e].append(c_idx)
                        
                    # 0-1-3-2-0
                    e1 = edge_index[(verts[0], verts[1])]
                    e2 = edge_index[(verts[1], verts[3])]
                    e3 = edge_index[(verts[3], verts[2])]
                    e4 = edge_index[(verts[2], verts[0])]
                    c_idx = len(cycles)
                    cycles.append((e1, e2, e3, e4))
                    for e in (e1, e2, e3, e4):
                        edge_to_cycles[e].append(c_idx)
                        
                    # 0-2-1-3-0
                    e1 = edge_index[(verts[0], verts[2])]
                    e2 = edge_index[(verts[2], verts[1])]
                    e3 = edge_index[(verts[1], verts[3])]
                    e4 = edge_index[(verts[3], verts[0])]
                    c_idx = len(cycles)
                    cycles.append((e1, e2, e3, e4))
                    for e in (e1, e2, e3, e4):
                        edge_to_cycles[e].append(c_idx)
                        
    num_cycles = len(cycles)
    
    def is_rainbow(cycle, ec):
        c1 = ec[cycle[0]]
        c2 = ec[cycle[1]]
        c3 = ec[cycle[2]]
        c4 = ec[cycle[3]]
        return len({c1, c2, c3, c4}) == 4

    def solve(seed):
        random.seed(seed)
        ep = [True] * num_edges
        ec = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
        
        bad_cycle_set = set()
        for i in range(num_cycles):
            if not is_rainbow(cycles[i], ec):
                bad_cycle_set.add(i)
                
        current_edges = num_edges
        
        def try_fix():
            nonlocal bad_cycle_set, ec, ep, current_edges
            
            if not bad_cycle_set:
                return True
                
            # Pick a random bad cycle
            bad_cycle_idx = random.choice(list(bad_cycle_set))
            cycle = cycles[bad_cycle_idx]
            
            # Try to recolor
            edge_indices = [0, 1, 2, 3]
            random.shuffle(edge_indices)
            
            best_delta = 0
            best_action = None
            
            for e_idx in edge_indices:
                e = cycle[e_idx]
                if not ep[e]:
                    continue
                
                current_color = ec[e]
                
                other_colors = set()
                for other_e_idx in [0, 1, 2, 3]:
                    if other_e_idx != e_idx:
                        other_e = cycle[other_e_idx]
                        if ep[other_e]:
                            other_colors.add(ec[other_e])
                            
                available_colors = [c for c in range(num_colors) if c not in other_colors]
                if not available_colors:
                    continue
                    
                for c in available_colors:
                    if c == current_color:
                        continue
                        
                    # Calculate delta
                    delta = 0
                    ec[e] = c
                    for c_idx in edge_to_cycles[e]:
                        c_cycle = cycles[c_idx]
                        # Check validity
                        valid = True
                        for ce in c_cycle:
                            if not ep[ce]:
                                valid = False
                                break
                        if not valid:
                            continue
                            
                        was_bad = c_idx in bad_cycle_set
                        is_bad_now = not is_rainbow(c_cycle, ec)
                        
                        if was_bad and not is_bad_now:
                            delta -= 1
                        elif not was_bad and is_bad_now:
                            delta += 1
                            
                    ec[e] = current_color
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_action = (e, c)
                        
            if best_action:
                e, c = best_action
                ec[e] = c
                for c_idx in edge_to_cycles[e]:
                    c_cycle = cycles[c_idx]
                    valid = True
                    for ce in c_cycle:
                        if not ep[ce]:
                            valid = False
                            break
                    if not valid:
                        if c_idx in bad_cycle_set:
                            bad_cycle_set.remove(c_idx)
                        continue
                        
                    is_bad = not is_rainbow(c_cycle, ec)
                    if is_bad:
                        bad_cycle_set.add(c_idx)
                    else:
                        if c_idx in bad_cycle_set:
                            bad_cycle_set.remove(c_idx)
                return True
            
            # Try removal
            best_removal_delta = 0
            best_removal_edge = None
            
            for e_idx in edge_indices:
                e = cycle[e_idx]
                if not ep[e]:
                    continue
                
                bad_count = 0
                for c_idx in edge_to_cycles[e]:
                    if c_idx in bad_cycle_set:
                        bad_count += 1
                        
                if bad_count > best_removal_delta:
                    best_removal_delta = bad_count
                    best_removal_edge = e
                    
            if best_removal_edge is not None:
                e = best_removal_edge
                ep[e] = False
                current_edges -= 1
                for c_idx in edge_to_cycles[e]:
                    if c_idx in bad_cycle_set:
                        bad_cycle_set.remove(c_idx)
                return True
            
            return False

        while bad_cycle_set and time.time() < deadline:
            if not try_fix():
                break
                
        result = []
        for i in range(num_edges):
            if ep[i]:
                u, v = edges_list[i]
                result.append([u, v, ec[i]])
                
        return result

    best_result = []
    best_count = 0
    
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        res = solve(seed)
        if len(res) > best_count:
            best_count = len(res)
            best_result = res
            
    return best_result

