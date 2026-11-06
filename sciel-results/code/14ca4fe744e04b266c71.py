import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles and edge-to-cycle mappings
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        
        for cycle in [c1, c2, c3]:
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            if len(set(norm_cycle)) == 4:
                cycle_idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(cycle_idx)
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # State
    edge_colors = {}
    missing_edges = set()
    
    def get_conflict_edges():
        conflict_edges = set()
        conflict_count = defaultdict(int)
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict_edges.add(e)
                        conflict_count[e] += 1
        return conflict_edges, conflict_count
    
    def is_color_valid_for_edge(e, c, current_edge_colors):
        if e not in edge_to_cycles:
            return True
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_edges = [other_e for other_e in cycle if other_e != e]
            if not all(other_e in current_edge_colors for other_e in other_edges):
                continue
            other_colors = [current_edge_colors[other_e] for other_e in other_edges]
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True
    
    def try_add_edge(e, prefer_random_color=True):
        if e in edge_colors:
            return True
        if e not in missing_edges:
            return False
        colors = list(range(num_colors))
        random.shuffle(colors)
        for c in colors:
            if is_color_valid_for_edge(e, c, edge_colors):
                edge_colors[e] = c
                missing_edges.remove(e)
                return True
        return False
    
    # Initialize with a dense random graph
    p = 0.95
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < p:
                edge_colors[(u,v)] = random.randint(0, num_colors - 1)
            else:
                missing_edges.add((u,v))
    
    # Phase 1: Repair
    max_repair_steps = 1000
    for step in range(max_repair_steps):
        if time.time() > deadline:
            break
        conflict_edges, conflict_count = get_conflict_edges()
        if not conflict_edges:
            break
        max_count = max(conflict_count.values())
        candidates = [e for e in conflict_edges if conflict_count[e] == max_count]
        e_to_remove = random.choice(candidates)
        del edge_colors[e_to_remove]
        missing_edges.add(e_to_remove)
    
    best_edge_count = len(edge_colors)
    no_improve_count = 0
    max_no_improve = 800
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Try to add a missing edge
        if missing_edges:
            missing_list = list(missing_edges)
            random.shuffle(missing_list)
            for e_add in missing_list[:min(20, len(missing_list))]:
                if try_add_edge(e_add):
                    improved = True
                    if len(edge_colors) > best_edge_count:
                        best_edge_count = len(edge_colors)
                    no_improve_count = 0
                    break
            if improved:
                continue
        
        # Try recoloring
        if edge_colors and missing_edges:
            e_recolor = random.choice(list(edge_colors.keys()))
            old_color = edge_colors[e_recolor]
            new_colors = list(range(num_colors))
            random.shuffle(new_colors)
            found_new = False
            for new_color in new_colors:
                if new_color == old_color:
                    continue
                del edge_colors[e_recolor]
                if is_color_valid_for_edge(e_recolor, new_color, edge_colors):
                    edge_colors[e_recolor] = new_color
                    if missing_edges:
                        for e_try in random.sample(list(missing_edges), min(10, len(missing_edges))):
                            if try_add_edge(e_try):
                                improved = True
                                if len(edge_colors) > best_edge_count:
                                    best_edge_count = len(edge_colors)
                                no_improve_count = 0
                                found_new = True
                                break
                    if found_new:
                        break
                    else:
                        pass
                else:
                    edge_colors[e_recolor] = old_color
                    break
            if improved:
                continue
        
        # Swap
        if random.random() < 0.3 and edge_colors and missing_edges:
            e_remove = random.choice(list(edge_colors.keys()))
            del edge_colors[e_remove]
            e_add = random.choice(list(missing_edges))
            added = try_add_edge(e_add)
            if added:
                readded = False
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove, c, edge_colors):
                        edge_colors[e_remove] = c
                        readded = True
                        break
                if readded:
                    improved = True
                    if len(edge_colors) > best_edge_count:
                        best_edge_count = len(edge_colors)
                    no_improve_count = 0
                    continue
            else:
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove, c, edge_colors):
                        edge_colors[e_remove] = c
                        break
        
        # Double swap
        if random.random() < 0.1 and len(edge_colors) >= 2 and len(missing_edges) >= 2:
            e_remove1 = random.choice(list(edge_colors.keys()))
            e_remove2 = random.choice([e for e in edge_colors if e != e_remove1])
            del edge_colors[e_remove1]
            del edge_colors[e_remove2]
            e_add1 = random.choice(list(missing_edges))
            e_add2 = random.choice([e for e in missing_edges if e != e_add1])
            added1 = try_add_edge(e_add1)
            added2 = try_add_edge(e_add2)
            if added1 and added2:
                readded1 = False
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove1, c, edge_colors):
                        edge_colors[e_remove1] = c
                        readded1 = True
                        break
                readded2 = False
                if readded1:
                    for c in range(num_colors):
                        if is_color_valid_for_edge(e_remove2, c, edge_colors):
                            edge_colors[e_remove2] = c
                            readded2 = True
                            break
                if readded1 and readded2:
                    improved = True
                    if len(edge_colors) > best_edge_count:
                        best_edge_count = len(edge_colors)
                    no_improve_count = 0
                    continue
            else:
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove1, c, edge_colors):
                        edge_colors[e_remove1] = c
                        break
                for c in range(num_colors):
                    if is_color_valid_for_edge(e_remove2, c, edge_colors):
                        edge_colors[e_remove2] = c
                        break
        
        if not improved:
            no_improve_count += 1
    
    # Final cleanup
    max_final_repair = 200
    for _ in range(max_final_repair):
        if time.time() > deadline:
            break
        conflict_edges, _ = get_conflict_edges()
        if not conflict_edges:
            break
        e_to_remove = random.choice(list(conflict_edges))
        del edge_colors[e_to_remove]
        missing_edges.add(e_to_remove)
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    
    return result
