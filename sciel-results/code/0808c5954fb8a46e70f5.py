import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles and edge-to-cycle mappings
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Three distinct 4-cycles on 4 vertices
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        
        for cycle in [c1, c2, c3]:
            # Normalize edges
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            # Check for duplicates in the cycle itself
            if len(set(norm_cycle)) == 4:
                cycle_idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(cycle_idx)
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def is_color_valid_for_edge(e, c, current_edge_colors):
        """Check if assigning color c to edge e results in all cycles containing e being rainbow."""
        if e not in edge_to_cycles:
            return True
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            # Check if all other edges in this cycle exist
            other_edges = [other_e for other_e in cycle if other_e != e]
            if not all(other_e in current_edge_colors for other_e in other_edges):
                continue
            # Check colors
            other_colors = [current_edge_colors[other_e] for other_e in other_edges]
            # The new color c must be distinct from all other_colors
            if c in other_colors:
                return False
            # If other colors have duplicates, the cycle is already broken
            if len(set(other_colors)) < 3:
                return False
        return True
    
    def get_conflict_edges(edge_colors):
        """Find edges involved in non-rainbow cycles."""
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
    
    def try_add_edge(e, edge_colors, missing_edges):
        """Try to add edge e with a valid color. Returns True if successful."""
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
    
    def full_repair(edge_colors, missing_edges, max_steps=500):
        """Repair conflicts by removing edges, prioritizing high-conflict edges."""
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            # Find conflicts
            conflict_edges, conflict_count = get_conflict_edges(edge_colors)
            if not conflict_edges:
                break
            max_count = max(conflict_count.values())
            candidates = [e for e in conflict_edges if conflict_count[e] == max_count]
            e_to_remove = random.choice(candidates)
            del edge_colors[e_to_remove]
            missing_edges.add(e_to_remove)
    
    def local_search(edge_colors, missing_edges, max_no_improve=800):
        """Local search to add more edges."""
        best_edge_count = len(edge_colors)
        no_improve_count = 0
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # 1. Try to add a missing edge (prioritize edges with fewer cycle constraints)
            if missing_edges:
                missing_list = list(missing_edges)
                # Sort by number of cycles (fewer = easier to add)
                missing_list.sort(key=lambda e: len(edge_to_cycles.get(e, [])))
                random.shuffle(missing_list[:min(20, len(missing_list))])
                for e_add in missing_list[:min(40, len(missing_list))]:
                    if try_add_edge(e_add, edge_colors, missing_edges):
                        improved = True
                        if len(edge_colors) > best_edge_count:
                            best_edge_count = len(edge_colors)
                        no_improve_count = 0
                        break
                if improved:
                    continue
            
            # 2. Try recoloring an existing edge to enable adding a missing edge
            if edge_colors and missing_edges:
                # Prioritize recoloring edges that are in many cycles
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
                        # Check if this recoloring enables adding a missing edge
                        if missing_edges:
                            sample_size = min(20, len(missing_edges))
                            for e_try in random.sample(list(missing_edges), sample_size):
                                if try_add_edge(e_try, edge_colors, missing_edges):
                                    improved = True
                                    if len(edge_colors) > best_edge_count:
                                        best_edge_count = len(edge_colors)
                                    no_improve_count = 0
                                    found_new = True
                                    break
                        if found_new:
                            break
                    else:
                        edge_colors[e_recolor] = old_color
                        break
                
                if improved:
                    continue
            
            # 3. Try a swap: remove one edge, add a missing one, try to re-add removed
            if random.random() < 0.5 and edge_colors and missing_edges:
                e_remove = random.choice(list(edge_colors.keys()))
                del edge_colors[e_remove]
                
                e_add = random.choice(list(missing_edges))
                added = try_add_edge(e_add, edge_colors, missing_edges)
                
                if added:
                    # Try to re-add e_remove
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
                    # e_add failed, re-add e_remove
                    for c in range(num_colors):
                        if is_color_valid_for_edge(e_remove, c, edge_colors):
                            edge_colors[e_remove] = c
                            break
            
            # 4. Try double swap
            if random.random() < 0.2 and len(edge_colors) >= 2 and len(missing_edges) >= 2:
                e_remove1 = random.choice(list(edge_colors.keys()))
                e_remove2 = random.choice([e for e in edge_colors if e != e_remove1])
                del edge_colors[e_remove1]
                del edge_colors[e_remove2]
                
                e_add1 = random.choice(list(missing_edges))
                e_add2 = random.choice([e for e in missing_edges if e != e_add1])
                
                added1 = try_add_edge(e_add1, edge_colors, missing_edges)
                added2 = try_add_edge(e_add2, edge_colors, missing_edges)
                
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
                    # Re-add removed edges
                    for c in range(num_colors):
                        if is_color_valid_for_edge(e_remove1, c, edge_colors):
                            edge_colors[e_remove1] = c
                            break
                    for c in range(num_colors):
                        if is_color_valid_for_edge(e_remove2, c, edge_colors):
                            edge_colors[e_remove2] = c
                            break
            
            # 5. Try triple recolor: change colors of 3 edges to unlock a missing edge
            if random.random() < 0.1 and len(edge_colors) >= 3 and missing_edges:
                edges_to_recolor = random.sample(list(edge_colors.keys()), 3)
                old_colors = {e: edge_colors[e] for e in edges_to_recolor}
                
                # Try random permutations of colors
                new_colors_pool = list(range(num_colors))
                random.shuffle(new_colors_pool)
                
                success = False
                for perm in itertools.permutations(new_colors_pool[:3]):
                    # Remove the 3 edges
                    for e in edges_to_recolor:
                        del edge_colors[e]
                    
                    # Assign new colors
                    valid = True
                    for i, e in enumerate(edges_to_recolor):
                        if not is_color_valid_for_edge(e, perm[i], edge_colors):
                            valid = False
                            break
                        edge_colors[e] = perm[i]
                    
                    if valid:
                        # Try to add a missing edge
                        if missing_edges:
                            for e_try in random.sample(list(missing_edges), min(10, len(missing_edges))):
                                if try_add_edge(e_try, edge_colors, missing_edges):
                                    success = True
                                    break
                    
                    # Restore if not successful
                    if not success:
                        for e in edges_to_recolor:
                            if e in edge_colors:
                                del edge_colors[e]
                        for e, c in old_colors.items():
                            edge_colors[e] = c
                    
                    if success:
                        improved = True
                        if len(edge_colors) > best_edge_count:
                            best_edge_count = len(edge_colors)
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
        
        return edge_colors, missing_edges
    
    # Multiple restarts
    best_result = None
    best_count = 0
    
    # Use remaining time for multiple restarts
    num_restarts = 6
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        edge_colors = {}
        missing_edges = set()
        
        # Initialize with a dense random graph
        p = 0.95
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    edge_colors[(u,v)] = random.randint(0, num_colors - 1)
                else:
                    missing_edges.add((u,v))
        
        # Phase 1: Repair
        full_repair(edge_colors, missing_edges, max_steps=300)
        
        # Phase 2: Local search with remaining time for this restart
        remaining = deadline - time.time()
        if remaining > 5:
            local_search(edge_colors, missing_edges, max_no_improve=600)
        
        # Final cleanup
        full_repair(edge_colors, missing_edges, max_steps=100)
        
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_result = [(u, v, c) for (u, v), c in edge_colors.items()]
    
    # Final polish pass on the best result
    if best_result is not None and time.time() < deadline - 5:
        # Convert back to internal representation
        edge_colors = {}
        for u, v, c in best_result:
            edge_colors[(u, v)] = c
        
        missing_edges = set(all_edges) - set(edge_colors.keys())
        
        # Do one more round of local search
        local_search(edge_colors, missing_edges, max_no_improve=200)
        full_repair(edge_colors, missing_edges, max_steps=50)
        
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_result = [(u, v, c) for (u, v), c in edge_colors.items()]
    
    # If no valid result found, return empty
    if best_result is None:
        return []
    
    return best_result
