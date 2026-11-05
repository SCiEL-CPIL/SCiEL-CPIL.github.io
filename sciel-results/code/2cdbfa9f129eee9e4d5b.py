import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable.
    Returns False if no 2-coloring exists (non-2-colorable).
    Optimized for speed with unit propagation.
    """
    if not edges:
        return True
        
    m = len(edges)
    # Map vertex to list of edge indices
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    # State
    colors = [-1] * n
    # edge_counts[e] = [count_0, count_1, count_unknown]
    edge_counts = [[0, 0, 5] for _ in range(m)]
    
    def propagate(queue):
        """
        Process queue of (vertex, color) assignments.
        Returns False if contradiction, True otherwise.
        Modifies colors and edge_counts.
        """
        while queue:
            v, c = queue.pop()
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue
            
            # Assign v to c
            colors[v] = c
            
            for e_idx in vertex_to_edges[v]:
                # Update counts
                edge_counts[e_idx][2] -= 1 # unknown
                edge_counts[e_idx][c] += 1
                
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                
                # Check contradiction: monochromatic
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit propagation
                if cU == 1:
                    # One unknown left
                    e = edges[e_idx]
                    unknown_v = -1
                    for u in e:
                        if colors[u] == -1:
                            unknown_v = u
                            break
                    if unknown_v == -1:
                        continue
                    
                    if c0 == 4:
                        # Must be 1
                        if colors[unknown_v] == 0:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 1))
                    elif c1 == 4:
                        # Must be 0
                        if colors[unknown_v] == 1:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 0))
        return True

    def backtrack():
        # Heuristic: Pick unassigned vertex with highest "pressure"
        best_v = -1
        best_score = -1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            score = 0
            for e_idx in vertex_to_edges[v]:
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                if cU == 1:
                    if c0 == 4 or c1 == 4:
                        score += 10
                    else:
                        score += 1
                elif cU == 2:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_v = v
                
        if best_v == -1:
            return True # All assigned
            
        v = best_v
        
        # Try 0 then 1
        for c in [0, 1]:
            old_colors = colors.copy()
            old_edge_counts = [list(ec) for ec in edge_counts]
            
            if propagate([(v, c)]):
                if backtrack():
                    return True
            
            colors[:] = old_colors
            edge_counts[:] = old_edge_counts
            
        return False

    if propagate([]):
        return backtrack()
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_result = None
    best_edge_count = 200
    
    # We target n=14, 15, 16 as the 51-edge solution is likely in this range.
    # Start with a random dense set, ensure non-2-colorable, then minimize via SA with swaps.
    
    for n in [14, 15, 16]:
        if time.time() > deadline - 10.0:
            break
            
        all_edges = list(combinations(range(n), 5))
        num_all_edges = len(all_edges)
        
        # Try a few initial configurations
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
                
            # Start with a random subset of size ~80
            random.shuffle(all_edges)
            current_edges = set(all_edges[:80])
            
            # Ensure non-2-colorable
            # If it is 2-colorable, add random edges until it isn't
            while is_2_colorable(n, list(current_edges)):
                if time.time() > deadline - 10.0:
                    break
                # Add a random edge not in current set
                candidate = random.choice(all_edges)
                if candidate not in current_edges:
                    current_edges.add(candidate)
                else:
                    # If all edges are in, it must be non-2-colorable (for n>=10 dense enough)
                    # But for n=14, complete is 2002 edges. We only have 80+.
                    # If we can't find one, break
                    break
            
            if is_2_colorable(n, list(current_edges)):
                continue # Failed to make non-2-colorable in time or space
                
            # Local Search / Simulated Annealing
            # Objective: Minimize len(current_edges)
            # Constraint: Not 2-colorable
            
            current_edges_list = list(current_edges)
            temperature = 10.0
            cooling = 0.95
            min_temp = 0.1
            no_improve_count = 0
            max_no_improve = 200
            
            while temperature > min_temp and time.time() < deadline - 5.0:
                # Try a move
                # 50% chance: Remove an edge
                # 50% chance: Swap an edge (remove one, add one)
                
                if random.random() < 0.5 and len(current_edges) > 30:
                    # Try removal
                    idx = random.randint(0, len(current_edges_list) - 1)
                    edge_to_remove = current_edges_list[idx]
                    new_edges_set = current_edges.copy()
                    new_edges_set.discard(edge_to_remove)
                    new_edges_list = list(new_edges_set)
                    
                    if not is_2_colorable(n, new_edges_list):
                        # Accept
                        current_edges = new_edges_set
                        current_edges_list = new_edges_list
                        no_improve_count = 0
                        if len(current_edges) < best_edge_count:
                            best_edge_count = len(current_edges)
                            best_result = {'n': n, 'edges': current_edges_list}
                    else:
                        no_improve_count += 1
                else:
                    # Try swap
                    if len(current_edges) > 30:
                        idx = random.randint(0, len(current_edges_list) - 1)
                        edge_to_remove = current_edges_list[idx]
                        
                        # Pick a random edge to add
                        edge_to_add = random.choice(all_edges)
                        
                        if edge_to_add == edge_to_remove:
                            continue
                            
                        new_edges_set = current_edges.copy()
                        new_edges_set.discard(edge_to_remove)
                        new_edges_set.add(edge_to_add)
                        new_edges_list = list(new_edges_set)
                        
                        if not is_2_colorable(n, new_edges_list):
                            # Accept swap (size unchanged, but structure changed)
                            current_edges = new_edges_set
                            current_edges_list = new_edges_list
                            # Don't reset no_improve_count fully, but give it a break
                            if no_improve_count > 100:
                                no_improve_count = 0
                        else:
                            no_improve_count += 1
                
                temperature *= cooling
                
                if no_improve_count > max_no_improve:
                    # Restart with new random initial state? Or just break
                    # Let's try a few restarts within the same n
                    break
                
        if best_edge_count <= 55:
            break # Good enough, stop early
    
    # Fallback
    if best_result is None:
        n_fb = 9
        all_edges_fb = list(combinations(range(n_fb), 5))
        return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
        
    return best_result
