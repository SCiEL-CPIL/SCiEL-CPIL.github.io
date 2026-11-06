import time
import random
import math
from itertools import combinations

def is_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses optimized backtracking with unit propagation.
    """
    if not edges:
        return False
    
    num_edges = len(edges)
    
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Precompute for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        for v in range(n):
            if em & (1 << v):
                vertex_edges[v].append(i)
    
    # State: 
    # color[v] = -1 (unassigned), 0, or 1
    # c0[i], c1[i] = count of color 0/1 in edge i
    color = [-1] * n
    c0 = [0] * num_edges
    c1 = [0] * num_edges
    
    # Unit propagation
    # If an edge has 4 of one color and 0 of the other, the 5th must be the other color.
    # If an edge has 5 of one color, it's a conflict (monochromatic edge).
    
    def propagate():
        """
        Returns False on conflict, True if no conflict.
        Modifies color, c0, c1 in place.
        """
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                # If 4 are 0 and 0 are 1, the remaining must be 1
                if c0[i] == 4 and c1[i] == 0:
                    # Find the unassigned vertex in this edge
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 1
                            c1[i] += 1
                            # Update other edges containing v
                            for other_e in vertex_edges[v]:
                                c1[other_e] += 1
                            changed = True
                            if c1[i] == 5:
                                return False
                            break
                # If 4 are 1 and 0 are 0, the remaining must be 0
                elif c1[i] == 4 and c0[i] == 0:
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            color[v] = 0
                            c0[i] += 1
                            for other_e in vertex_edges[v]:
                                c0[other_e] += 1
                            changed = True
                            if c0[i] == 5:
                                return False
                            break
        return True

    def backtrack():
        # Heuristic: Find the unassigned vertex that is in the most constrained edge.
        # Constraint score: max(c0, c1) for edges containing v, or number of unassigned in edge.
        # Let's pick the edge with the fewest unassigned vertices.
        best_edge = -1
        min_unassigned = 6
        best_constraint = 0
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            if unassigned < min_unassigned:
                min_unassigned = unassigned
                best_edge = i
                best_constraint = max(c0[i], c1[i])
                if unassigned == 0:
                    # This edge is fully assigned and not monochromatic (checked above)
                    # So it's satisfied.
                    pass
        
        if best_edge == -1:
            # All edges have 0 unassigned vertices? 
            # If all edges are satisfied, we found a coloring.
            # Wait, if unassigned is 0 for all, and no conflict, then it's a valid 2-coloring.
            # So the hypergraph IS 2-colorable.
            # The function is_non_2_colorable returns False if it IS 2-colorable.
            return True # Found a valid coloring, so it IS 2-colorable. 
            # Wait, the logic of backtrack is usually: "Does a valid coloring exist?"
            # If backtrack returns True, a valid coloring exists.
            # So is_non_2_colorable should return NOT backtrack().
            # Let's refine: backtrack returns True if a valid 2-coloring exists.
            
        # Find an unassigned vertex in best_edge
        candidates = []
        for v in range(n):
            if color[v] == -1 and (edge_masks[best_edge] & (1 << v)):
                candidates.append(v)
                
        if not candidates:
            # Should not happen if min_unassigned > 0
            return True 

        # Try assigning colors to the first candidate (or all? usually one is enough if we branch on value)
        # We pick one vertex and branch on its color (0 or 1).
        v = candidates[0]
        
        # Try Color 0
        saved_c0 = c0[:]
        saved_c1 = c1[:]
        color[v] = 0
        ok = True
        for e_idx in vertex_edges[v]:
            c0[e_idx] += 1
            if c0[e_idx] == 5:
                ok = False
                break
        if ok and propagate():
            if backtrack():
                return True
        color[v] = -1
        c0[:] = saved_c0
        c1[:] = saved_c1
        
        # Try Color 1
        saved_c0 = c0[:]
        saved_c1 = c1[:]
        color[v] = 1
        ok = True
        for e_idx in vertex_edges[v]:
            c1[e_idx] += 1
            if c1[e_idx] == 5:
                ok = False
                break
        if ok and propagate():
            if backtrack():
                return True
        color[v] = -1
        c0[:] = saved_c0
        c1[:] = saved_c1
        
        return False
    
    # If backtrack() returns True, a valid 2-coloring exists.
    # We want to know if it is NON-2-colorable.
    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_edges = None
    best_n = 0
    best_m = float('inf')
    
    # Initial candidate: K_9^5
    n_init = 9
    init_edges = [list(e) for e in combinations(range(n_init), 5)]
    
    if is_non_2_colorable(n_init, init_edges):
        best_edges = init_edges[:]
        best_n = n_init
        best_m = len(init_edges)
    else:
        # Fallback if K9 is somehow colorable (it shouldn't be)
        # Try K_10^5
        n_init = 10
        init_edges = [list(e) for e in combinations(range(n_init), 5)]
        if is_non_2_colorable(n_init, init_edges):
            best_edges = init_edges[:]
            best_n = n_init
            best_m = len(init_edges)
        else:
            return {'n': 0, 'edges': []}
            
    current_n = best_n
    current_edges = best_edges[:]
    current_m = len(current_edges)
    
    # Precompute all possible edges for current_n
    all_possible_edges = list(combinations(range(current_n), 5))
    all_possible_set = set(all_possible_edges)
    
    # Simulated Annealing Parameters
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # To allow changing n, we can occasionally try a different n.
    # But for now, let's fix n=9 and try to minimize edges.
    # If we get stuck, we can try n=10 or n=8.
    
    # Heuristic: If we get stuck at a local minimum for a while, try a different n.
    # But let's start with n=9.
    
    # Optimization: Keep a set of current edges for O(1) add/remove checks
    current_edge_set = set(current_edges)
    
    no_improve_count = 0
    last_improve_time = time.time()
    
    while time.time() < deadline and T > min_T:
        # Generate a random move
        move_type = random.random()
        
        new_edges = current_edges[:]
        
        if move_type < 0.5:
            # Remove a random edge
            if current_m > 1:
                idx = random.randint(0, current_m - 1)
                edge_to_remove = current_edges[idx]
                new_edges.pop(idx)
                new_edge_set = set(new_edges)
                
                # Check validity
                if is_non_2_colorable(current_n, new_edges):
                    # Valid and smaller
                    current_edges = new_edges
                    current_m = len(current_edges)
                    no_improve_count = 0
                    last_improve_time = time.time()
                    
                    if current_m < best_m:
                        best_m = current_m
                        best_edges = current_edges[:]
                        best_n = current_n
                        
        elif move_type < 0.8:
            # Add a random edge
            # Pick an edge not in current set
            # For efficiency, pick from a random subset or just random from all_possible
            # If n is small, all_possible is small.
            edge_to_add = random.choice(all_possible_edges)
            if edge_to_add not in current_edge_set:
                new_edges.append(edge_to_add)
                # new_edge_set.add(edge_to_add)
                
                # Check validity
                if is_non_2_colorable(current_n, new_edges):
                    # Valid but larger
                    delta_E = 1
                    if random.random() < math.exp(-delta_E / T):
                        current_edges = new_edges
                        current_m = len(current_edges)
                        no_improve_count = 0
                        last_improve_time = time.time()
                        
        else:
            # Swap: Remove one, Add one
            if current_m > 0:
                idx = random.randint(0, current_m - 1)
                edge_to_remove = current_edges[idx]
                edge_to_add = random.choice(all_possible_edges)
                
                if edge_to_add != edge_to_remove:
                    new_edges.pop(idx)
                    new_edges.append(edge_to_add)
                    
                    if is_non_2_colorable(current_n, new_edges):
                        # Size is same, but structure changed. Might help escape local minima?
                        # Actually, if size is same, it doesn't improve best_m, but might allow future removals.
                        # Accept with some probability to explore.
                        if random.random() < 0.1: # Low prob to accept neutral moves
                            current_edges = new_edges
                            current_m = len(current_edges)
                            no_improve_count = 0
                            last_improve_time = time.time()
        
        T *= cooling
        
        # If stuck for a long time, try a different n?
        # Let's add a mechanism to restart with a different n if stuck.
        if time.time() - last_improve_time > 5.0:
            # Try a different n
            new_n = random.choice([8, 9, 10, 11])
            if new_n != current_n:
                new_init_edges = [list(e) for e in combinations(range(new_n), 5)]
                if is_non_2_colorable(new_n, new_init_edges):
                    # Reset current state to this new complete graph
                    current_n = new_n
                    current_edges = new_init_edges[:]
                    current_m = len(current_edges)
                    all_possible_edges = list(combinations(range(current_n), 5))
                    all_possible_set = set(all_possible_edges)
                    current_edge_set = set(current_edges)
                    last_improve_time = time.time()
                    no_improve_count = 0
                    T = 5.0 # Reset temperature
                    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
