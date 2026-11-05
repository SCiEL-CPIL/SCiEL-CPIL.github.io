import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    d = 7
    n = 2 ** d
    total_edges = d * 2 ** (d - 1)
    
    # Precompute all edges
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    # Precompute edge index
    edge_set_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_set_idx[(u, v)] = i
    
    # Precompute for each edge, the set of other edges that would form a 4-cycle
    # If we add edge (u,v), it completes a 4-cycle if there exists a path of length 3 from u to v
    # through already-selected edges.
    # For edge (u,v), the 4-cycles containing it are determined by choosing one other bit j != i
    # where i is the bit that differs between u and v.
    # Let's precompute for each edge, the list of "conflict triples" - sets of 3 edges that if all present,
    # would make this edge create a 4-cycle.
    
    # Actually, let's precompute the neighbors structure differently.
    # For each edge e=(u,v), the 4-cycles containing e correspond to vertices w such that
    # dist(u,w)=1, dist(v,w)=1, w!=u, w!=v? No, that's not right.
    
    # A 4-cycle in Q7: vertices a, b, c, d with edges (a,b), (b,c), (c,d), (d,a).
    # If we have edge (u,v) and want to check if it completes a 4-cycle:
    # There must be a path u-w1-w2-v of length 3 using already-selected edges.
    # w1 is a neighbor of u (w1 != v), w2 is a neighbor of w1 (w2 != v), and v is neighbor of w2.
    
    # For efficiency, let's precompute for each edge (u,v) with bit i:
    # The other bits j != i. For each j, the two "opposite" edges in the square are:
    # (u^(1<<j), v^(1<<j)) and the two "side" edges:
    # (u, u^(1<<j)) and (v, v^(1<<j))
    # So the 4-cycle is: (u,v), (v, v^(1<<j)), (v^(1<<j), u^(1<<j)), (u^(1<<j), u)
    # Or equivalently: (u,v), (u, u^(1<<j)), (u^(1<<j), v^(1<<j)), (v^(1<<j), v)
    
    # Precompute for each edge, the 6 (d-1) 4-cycles it belongs to, and the indices of the other 3 edges in each cycle.
    edge_conflicts = [[] for _ in range(len(edges))]
    for idx, (u, v) in enumerate(edges):
        # Find which bit differs
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            # The other 3 edges in this 4-cycle
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            # Edges: (u, u2), (u2, v2), (v2, v)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(u2, v2), max(u2, v2))
            e3 = (min(v2, v), max(v2, v))
            i1 = edge_set_idx.get(e1)
            i2 = edge_set_idx.get(e2)
            i3 = edge_set_idx.get(e3)
            if i1 is not None and i2 is not None and i3 is not None:
                edge_conflicts[idx].append((i1, i2, i3))
    
    # Greedy initialization
    selected = [False] * len(edges)
    selected_count = 0
    
    for idx in range(len(edges)):
        ok = True
        for (i1, i2, i3) in edge_conflicts[idx]:
            if selected[i1] and selected[i2] and selected[i3]:
                ok = False
                break
        if ok:
            selected[idx] = True
            selected_count += 1
    
    best_count = selected_count
    best_selected = selected[:]
    
    def can_add(idx):
        for (i1, i2, i3) in edge_conflicts[idx]:
            if selected[i1] and selected[i2] and selected[i3]:
                return False
        return True
    
    def count_removable():
        """Count how many selected edges can be removed without breaking constraints."""
        cnt = 0
        for idx in range(len(edges)):
            if not selected[idx]:
                continue
            # Check if this edge is "critical" - removing it would allow some other edge to be added
            # Actually, for swap moves, we just need to check if removing it and adding another is valid
            pass
        return cnt
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.001
    
    random.seed(42)
    
    while time.time() < deadline:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random selected edge
            selected_indices = [i for i in range(len(edges)) if selected[i]]
            if not selected_indices:
                continue
            idx = random.choice(selected_indices)
            selected[idx] = False
            selected_count -= 1
            delta = -1
            # Accept with some probability based on temperature
            if delta < 0 and random.random() > T:
                selected[idx] = True
                selected_count += 1
            else:
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
        elif move_type < 0.6:
            # Add a random unselected edge if possible
            unselected_indices = [i for i in range(len(edges)) if not selected[i]]
            if not unselected_indices:
                continue
            idx = random.choice(unselected_indices)
            if can_add(idx):
                selected[idx] = True
                selected_count += 1
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
        else:
            # Swap: remove one selected, try to add one unselected
            selected_indices = [i for i in range(len(edges)) if selected[i]]
            unselected_indices = [i for i in range(len(edges)) if not selected[i]]
            if not selected_indices or not unselected_indices:
                continue
            rem_idx = random.choice(selected_indices)
            add_idx = random.choice(unselected_indices)
            
            # Temporarily remove
            selected[rem_idx] = False
            selected_count -= 1
            
            if can_add(add_idx):
                selected[add_idx] = True
                selected_count += 1
                if selected_count > best_count:
                    best_count = selected_count
                    best_selected = selected[:]
            else:
                # Revert
                selected[rem_idx] = True
                selected_count += 1
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    # Use the best solution found
    final_selected = best_selected
    result = []
    for idx in range(len(edges)):
        if final_selected[idx]:
            result.append([edges[idx][0], edges[idx][1]])
    
    return result
