import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    d = 7
    n_vertices = 1 << d
    num_edges = d * (n_vertices >> 1) # 448

    # 1. Generate Edges
    edge_u = [0] * num_edges
    edge_v = [0] * num_edges
    edge_idx = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_u[idx] = u
                edge_v[idx] = v
                edge_idx[(u, v)] = idx
                idx += 1

    # 2. Generate Squares (2-faces)
    squares = []
    edges_in_square = [] # List of lists of 4 edge indices

    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_idx.get((min(u, v1), max(u, v1)))
                e2 = edge_idx.get((min(u, v2), max(u, v2)))
                e3 = edge_idx.get((min(v1, v3), max(v1, v3)))
                e4 = edge_idx.get((min(v2, v3), max(v2, v3)))
                
                if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                    sq_edges = [e1, e2, e3, e4]
                    squares.append(sq_edges)
                    edges_in_square.append(sq_edges)
                    
    # Map edge -> list of square indices it belongs to
    squares_by_edge = [[] for _ in range(num_edges)]
    for s_idx, sq in enumerate(squares):
        for e in sq:
            squares_by_edge[e].append(s_idx)

    def can_add(e, selected):
        for s_idx in squares_by_edge[e]:
            sq = edges_in_square[s_idx]
            count = 0
            for oe in sq:
                if oe != e and selected[oe]:
                    count += 1
            if count == 3:
                return False
        return True

    def greedy_init(rng):
        selected = [False] * num_edges
        count = 0
        edges_order = list(range(num_edges))
        rng.shuffle(edges_order)
        
        for e in edges_order:
            if can_add(e, selected):
                selected[e] = True
                count += 1
        return selected, count

    def local_search(selected, count, rng, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Add any valid edge
            for e in range(num_edges):
                if not selected[e]:
                    if can_add(e, selected):
                        selected[e] = True
                        count += 1
                        improved = True
            
            if not improved:
                # Phase 2: Try removing one edge and adding many (1-for-k)
                selected_edges = [i for i in range(num_edges) if selected[i]]
                if not selected_edges:
                    break
                
                found_gain = False
                for _ in range(20):
                    if time.time() >= time_limit:
                        break
                    
                    e_remove = rng.choice(selected_edges)
                    
                    # Save state
                    old_selected = selected.copy()
                    old_count = count
                    
                    # Remove e_remove
                    selected[e_remove] = False
                    count -= 1
                    
                    # Greedily add
                    for e in range(num_edges):
                        if not selected[e]:
                            if can_add(e, selected):
                                selected[e] = True
                                count += 1
                                
                    if count > old_count:
                        found_gain = True
                        improved = True
                        break
                    else:
                        # Revert
                        selected = old_selected
                        count = old_count
                        
                if not found_gain:
                    # Phase 3: Try 1-1 swaps (neutral moves)
                    swap_success = False
                    for _ in range(50):
                        if time.time() >= time_limit:
                            break
                        e_out = rng.choice(selected_edges)
                        unselected = [i for i in range(num_edges) if not selected[i]]
                        if not unselected:
                            break
                        e_in = rng.choice(unselected)
                        
                        if e_out == e_in:
                            continue
                        
                        selected[e_out] = False
                        if can_add(e_in, selected):
                            selected[e_in] = True
                            improved = True
                            swap_success = True
                            break
                        else:
                            selected[e_out] = True
                            
                    if not swap_success:
                        break
                        
        return selected, count

    rng = random.Random(42)
    best_selected, best_count = greedy_init(rng)
    best_selected, best_count = local_search(best_selected, best_count, rng, deadline - 10)
    
    while time.time() < deadline - 5:
        current_selected, current_count = best_selected.copy(), best_count
        selected_edges = [i for i in range(num_edges) if current_selected[i]]
        
        num_remove = rng.randint(5, 10)
        for _ in range(num_remove):
            if selected_edges:
                e = rng.choice(selected_edges)
                current_selected[e] = False
                current_count -= 1
                selected_edges.remove(e)
        
        current_selected, current_count = local_search(current_selected, current_count, rng, min(deadline - 2, time.time() + 5))
        
        if current_count > best_count:
            best_count = current_count
            best_selected = current_selected

    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append([edge_u[i], edge_v[i]])
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
