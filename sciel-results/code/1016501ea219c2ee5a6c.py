import time
import random

def construct():
    # Set seed for reproducibility
    random.seed(42)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_to_idx = {}
    
    # Generate edges
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    num_edges = len(edges) # 448
    
    # Generate 4-cycles
    cycles = []
    cycle_to_edges = []
    edge_to_cycles = [[] for _ in range(num_edges)]
    
    for x in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                idxs = [edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]
                cycle_id = len(cycles)
                cycles.append(idxs)
                cycle_to_edges.append(idxs)
                for idx in idxs:
                    edge_to_cycles[idx].append(cycle_id)
                    
    num_cycles = len(cycles) # 672
    
    best_count = 0
    best_set = set()
    
    def run_search(order, time_limit):
        selected = [False] * num_edges
        cycle_counts = [0] * num_cycles
        
        # Phase 1: Greedy
        for idx in order:
            valid = True
            for c_idx in edge_to_cycles[idx]:
                if cycle_counts[c_idx] == 3:
                    valid = False
                    break
            if valid:
                selected[idx] = True
                for c_idx in edge_to_cycles[idx]:
                    cycle_counts[c_idx] += 1
        
        current_count = sum(selected)
        
        # Phase 2: Local Search
        if time_limit > 0:
            search_start = time.time()
            while time.time() - search_start < time_limit:
                added_any = False
                # Try to add edges
                # We scan all edges. If we add one, we continue scanning.
                for idx in range(num_edges):
                    if not selected[idx]:
                        valid = True
                        for c_idx in edge_to_cycles[idx]:
                            if cycle_counts[c_idx] == 3:
                                valid = False
                                break
                        if valid:
                            selected[idx] = True
                            for c_idx in edge_to_cycles[idx]:
                                cycle_counts[c_idx] += 1
                            current_count += 1
                            added_any = True
                
                if not added_any:
                    # Stuck. Perform a shake.
                    non_selected = [i for i in range(num_edges) if not selected[i]]
                    if not non_selected:
                        break
                    target_idx = random.choice(non_selected)
                    
                    blocking_cycles = []
                    for c_idx in edge_to_cycles[target_idx]:
                        if cycle_counts[c_idx] == 3:
                            blocking_cycles.append(c_idx)
                    
                    edges_to_remove = set()
                    for c_idx in blocking_cycles:
                        c_edges = cycle_to_edges[c_idx]
                        sel_edges_in_cycle = [e for e in c_edges if selected[e]]
                        if sel_edges_in_cycle:
                            rem = random.choice(sel_edges_in_cycle)
                            edges_to_remove.add(rem)
                    
                    for idx in edges_to_remove:
                        selected[idx] = False
                        for c_idx in edge_to_cycles[idx]:
                            cycle_counts[c_idx] -= 1
                        current_count -= 1
                    
                    # Try to add target immediately
                    valid = True
                    for c_idx in edge_to_cycles[target_idx]:
                        if cycle_counts[c_idx] == 3:
                            valid = False
                            break
                    if valid:
                        selected[target_idx] = True
                        for c_idx in edge_to_cycles[target_idx]:
                            cycle_counts[c_idx] += 1
                        current_count += 1
        
        return current_count, set(i for i in range(num_edges) if selected[i])

    # Strategy:
    # 1. Run a few quick greedy passes to find a good starting point.
    # 2. Run local searches with time budgets.
    
    # Quick Greedy
    for _ in range(20):
        if time.time() > deadline - 10.0: break
        order = list(range(num_edges))
        random.shuffle(order)
        count, s = run_search(order, 0.0) # 0.0 time for local search, just greedy
        if count > best_count:
            best_count = count
            best_set = s

    # Main Search
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        # Not enough time for local search, just return best greedy
        pass
    else:
        # Divide time among a few searches
        # Let's do 3 searches, each getting 1/3 of the time
        num_searches = 3
        time_per_search = remaining_time / num_searches
        
        for i in range(num_searches):
            if time.time() > deadline - 2.0: break
            order = list(range(num_edges))
            random.shuffle(order)
            count, s = run_search(order, time_per_search)
            if count > best_count:
                best_count = count
                best_set = s

    result = []
    for idx in best_set:
        result.append(list(edges[idx]))
        
    return result
