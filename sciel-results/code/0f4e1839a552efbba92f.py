import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 20
    C = 20
    
    # Precompute all 4-cycles
    all_cycles = []
    edge_to_cycles = {}
    
    def get_edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
        
    def register_cycle(cycle_edges):
        idx = len(all_cycles)
        all_cycles.append(cycle_edges)
        for e in cycle_edges:
            if e not in edge_to_cycles:
                edge_to_cycles[e] = []
            edge_to_cycles[e].append(idx)
            
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = get_edge_key(a,b)
                    e2 = get_edge_key(b,c)
                    e3 = get_edge_key(c,d)
                    e4 = get_edge_key(d,a)
                    register_cycle((e1, e2, e3, e4))
                    
                    # Cycle 2: a-b-d-c-a
                    e1 = get_edge_key(a,b)
                    e2 = get_edge_key(b,d)
                    e3 = get_edge_key(d,c)
                    e4 = get_edge_key(c,a)
                    register_cycle((e1, e2, e3, e4))
                    
                    # Cycle 3: a-c-b-d-a
                    e1 = get_edge_key(a,c)
                    e2 = get_edge_key(c,b)
                    e3 = get_edge_key(b,d)
                    e4 = get_edge_key(d,a)
                    register_cycle((e1, e2, e3, e4))
                    
    num_cycles = len(all_cycles)
    
    # State
    edges = [[-1]*N for _ in range(N)]
    
    # Initialize with complete graph and random colors
    random.seed(42)
    for u in range(N):
        for v in range(u+1, N):
            edges[u][v] = random.randint(0, C-1)
            edges[v][u] = edges[u][v]
            
    cycle_valid = [False] * num_cycles
    total_violations = num_cycles
    
    def is_cycle_rainbow(cycle_idx):
        e1, e2, e3, e4 = all_cycles[cycle_idx]
        c1 = edges[e1[0]][e1[1]]
        c2 = edges[e2[0]][e2[1]]
        c3 = edges[e3[0]][e3[1]]
        c4 = edges[e4[0]][e4[1]]
        
        if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
            return True
        
        return len({c1, c2, c3, c4}) == 4

    for i in range(num_cycles):
        if is_cycle_rainbow(i):
            cycle_valid[i] = True
            total_violations -= 1
        else:
            cycle_valid[i] = False

    best_edges = [row[:] for row in edges]
    best_score = 0
    if total_violations == 0:
        best_score = sum(1 for u in range(N) for v in range(u+1, N) if edges[u][v] >= 0)
        
    def count_edges():
        return sum(1 for u in range(N) for v in range(u+1, N) if edges[u][v] >= 0)

    def update_cycle_status(cycle_idx, new_valid):
        nonlocal total_violations
        if cycle_valid[cycle_idx] and not new_valid:
            total_violations += 1
        elif not cycle_valid[cycle_idx] and new_valid:
            total_violations -= 1
        cycle_valid[cycle_idx] = new_valid

    def try_recolor(u, v, new_color):
        if u > v: u, v = v, u
        if edges[u][v] == -1:
            return False
        if edges[u][v] == new_color:
            return True
            
        old_color = edges[u][v]
        
        edges[u][v] = new_color
        edges[v][u] = new_color
        
        delta_violations = 0
        affected_cycles = edge_to_cycles.get((u,v), [])
        
        for c_idx in affected_cycles:
            was_valid = cycle_valid[c_idx]
            e1, e2, e3, e4 = all_cycles[c_idx]
            c1 = edges[e1[0]][e1[1]]
            c2 = edges[e2[0]][e2[1]]
            c3 = edges[e3[0]][e3[1]]
            c4 = edges[e4[0]][e4[1]]
            
            if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                is_now_valid = True
            else:
                is_now_valid = len({c1, c2, c3, c4}) == 4
                
            if was_valid and not is_now_valid:
                delta_violations += 1
            elif not was_valid and is_now_valid:
                delta_violations -= 1
                
        if delta_violations > 0:
            edges[u][v] = old_color
            edges[v][u] = old_color
            return False
        else:
            for c_idx in affected_cycles:
                e1, e2, e3, e4 = all_cycles[c_idx]
                c1 = edges[e1[0]][e1[1]]
                c2 = edges[e2[0]][e2[1]]
                c3 = edges[e3[0]][e3[1]]
                c4 = edges[e4[0]][e4[1]]
                
                if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                    is_now_valid = True
                else:
                    is_now_valid = len({c1, c2, c3, c4}) == 4
                    
                update_cycle_status(c_idx, is_now_valid)
            return True

    def try_add_edge(u, v):
        if u > v: u, v = v, u
        if edges[u][v] != -1:
            return False
            
        for c in range(C):
            edges[u][v] = c
            edges[v][u] = c
            
            delta_violations = 0
            affected_cycles = edge_to_cycles.get((u,v), [])
            
            for c_idx in affected_cycles:
                was_valid = cycle_valid[c_idx]
                e1, e2, e3, e4 = all_cycles[c_idx]
                c1 = edges[e1[0]][e1[1]]
                c2 = edges[e2[0]][e2[1]]
                c3 = edges[e3[0]][e3[1]]
                c4 = edges[e4[0]][e4[1]]
                
                is_now_valid = len({c1, c2, c3, c4}) == 4
                
                if was_valid and not is_now_valid:
                    delta_violations += 1
                elif not was_valid and is_now_valid:
                    delta_violations -= 1
                    
            if delta_violations == 0:
                for c_idx in affected_cycles:
                    e1, e2, e3, e4 = all_cycles[c_idx]
                    c1 = edges[e1[0]][e1[1]]
                    c2 = edges[e2[0]][e2[1]]
                    c3 = edges[e3[0]][e3[1]]
                    c4 = edges[e4[0]][e4[1]]
                    is_now_valid = len({c1, c2, c3, c4}) == 4
                    update_cycle_status(c_idx, is_now_valid)
                return True
            else:
                edges[u][v] = -1
                edges[v][u] = -1
        return False

    def try_remove_edge(u, v):
        if u > v: u, v = v, u
        if edges[u][v] == -1:
            return False
            
        old_color = edges[u][v]
        edges[u][v] = -1
        edges[v][u] = -1
        
        affected_cycles = edge_to_cycles.get((u,v), [])
        for c_idx in affected_cycles:
            e1, e2, e3, e4 = all_cycles[c_idx]
            c1 = edges[e1[0]][e1[1]]
            c2 = edges[e2[0]][e2[1]]
            c3 = edges[e3[0]][e3[1]]
            c4 = edges[e4[0]][e4[1]]
            
            is_now_valid = True
            update_cycle_status(c_idx, is_now_valid)
        return True

    while time.time() < deadline:
        if total_violations > 0:
            violated_cycles = [i for i in range(num_cycles) if not cycle_valid[i]]
            if not violated_cycles:
                break
            target_cycle = random.choice(violated_cycles)
            e1, e2, e3, e4 = all_cycles[target_cycle]
            
            edges_in_cycle = [e1, e2, e3, e4]
            random.shuffle(edges_in_cycle)
            
            improved = False
            for e in edges_in_cycle:
                u, v = e
                if edges[u][v] == -1:
                    continue
                for _ in range(5):
                    new_c = random.randint(0, C-1)
                    if try_recolor(u, v, new_c):
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                e = random.choice(edges_in_cycle)
                u, v = e
                try_remove_edge(u, v)
                
        else:
            missing_edges = [(u, v) for u in range(N) for v in range(u+1, N) if edges[u][v] == -1]
            if not missing_edges:
                break
                
            random.shuffle(missing_edges)
            added = False
            for u, v in missing_edges:
                if try_add_edge(u, v):
                    added = True
                    if total_violations == 0:
                        current_score = count_edges()
                        if current_score > best_score:
                            best_score = current_score
                            best_edges = [row[:] for row in edges]
                    break
                    
            if not added:
                existing_edges = [(u, v) for u in range(N) for v in range(u+1, N) if edges[u][v] != -1]
                if existing_edges:
                    u, v = random.choice(existing_edges)
                    for _ in range(10):
                        new_c = random.randint(0, C-1)
                        if try_recolor(u, v, new_c):
                            break

    if total_violations == 0:
        result_edges = best_edges
    else:
        result_edges = [[-1]*N for _ in range(N)]
        for v in range(1, N):
            result_edges[0][v] = v % C
            result_edges[v][0] = v % C

    result = []
    for u in range(N):
        for v in range(u+1, N):
            if result_edges[u][v] >= 0:
                result.append([u, v, result_edges[u][v]])
                
    return result
