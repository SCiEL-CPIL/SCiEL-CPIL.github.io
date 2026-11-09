import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    M = [0] * n
    
    def can_add(u, v):
        if u == v:
            return False
        if (M[u] >> v) & 1:
            return False
        
        # Check triples {u, x, y} where x,y in N(v): need |N(u)∩N(x)∩N(y)| < 2
        nv = M[v]
        neigh_v = [i for i in range(n) if (nv >> i) & 1]
        
        for i in range(len(neigh_v)):
            x = neigh_v[i]
            for j in range(i + 1, len(neigh_v)):
                y = neigh_v[j]
                common = M[u] & M[x] & M[y]
                if common.bit_count() >= 2:
                    return False

        # Check triples {v, x, y} where x,y in N(u): need |N(v)∩N(x)∩N(y)| < 2
        nu = M[u]
        neigh_u = [i for i in range(n) if (nu >> i) & 1]
        
        for i in range(len(neigh_u)):
            x = neigh_u[i]
            for j in range(i + 1, len(neigh_u)):
                y = neigh_u[j]
                common = M[v] & M[x] & M[y]
                if common.bit_count() >= 2:
                    return False

        return True

    def add_edge(u, v):
        M[u] |= (1 << v)
        M[v] |= (1 << u)

    def remove_edge(u, v):
        M[u] &= ~(1 << v)
        M[v] &= ~(1 << u)

    def count_edges():
        return sum(M[i].bit_count() for i in range(n)) // 2

    def get_edge_list():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (M[u] >> v) & 1:
                    edges.append((u, v))
        return edges

    def get_missing_edges():
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((M[u] >> v) & 1):
                    missing.append((u, v))
        return missing

    def try_add_edge(u, v):
        if can_add(u, v):
            add_edge(u, v)
            return True
        return False

    def greedy_fill():
        changed = True
        while changed and time.time() < deadline:
            changed = False
            missing = get_missing_edges()
            random.shuffle(missing)
            for u, v in missing:
                if try_add_edge(u, v):
                    changed = True
    
    def local_search():
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any single edge
            missing = get_missing_edges()
            random.shuffle(missing)
            for u, v in missing:
                if try_add_edge(u, v):
                    improved = True
                    break
                    
            if improved:
                continue

            # 2. Try "Remove 1, Add 2" (Net +1)
            edges = get_edge_list()
            random.shuffle(edges)
            for a, b in edges:
                if time.time() > deadline:
                    return
                remove_edge(a, b)
                
                added = 0
                added_edges = []
                missing = get_missing_edges()
                random.shuffle(missing)
                
                for u, v in missing:
                    if added == 2:
                        break
                    if try_add_edge(u, v):
                        added += 1
                        added_edges.append((u, v))
                        
                if added == 2:
                    improved = True
                    break 
                else:
                    for u, v in added_edges:
                        remove_edge(u, v)
                    add_edge(a, b)
            
            if improved:
                continue

            # 3. Try "Remove 2, Add 3" (Net +1) - more expensive but can escape deeper local optima
            if time.time() < deadline - 5:
                edges = get_edge_list()
                random.shuffle(edges)
                # Limit number of 2-remove attempts to control time
                max_attempts = min(len(edges), 30)
                for ai in range(max_attempts):
                    if time.time() > deadline:
                        return
                    a1, b1 = edges[ai]
                    # Try all other edges as second removal
                    for bj in range(ai + 1, len(edges)):
                        if time.time() > deadline:
                            return
                        a2, b2 = edges[bj]
                        remove_edge(a1, b1)
                        remove_edge(a2, b2)
                        
                        added = 0
                        added_edges = []
                        missing = get_missing_edges()
                        random.shuffle(missing)
                        
                        for u, v in missing:
                            if added == 3:
                                break
                            if try_add_edge(u, v):
                                added += 1
                                added_edges.append((u, v))
                        
                        if added == 3:
                            improved = True
                            # Keep new state, don't restore
                            break
                        else:
                            for u, v in added_edges:
                                remove_edge(u, v)
                            add_edge(a1, b1)
                            add_edge(a2, b2)
                    
                    if improved:
                        break
            
            if improved:
                continue

            # 4. Try "Swap" (Remove 1, Add 1) to rearrange
            edges = get_edge_list()
            random.shuffle(edges)
            for a, b in edges:
                remove_edge(a, b)
                added = 0
                missing = get_missing_edges()
                random.shuffle(missing)
                for u, v in missing:
                    if try_add_edge(u, v):
                        added = 1
                        break
                if added == 1:
                    improved = True
                    break
                else:
                    add_edge(a, b)
                
            if improved:
                continue
                
            # 5. Kick: Randomly remove a few edges and greedy fill
            if time.time() < deadline:
                num_remove = random.randint(3, 8)
                edges = get_edge_list()
                random.shuffle(edges)
                for i in range(min(num_remove, len(edges))):
                    remove_edge(edges[i][0], edges[i][1])
                greedy_fill()
                improved = True

        return count_edges()

    best_edges = 0
    best_M = [0] * n
    random.seed(42)
    
    # Run multiple restarts
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        M = [0] * n
        
        # Random initialization
        density = 0.4
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    try_add_edge(u, v)
                    
        greedy_fill()
        local_search()
        
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]
            
        # If we hit a high number, we might stop early?
        if best_edges >= 108 and time.time() > start_time + 60:
            break

    # Final refinement with more aggressive search
    if best_edges > 0 and time.time() < deadline:
        M = best_M[:]
        local_search()
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]
        
        # Additional polish: try more local search iterations
        if time.time() < deadline - 2:
            M = best_M[:]
            # Run local search again with different random seed
            random.seed(12345)
            local_search()
            current = count_edges()
            if current > best_edges:
                best_edges = current
                best_M = M[:]

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_M[u] >> v) & 1:
                edges.append([u, v])
    return edges
