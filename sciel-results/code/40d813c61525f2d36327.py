import time
import random
import itertools

def construct():
    deadline = time.time() + 115
    
    # Fallback solution for n=9
    n_fallback = 9
    colors_fallback = [min(j - i, 9 - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    
    # Try to find larger n
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline:
            break
        
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return i * (i - 1) // 2 + (j - i - 1)
        
        num_edges = n * (n - 1) // 2
        
        # Build cycle list
        cycles = []
        edge_to_cycles = [[] for _ in range(num_edges)]
        
        for combo in itertools.combinations(range(n), 6):
            v0 = combo[0]
            rest = list(combo[1:])
            for perm in itertools.permutations(rest):
                v1, v2, v3, v4, v5 = perm
                if v1 >= v5:
                    continue
                edges = [
                    edge_idx(v0, v1),
                    edge_idx(v1, v2),
                    edge_idx(v2, v3),
                    edge_idx(v3, v4),
                    edge_idx(v4, v5),
                    edge_idx(v5, v0)
                ]
                idx = len(cycles)
                cycles.append(edges)
                for e in edges:
                    edge_to_cycles[e].append(idx)
        
        # Local Search
        def solve_local_search():
            start_time = time.time()
            # Random restarts
            for _ in range(10):
                if time.time() > deadline:
                    return None
                
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                
                # Init cost
                cost = [0, 0, 0, 0]
                for c_idx, edges in enumerate(cycles):
                    c0 = colors[edges[0]]
                    if all(colors[e] == c0 for e in edges[1:]):
                        cost[c0] += 1
                
                total_cost = sum(cost)
                if total_cost == 0:
                    return colors
                
                best_cost = total_cost
                best_colors = colors[:]
                
                # Steps
                for step in range(2000):
                    if time.time() > deadline:
                        break
                    
                    e = random.randint(0, num_edges - 1)
                    c_old = colors[e]
                    
                    best_delta = float('inf')
                    best_c_new = c_old
                    best_break = 0
                    best_make = 0
                    
                    for c_new in range(4):
                        if c_new == c_old:
                            continue
                        
                        break_count = 0
                        make_count = 0
                        
                        for k in edge_to_cycles[e]:
                            edges_in_cycle = cycles[k]
                            all_old = True
                            all_new = True
                            for e2 in edges_in_cycle:
                                if e2 == e:
                                    continue
                                if colors[e2] != c_old:
                                    all_old = False
                                if colors[e2] != c_new:
                                    all_new = False
                                if not all_old and not all_new:
                                    break
                            
                            if all_old:
                                break_count += 1
                            if all_new:
                                make_count += 1
                        
                        delta = -break_count + make_count
                        if delta < best_delta:
                            best_delta = delta
                            best_c_new = c_new
                            best_break = break_count
                            best_make = make_count
                    
                    if best_c_new != c_old:
                        colors[e] = best_c_new
                        cost[c_old] -= best_break
                        cost[best_c_new] += best_make
                        total_cost = sum(cost)
                        
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_colors = colors[:]
                        
                        if total_cost == 0:
                            return colors
                    else:
                        # Random kick
                        if random.random() < 0.05:
                            c_rand = random.choice([c for c in range(4) if c != c_old])
                            # Update cost for random kick
                            # (Simplified: just recalculate cost? No, too slow.
                            # Just accept and hope it helps escape)
                            # Actually, let's just restart if stuck for a while?
                            # For now, just skip.
                        pass
                    
                if best_cost == 0:
                    return best_colors
            return None
        
        result = solve_local_search()
        if result is not None:
            return {'n': n, 'colors': result}
            
    return {'n': n_fallback, 'colors': colors_fallback}
