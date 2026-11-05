import random
import time

# EVOLVE-BLOCK-START
def construct():
    # Precomputation
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
    num_edges = len(edges) # 120
    
    k4s = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    k4s.append((i, j, k, l))
    num_k4s = len(k4s) # 1820
    
    # Map edge index to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for idx, k4 in enumerate(k4s):
        i, j, k, l = k4
        e1 = edge_index[(i, j)]
        e2 = edge_index[(i, k)]
        e3 = edge_index[(i, l)]
        e4 = edge_index[(j, k)]
        e5 = edge_index[(j, l)]
        e6 = edge_index[(k, l)]
        edge_to_k4s[e1].append(idx)
        edge_to_k4s[e2].append(idx)
        edge_to_k4s[e3].append(idx)
        edge_to_k4s[e4].append(idx)
        edge_to_k4s[e5].append(idx)
        edge_to_k4s[e6].append(idx)

    def count_distinct_colors(k4_idx, colors):
        i, j, k, l = k4s[k4_idx]
        c1 = colors[edge_index[(i, j)]]
        c2 = colors[edge_index[(i, k)]]
        c3 = colors[edge_index[(i, l)]]
        c4 = colors[edge_index[(j, k)]]
        c5 = colors[edge_index[(j, l)]]
        c6 = colors[edge_index[(k, l)]]
        return len({c1, c2, c3, c4, c5, c6})

    def is_k4_valid(k4_idx, colors):
        return count_distinct_colors(k4_idx, colors) >= 5

    def local_search(k, time_limit=10):
        # Initialize randomly
        colors = [random.randint(0, k - 1) for _ in range(num_edges)]
        
        # Calculate initial state
        is_bad = [not is_k4_valid(idx, colors) for idx in range(num_k4s)]
        total_bad = sum(is_bad)
        
        if total_bad == 0:
            return colors
            
        start_time = time.time()
        T = 1.0
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if total_bad == 0:
                return colors
                
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue
                
            # Calculate delta
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                was_bad = is_bad[k4_idx]
                # Temporarily change color
                colors[e_idx] = new_color
                now_valid = is_k4_valid(k4_idx, colors)
                now_bad = not now_valid
                # Revert
                colors[e_idx] = old_color
                
                if was_bad and not now_bad:
                    delta -= 1
                elif not was_bad and now_bad:
                    delta += 1
                    
            # Acceptance criterion
            if delta < 0:
                colors[e_idx] = new_color
                # Update is_bad
                for k4_idx in edge_to_k4s[e_idx]:
                    # Recalculate or track? 
                    # Since we know delta, we can update is_bad if we tracked which ones changed.
                    # Simpler: just recalculate is_bad for affected k4s? 
                    # Or just update total_bad and is_bad based on the check we did?
                    # We didn't store the 'now_bad' values. Let's just recalculate is_bad for the affected k4s.
                    pass 
                # For simplicity and correctness, if we accept, we update is_bad for all affected k4s.
                for k4_idx in edge_to_k4s[e_idx]:
                    is_bad[k4_idx] = not is_k4_valid(k4_idx, colors)
                total_bad += delta
            else:
                # Simulated annealing
                if random.random() < pow(2.71828, -delta / T):
                    colors[e_idx] = new_color
                    for k4_idx in edge_to_k4s[e_idx]:
                        is_bad[k4_idx] = not is_k4_valid(k4_idx, colors)
                    total_bad += delta
            T *= cooling
            
        return None

    best_solution = None
    
    # Try 15, 14, 13
    for k in [15, 14, 13]:
        sol = local_search(k, time_limit=10)
        if sol is not None:
            best_solution = sol
            # If we found 13, we can stop as it's the lower bound
            if k == 13:
                break

    if best_solution is not None:
        return best_solution
    else:
        # Fallback to seed's 17-coloring
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
