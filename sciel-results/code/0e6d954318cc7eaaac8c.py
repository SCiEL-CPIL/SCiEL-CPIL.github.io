import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_edges = None
    best_n = 9
    best_count = 126
    
    def is_2_colorable(n, edges):
        # Returns True if 2-colorable, False if not
        if not edges:
            return True
            
        # Precompute vertex-edge incidence
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
                
        # State: colors (list of -1, 0, 1), edge_counts (list of uncolored counts per edge)
        init_colors = [-1] * n
        init_counts = [5] * len(edges)
        
        def solve(colors, counts):
            # Check for conflicts in current state
            for i, e in enumerate(edges):
                c0 = 0
                c1 = 0
                uncol = 0
                for v in e:
                    if colors[v] == 0: c0 += 1
                    elif colors[v] == 1: c1 += 1
                    else: uncol += 1
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and uncol == 1:
                    # Force the last one to be 1
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 1
                            counts[i] -= 1
                            break
                elif c1 == 4 and uncol == 1:
                    for v in e:
                        if colors[v] == -1:
                            colors[v] = 0
                            counts[i] -= 1
                            break
                            
            # Find unassigned vertex
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
                    
            if v == -1:
                return True
                
            # Try 0
            colors[v] = 0
            # Update counts
            affected = vertex_edges[v]
            for ei in affected:
                counts[ei] -= 1
            if solve(colors, counts):
                return True
            # Undo
            colors[v] = -1
            for ei in affected:
                counts[ei] += 1
                
            # Try 1
            colors[v] = 1
            for ei in affected:
                counts[ei] -= 1
            if solve(colors, counts):
                return True
            # Undo
            colors[v] = -1
            for ei in affected:
                counts[ei] += 1
                
            return False

        return solve(init_colors, init_counts)

    # Search
    for n in range(10, 14):
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        
        for trial in range(50):
            if time.time() > deadline:
                break
                
            random.shuffle(all_edges)
            # Start with a random subset
            k = random.randint(40, 60)
            current_edges = [list(e) for e in all_edges[:k]]
            
            if not is_2_colorable(n, current_edges):
                # Minimize
                edges_list = [list(e) for e in current_edges]
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    random.shuffle(edges_list)
                    for i in range(len(edges_list)):
                        e_to_remove = edges_list[i]
                        new_edges = [e for j, e in enumerate(edges_list) if j != i]
                        if not is_2_colorable(n, new_edges):
                            edges_list = new_edges
                            improved = True
                            break
                            
                if len(edges_list) < best_count:
                    best_count = len(edges_list)
                    best_n = n
                    best_edges = edges_list

    if best_edges:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
