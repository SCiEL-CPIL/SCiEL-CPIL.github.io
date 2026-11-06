import time
import random

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4s using optimized backtracking.
    """
    
    def solve(N, time_limit=100):
        start_time = time.time()
        
        # neighbors[c][u] is a bitmask of v such that edge (u,v) is color c
        neighbors = [ [0] * N for _ in range(5) ]
        
        # Precompute bit masks for single vertices
        bit_masks = [1 << i for i in range(N)]
        
        # List of edges
        edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        
        def is_valid(u, v, c):
            # Check if u and v have < 2 common neighbors in color c
            common = neighbors[c][u] & neighbors[c][v]
            return common.bit_count() < 2
        
        def backtrack(unassigned_edges, assignment):
            if time.time() - start_time > time_limit:
                return None
            
            if not unassigned_edges:
                return assignment.copy()
            
            # MRV Heuristic: Find edge with fewest valid colors
            # Optimization: Only check a random subset of edges if list is large?
            # For N=26, E=325. Checking all is ~1600 ops.
            # Let's check all to be safe.
            
            best_idx = -1
            min_options = 5
            best_edge = None
            best_valid_colors = None
            
            # Randomize the order of checking edges to break symmetry?
            # The list is shuffled initially. We don't shuffle inside the loop to save time.
            
            for idx in range(len(unassigned_edges)):
                u, v = unassigned_edges[idx]
                valid_colors = []
                for c in range(5):
                    if is_valid(u, v, c):
                        valid_colors.append(c)
                
                if len(valid_colors) < min_options:
                    min_options = len(valid_colors)
                    best_idx = idx
                    best_edge = (u, v)
                    best_valid_colors = valid_colors
                    
                    if min_options == 0:
                        return None
                    if min_options == 1:
                        break
                        
            if best_idx == -1:
                return None
                
            u, v = best_edge
            # Remove edge
            unassigned_edges[best_idx] = unassigned_edges[-1]
            unassigned_edges.pop()
            
            random.shuffle(best_valid_colors)
            
            for c in best_valid_colors:
                # Assign
                neighbors[c][u] |= bit_masks[v]
                neighbors[c][v] |= bit_masks[u]
                assignment[(u,v)] = c
                
                res = backtrack(unassigned_edges, assignment)
                if res is not None:
                    return res
                
                # Undo
                neighbors[c][u] &= ~bit_masks[v]
                neighbors[c][v] &= ~bit_masks[u]
                del assignment[(u,v)]
                
            return None

        # Try multiple restarts
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            
            neighbors = [ [0] * N for _ in range(5) ]
            assignment = {}
            current_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
            random.shuffle(current_edges)
            
            result = backtrack(current_edges, assignment)
            if result is not None:
                final_colors = []
                for i in range(N):
                    for j in range(i+1, N):
                        final_colors.append(result[(i,j)])
                return {'n': N, 'colors': final_colors}
                
        return None

    total_time_limit = 110
    start_total = time.time()
    
    # Try high N first
    for N in range(26, 19, -1):
        time_remaining = total_time_limit - (time.time() - start_total)
        if time_remaining < 5:
            break
        alloc_time = min(15, time_remaining)
        res = solve(N, time_limit=alloc_time)
        if res is not None:
            return res
            
    # Fallback: Return a small valid solution to ensure non-zero score
    # N=11 solution from seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
