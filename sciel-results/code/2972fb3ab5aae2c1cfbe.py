import time
import random
from itertools import combinations

def _is_2_colorable_bitmask(n, edges_bits):
    """
    Checks if the 5-uniform hypergraph is 2-colorable using bitmasks.
    n: number of vertices
    edges_bits: list of integers, each representing an edge as a bitmask
    Returns True if 2-colorable, False otherwise.
    """
    if not edges_bits:
        return True
    
    # For n <= 12, brute force is very fast
    if n <= 12:
        for mask in range(1 << n):
            is_valid = True
            for e in edges_bits:
                # Count bits in (e & mask) and (e & ~mask)
                # If all 5 bits are in mask or all 5 are in ~mask, it's monochromatic
                # We want to check if NO edge is monochromatic.
                # Edge is monochromatic if (e & mask) == e OR (e & mask) == 0
                # i.e., all vertices in edge are 1 OR all are 0.
                
                # Optimization: 
                # c1 = popcount(e & mask)
                # if c1 == 0 or c1 == 5: monochromatic
                c1 = bin(e & mask).count('1')
                if c1 == 0 or c1 == 5:
                    is_valid = False
                    break
            if is_valid:
                return True
        return False
    else:
        # Fallback for larger n (not expected to be used for optimal search)
        # Simple backtracking
        colors = [-1] * n
        edges_list = []
        for e in edges_bits:
            verts = [i for i in range(n) if (e >> i) & 1]
            edges_list.append(verts)
            
        def propagate():
            changed = True
            while changed:
                changed = False
                for edge in edges_list:
                    c0 = 0
                    c1 = 0
                    unassigned = []
                    for v in edge:
                        if colors[v] == 0:
                            c0 += 1
                        elif colors[v] == 1:
                            c1 += 1
                        else:
                            unassigned.append(v)
                    
                    if c0 == 5 or c1 == 5:
                        return False
                    
                    if c0 == 4 and len(unassigned) == 1:
                        if colors[unassigned[0]] == -1:
                            colors[unassigned[0]] = 1
                            changed = True
                    elif c1 == 4 and len(unassigned) == 1:
                        if colors[unassigned[0]] == -1:
                            colors[unassigned[0]] = 0
                            changed = True
            return True

        def backtrack():
            if not propagate():
                return False
            
            next_v = -1
            for v in range(n):
                if colors[v] == -1:
                    next_v = v
                    break
            
            if next_v == -1:
                return True
            
            colors[next_v] = 0
            if backtrack():
                return True
            
            colors[next_v] = 1
            if backtrack():
                return True
            
            colors[next_v] = -1
            return False

        return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_n = 0
    best_edges = []
    best_len = float('inf')
    
    def minimize(n, edges_bits, time_limit):
        current_edges = list(edges_bits)
        random.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges):
            if time.time() > time_limit:
                break
            
            e = current_edges[i]
            # Remove edge e
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if not _is_2_colorable_bitmask(n, test_edges):
                current_edges = test_edges
                # Do not increment i, check the next edge in the new list (which is at index i)
            else:
                i += 1
                
        return current_edges

    # Try n=9
    # K_9^5 has 126 edges
    all_edges_9 = [sum(1 << v for v in e) for e in combinations(range(9), 5)]
    
    # Run minimization on K_9^5
    # Do a few random restarts to ensure we find the global minimum (51 edges)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        t_limit = min(time.time() + 10.0, deadline - 10.0)
        minimized = minimize(9, all_edges_9, t_limit)
        if len(minimized) < best_len:
            best_len = len(minimized)
            best_n = 9
            best_edges = minimized
            # If we found 51, we can likely stop or try n=10 for safety
            if best_len <= 51:
                # We hit the theoretical minimum, no need for more restarts on n=9
                break

    # Try n=10 if time permits and we haven't found 51
    if time.time() < deadline - 20 and best_len > 51:
        all_edges_10 = [sum(1 << v for v in e) for e in combinations(range(10), 5)]
        # K_10^5 has 252 edges. Minimization might be slower.
        # Try a few random subsets of size ~100-150 to see if we find something smaller
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            # Pick a random subset of K_10^5
            k = 150
            random.shuffle(all_edges_10)
            candidate = all_edges_10[:k]
            # Check if candidate is non-2-colorable
            if not _is_2_colorable_bitmask(10, candidate):
                t_limit = min(time.time() + 5.0, deadline - 10.0)
                minimized = minimize(10, candidate, t_limit)
                if len(minimized) < best_len:
                    best_len = len(minimized)
                    best_n = 10
                    best_edges = minimized

    if best_len == float('inf'):
        # Fallback: K_9^5
        best_n = 9
        best_edges = all_edges_9

    # Convert bitmasks back to lists of vertices
    final_edges = []
    for e in best_edges:
        verts = [i for i in range(best_n) if (e >> i) & 1]
        final_edges.append(verts)

    return {'n': best_n, 'edges': final_edges}
