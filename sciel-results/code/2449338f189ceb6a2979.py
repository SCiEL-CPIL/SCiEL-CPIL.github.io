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
    
    # For n <= 12, brute force over all 2^n colorings is feasible and often faster 
    # than complex backtracking for dense graphs.
    # Optimization: Precompute nothing, just iterate.
    # We check if there exists a mask such that no edge is monochromatic.
    # Edge e is monochromatic if (e & mask) == 0 or (e & mask) == e.
    # We want a mask where for ALL e, 0 < (e & mask).bit_count() < 5.
    
    # If n is small, we can just loop.
    # For n=9, 2^9 = 512 masks.
    # For n=10, 2^10 = 1024 masks.
    # For n=11, 2^11 = 2048 masks.
    # For n=12, 2^12 = 4096 masks.
    # This is very fast in Python if the inner loop is tight.
    
    if n <= 12:
        # Optimization: If we have many edges, we can break early.
        # Also, we can order edges by frequency or just use the list as is.
        for mask in range(1 << n):
            # Skip mask 0 and mask (1<<n)-1? No, they are valid colorings (all 0 or all 1).
            # Wait, if mask=0, then (e & 0) = 0, so edge is monochromatic (all 0). 
            # So mask=0 is never a valid 2-coloring for any non-empty hypergraph.
            # Similarly for mask = (1<<n)-1.
            # But we can just check.
            
            valid = True
            for e in edges_bits:
                # Check if e is monochromatic under mask
                # monochromatic if (e & mask) == 0 OR (e & mask) == e
                # Equivalent to: 0 < popcount(e & mask) < 5
                # Using bit_count is fast.
                c = (e & mask).bit_count()
                if c == 0 or c == 5:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        # Fallback for larger n: Backtracking with unit propagation
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
    deadline = start_time + 140.0  # Safety margin
    
    best_n = 0
    best_edges = []
    best_len = float('inf')
    
    def minimize(n, edges_bits, time_limit):
        """
        Greedy minimization: try to remove each edge if the graph remains non-2-colorable.
        """
        current_edges = list(edges_bits)
        random.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges):
            if time.time() > time_limit:
                break
            
            e = current_edges[i]
            # Remove edge e
            # To avoid O(N) deletion, we can swap with last and pop, but we need to keep order?
            # No, order doesn't matter for the set.
            # However, if we swap, we disturb the iteration.
            # Simple approach: create a new list without e.
            # For N ~ 100-200, this is fine.
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if not _is_2_colorable_bitmask(n, test_edges):
                # Can remove
                current_edges = test_edges
                # Do not increment i, check the next edge (which is now at i)
            else:
                i += 1
                
        return current_edges

    # Strategy 1: Start from K_9^5
    # K_9^5 has 126 edges.
    n = 9
    all_edges_9 = [sum(1 << v for v in e) for e in combinations(range(9), 5)]
    
    # Multiple restarts with different random shuffles
    # We want to find the global minimum, which is 51.
    # Greedy removal is prone to local minima.
    
    for restart in range(20):  # Increase restarts if checker is fast
        if time.time() > deadline - 10:
            break
        
        # Allocate time for this restart
        # If we have 140s total, and 20 restarts, ~6s each?
        # But early restarts might finish faster.
        # Let's use a dynamic time limit.
        remaining = deadline - time.time()
        t_limit = time.time() + max(1.0, remaining / (20 - restart))
        
        minimized = minimize(n, all_edges_9, t_limit)
        if len(minimized) < best_len:
            best_len = len(minimized)
            best_n = n
            best_edges = minimized
            
            # If we hit 51, we can stop searching for better on n=9
            if best_len <= 51:
                # We found the theoretical minimum. 
                # We can break to save time or try to verify quickly.
                # Since 51 is the known best, we can break.
                break

    # If we didn't find 51, maybe try n=10?
    # m(5) is the minimum edges over all n. 
    # Usually smaller n yields smaller m for fixed uniformity? 
    # Not necessarily, but K_9^5 is a strong candidate.
    # Let's stick to n=9 primarily as it's the standard construction size.
    
    if best_len == float('inf'):
        # Fallback: return K_9^5
        best_n = 9
        best_edges = all_edges_9

    # Convert bitmasks back to lists of vertices
    final_edges = []
    for e in best_edges:
        verts = [i for i in range(best_n) if (e >> i) & 1]
        final_edges.append(verts)

    return {'n': best_n, 'edges': final_edges}
