import itertools
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def is_2_colorable(n, edges):
        """
        Returns True if the hypergraph (n vertices, edges) is 2-colorable.
        2-colorable means there exists a coloring c: V -> {0,1} such that
        for every edge e, not all vertices in e have the same color.
        """
        # Iterate over all 2^n colorings
        for mask in range(1 << n):
            valid = True
            for e in edges:
                # Check if edge e is monochromatic
                # Get colors of vertices in e
                # If all bits at positions e[i] are 0 or all are 1, it's monochromatic
                # We can check this by:
                # val = sum((mask >> v) & 1 for v in e)
                # if val == 0 or val == len(e): monochromatic
                val = 0
                for v in e:
                    val += (mask >> v) & 1
                if val == 0 or val == len(e):
                    valid = False
                    break
            if valid:
                return True
        return False

    best_result = None
    best_edge_count = 1000
    
    # Try n=9 and n=10
    for n in [9, 10]:
        if time.time() > deadline:
            break
            
        all_edges = list(itertools.combinations(range(n), 5))
        current_edges = set(all_edges)
        
        # Greedy deletion
        # Shuffle edges to get different local minima
        # We can do a few restarts if time permits
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Shuffle the order of removal attempts
            edges_list = list(current_edges)
            random.shuffle(edges_list)
            
            # Reset to full graph for this restart? 
            # No, greedy deletion is path-dependent. 
            # Starting from full graph every time is safer for finding a good minimal subgraph.
            current_edges = set(all_edges)
            
            # Perform deletion
            for e in edges_list:
                if time.time() > deadline:
                    break
                candidate = current_edges - {e}
                if not is_2_colorable(n, candidate):
                    current_edges = candidate
            
            if len(current_edges) < best_edge_count:
                best_edge_count = len(current_edges)
                best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
                
    if best_result is None:
        # Fallback to seed
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
        
    return best_result
