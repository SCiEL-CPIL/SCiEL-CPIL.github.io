import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115  # Leave 5s margin
    
    # Fallback: K_9^5 (126 edges) or Star on 10 (126 edges)
    # K_9^5 is simpler to generate.
    def get_fallback():
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    best_result = get_fallback()
    best_edge_count = len(best_result['edges'])
    
    # Fast Solver using bitmasks
    # n <= 12 for search
    def is_non_2_colorable(n, edges):
        m = len(edges)
        if m == 0:
            return False
        
        # Precompute bitmasks
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Vertex to edge index mapping
        v_to_e = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_to_e[v].append(i)
        
        full_mask = (1 << n) - 1
        
        def dfs(mask0, mask1):
            # Check for conflicts and propagate
            changed = True
            while changed:
                changed = False
                for e_idx, mask in enumerate(edge_masks):
                    c0 = bin(mask & mask0).count('1')
                    c1 = bin(mask & mask1).count('1')
                    
                    if c0 == 5 or c1 == 5:
                        return False
                    
                    if c0 == 4:
                        unassigned = mask & ~(mask0 | mask1)
                        if unassigned:
                            # Force to 1
                            if mask1 & unassigned:
                                return False # Already assigned 1? No, unassigned means not in mask0/1
                            # Wait, if unassigned has multiple bits, it's a problem?
                            # If c0=4, 1 vertex is unassigned.
                            # So unassigned should have exactly 1 bit.
                            mask1 |= unassigned
                            changed = True
                            
                    elif c1 == 4:
                        unassigned = mask & ~(mask0 | mask1)
                        if unassigned:
                            mask0 |= unassigned
                            changed = True
            
            unassigned = full_mask & ~(mask0 | mask1)
            if unassigned == 0:
                return True # Valid coloring found
            
            # Heuristic: Pick vertex with most constraints?
            # Simple: First unassigned
            v_bit = unassigned & -unassigned
            v = v_bit.bit_length() - 1
            
            # Try 0
            if dfs(mask0 | v_bit, mask1):
                return True
            # Try 1
            if dfs(mask0, mask1 | v_bit):
                return True
            
            return False
        
        if dfs(0, 0):
            return False
        else:
            return True

    # Search for smaller hypergraphs
    # Try n=10, 11
    for n in [10, 11]:
        if time.time() > deadline:
            break
            
        # Randomized construction
        # Start with a random set of edges, check if non-2-colorable.
        # If yes, minimize.
        # If no, add more edges.
        
        # Better: Start with a known non-2-colorable core?
        # We don't have one.
        # Let's try random sampling of edges from K_n^5.
        
        all_edges = list(combinations(range(n), 5))
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Try a random subset of edges
            # Size heuristic: Maybe around 50-100?
            # For n=11, K_11^5 has 462 edges.
            # Let's try random 60 edges.
            m = random.randint(40, 80)
            random.shuffle(all_edges)
            candidate_edges = [list(e) for e in all_edges[:m]]
            
            if is_non_2_colorable(n, candidate_edges):
                # Minimize
                current_edges = [list(e) for e in candidate_edges]
                improved = True
                while improved:
                    improved = False
                    random.shuffle(current_edges) # Randomize order for removal
                    for i in range(len(current_edges)):
                        if time.time() > deadline:
                            break
                        test_edges = [e for j, e in enumerate(current_edges) if j != i]
                        if is_non_2_colorable(n, test_edges):
                            current_edges = test_edges
                            improved = True
                            break
                    
                    if time.time() > deadline:
                        break
                
                if len(current_edges) < best_edge_count:
                    best_edge_count = len(current_edges)
                    best_result = {'n': n, 'edges': current_edges}
                    
    return best_result
