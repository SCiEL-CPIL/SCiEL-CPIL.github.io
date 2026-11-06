import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    def is_2_colorable_bitmask(n, edge_masks):
        """Check if hypergraph has a proper 2-colouring.
        Returns True if 2-colourable, False if not (UNSAT).
        Uses bitmask enumeration of all 2^n colorings."""
        m = len(edge_masks)
        if m == 0:
            return True
        
        # Check all non-trivial colorings (exclude all-0 and all-1)
        for c in range(1, (1 << n) - 1):
            valid = True
            for em in edge_masks:
                # Edge is monochromatic if all bits same
                if (c & em) == 0 or (c & em) == em:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    def greedy_eliminate(n, max_edges=200):
        """Greedily select edges to eliminate all 2-colorings.
        Returns list of edge tuples or None."""
        all_edges = list(combinations(range(n), 5))
        
        # Convert to masks
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Start with all valid colorings (exclude trivial all-0 and all-1)
        valid_colorings = set(range(1, (1 << n) - 1))
        
        selected_edges = []
        selected_masks = []
        selected_set = set()
        
        while valid_colorings and len(selected_edges) < max_edges:
            if time.time() > deadline - 5:
                return None
            
            best_edge_idx = -1
            best_kill_count = 0
            
            # Find edge that kills the most colorings
            for i, e in enumerate(all_edges):
                if e in selected_set:
                    continue
                
                em = edge_masks[i]
                kill_count = 0
                for c in valid_colorings:
                    if (c & em) == 0 or (c & em) == em:
                        kill_count += 1
                
                if kill_count > best_kill_count:
                    best_kill_count = kill_count
                    best_edge_idx = i
            
            if best_edge_idx == -1 or best_kill_count == 0:
                break
            
            # Add the best edge
            best_edge = all_edges[best_edge_idx]
            best_mask = edge_masks[best_edge_idx]
            selected_edges.append(best_edge)
            selected_masks.append(best_mask)
            selected_set.add(best_edge)
            
            # Remove killed colorings
            new_valid = set()
            for c in valid_colorings:
                if not ((c & best_mask) == 0 or (c & best_mask) == best_mask):
                    new_valid.add(c)
            valid_colorings = new_valid
        
        if not valid_colorings:
            return selected_edges
        return None
    
    def random_search(n, target_edges, trials=100):
        """Randomly sample edge sets and check if non-2-colorable."""
        all_edges = list(combinations(range(n), 5))
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        for _ in range(trials):
            if time.time() > deadline - 3:
                break
            
            indices = random.sample(range(len(all_edges)), target_edges)
            selected_masks = [edge_masks[i] for i in indices]
            
            if not is_2_colorable_bitmask(n, selected_masks):
                return [all_edges[i] for i in indices]
        
        return None
    
    best_result = None
    best_m = 201
    
    # Try greedy elimination for n=10, 11, 12
    for n in [10, 11, 12]:
        if time.time() > deadline - 15:
            break
        
        result = greedy_eliminate(n, max_edges=60)
        if result is not None:
            m = len(result)
            if m < best_m:
                best_m = m
                best_result = (n, result)
            if m <= 51:
                return {'n': n, 'edges': [list(e) for e in result]}
    
    # If greedy didn't find <=51, try random search
    for n in [10, 11, 12]:
        if time.time() > deadline - 10:
            break
        
        for target in [45, 50, 51, 55]:
            if time.time() > deadline - 5:
                break
            result = random_search(n, target, trials=200)
            if result is not None:
                m = len(result)
                if m < best_m:
                    best_m = m
                    best_result = (n, result)
                if m <= 51:
                    return {'n': n, 'edges': [list(e) for e in result]}
    
    # Fallback: return best found
    if best_result is not None:
        n, edges = best_result
        return {'n': n, 'edges': [list(e) for e in edges]}
    
    # Absolute fallback: complete hypergraph on 9 vertices
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
