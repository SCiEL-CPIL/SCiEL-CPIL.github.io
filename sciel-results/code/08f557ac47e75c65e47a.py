import time
import random
from itertools import combinations

def _is_2_colorable(n, edges, timeout=10.0):
    """Check if hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable (has a proper coloring), False if not, None on timeout."""
    if not edges:
        return True  # trivially 2-colorable
    
    start_time = time.time()
    
    # Convert edges to tuples for efficiency
    edge_list = [tuple(sorted(e)) for e in edges]
    
    # Build incidence structure
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_list):
        for v in e:
            vert_edges[v].append(i)
    
    # color[v] = -1 unassigned, 0 or 1
    color = [-1] * n
    
    def unit_propagate():
        """Apply unit propagation. Returns False if contradiction detected."""
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_list):
                c0 = c1 = 0
                unassigned = []
                for v in e:
                    if color[v] == 0:
                        c0 += 1
                    elif color[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                # If all assigned vertices in edge have same color, unassigned must be opposite
                if len(unassigned) == 1:
                    if c0 == 4 and c1 == 0:
                        v = unassigned[0]
                        if color[v] == 0:
                            return False
                        elif color[v] == -1:
                            color[v] = 1
                            changed = True
                    elif c1 == 4 and c0 == 0:
                        v = unassigned[0]
                        if color[v] == 1:
                            return False
                        elif color[v] == -1:
                            color[v] = 0
                            changed = True
                # Check for monochromatic edge (all assigned, all same color)
                if not unassigned:
                    if c0 == 5 or c1 == 5:
                        return False
        return True
    
    def backtrack(idx):
        """Try to find a valid 2-coloring. Returns True if found, False if not, None on timeout."""
        if time.time() - start_time > timeout:
            return None
        
        if idx == n:
            # Verify all edges are bichromatic
            for e in edge_list:
                c0 = sum(1 for v in e if color[v] == 0)
                c1 = sum(1 for v in e if color[v] == 1)
                if c0 == 0 or c1 == 0:
                    return False
            return True
        
        # Try color 0
        color[idx] = 0
        if unit_propagate():
            result = backtrack(idx + 1)
            if result is not None:
                return result
        color[idx] = -1
        
        # Try color 1
        color[idx] = 1
        if unit_propagate():
            result = backtrack(idx + 1)
            if result is not None:
                return result
        color[idx] = -1
        
        return False
    
    # Initial propagation
    if not unit_propagate():
        return False
    
    result = backtrack(0)
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_result = None
    best_count = float('inf')
    
    # Strategy 1: Start from complete 5-uniform on 9 vertices, greedily remove edges
    # This is guaranteed to work since K_9^5 is non-2-colorable
    
    def try_reduce(n, initial_edges, time_budget):
        """Try to reduce edge count by greedily removing edges."""
        current_edges = list(initial_edges)
        current_set = set(tuple(sorted(e)) for e in current_edges)
        
        # Try removing edges in random order
        random.shuffle(current_edges)
        
        for e in current_edges:
            if time.time() > start_time + time_budget:
                break
            e_tuple = tuple(sorted(e))
            if e_tuple not in current_set:
                continue
            if len(current_set) <= 28:
                break  # Can't go below 29 (all ≤28 are 2-colorable)
            
            candidate_set = current_set - {e_tuple}
            candidate_edges = [list(e) for e in candidate_set]
            
            result = _is_2_colorable(n, candidate_edges, timeout=5.0)
            if result is False:  # Still non-2-colorable
                current_set = candidate_set
        
        return list(current_set)
    
    # Phase 1: Reduce from K_9^5
    k9_edges = list(combinations(range(9), 5))
    
    for restart in range(3):
        if time.time() > start_time + 100:
            break
        
        budget = 30.0 if restart == 0 else 20.0
        reduced = try_reduce(9, k9_edges, budget)
        
        if len(reduced) < best_count:
            best_count = len(reduced)
            best_result = {'n': 9, 'edges': [list(e) for e in reduced]}
    
    # Phase 2: Try random constructions on different n values
    for n in [10, 11, 12, 13, 14, 15]:
        if time.time() > start_time + 130:
            break
        
        all_possible = list(combinations(range(n), 5))
        
        for attempt in range(5):
            if time.time() > start_time + 135:
                break
            
            # Start with a subset of random edges
            random.shuffle(all_possible)
            num_edges = min(n * 4, len(all_possible))
            selected = all_possible[:num_edges]
            
            result = _is_2_colorable(n, selected, timeout=10.0)
            if result is False:
                # Found non-2-colorable, try to reduce
                reduced = try_reduce(n, selected, time_budget=10.0)
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_result = {'n': n, 'edges': [list(e) for e in reduced]}
    
    # Phase 3: If we haven't found anything good, use the reduced K_9^5
    if best_result is None:
        # Fallback: return K_9^5 (126 edges) - guaranteed valid
        best_result = {'n': 9, 'edges': [list(e) for e in k9_edges]}
    
    # Final verification
    final_edges = best_result['edges']
    final_n = best_result['n']
    verify = _is_2_colorable(final_n, final_edges, timeout=15.0)
    if verify is not False:
        # If verification fails or times out, fall back to K_9^5
        best_result = {'n': 9, 'edges': [list(e) for e in k9_edges]}
    
    return best_result
