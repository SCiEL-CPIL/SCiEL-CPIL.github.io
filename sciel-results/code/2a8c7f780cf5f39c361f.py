import itertools
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140  # 10s safety margin
    
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    def is_not_2_colorable(edges):
        """Returns True if the hypergraph is NOT 2-colorable (no proper 2-coloring exists)."""
        m = len(edges)
        if m == 0:
            return False
        
        full_mask = (1 << n) - 1
        
        # Precompute for each edge: the bitmask of its 5 vertices
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Use iterative DFS with explicit stack to avoid recursion limit issues
        # State: (color0, color1, edge_index)
        # We'll use a more efficient approach with memoization on (color0, color1)
        
        # Actually, let's keep the recursive approach but optimize it
        # Key optimization: use a list of edge masks and track which edges are "satisfied"
        
        def check(color0, color1):
            """
            Returns True if a valid 2-coloring exists (hypergraph IS 2-colorable)
            Returns False if no valid 2-coloring exists (NOT 2-colorable)
            """
            # Use an iterative approach with explicit stack
            # Each stack entry: (color0, color1)
            # We need to handle backtracking properly
            
            # Let's use a different approach: convert to SAT-like structure
            # For each edge, at most 4 vertices can have the same color
            
            # Use recursive with careful state management
            # Optimize by precomputing edge vertex sets as frozensets for fast lookup
            
            edge_vertex_sets = [frozenset(e) for e in edges]
            
            # Use a more efficient backtracking with pruning
            # Track for each edge: count of 0s, count of 1s, count of unassigned
            edge_state = []
            for mask in edge_masks:
                c0 = bin(mask & color0).count('1')
                c1 = bin(mask & color1).count('1')
                un = 5 - c0 - c1
                edge_state.append([c0, c1, un])
            
            # Propagate unit clauses
            def propagate(edge_state, color0, color1):
                """
                Returns (success, new_color0, new_color1, new_edge_state)
                success=False means conflict detected
                """
                changed = True
                while changed:
                    changed = False
                    for i in range(len(edge_state)):
                        c0, c1, un = edge_state[i]
                        
                        if c0 == 5 or c1 == 5:
                            return (False, color0, color1, edge_state)
                        
                        if c0 == 4 and un == 1:
                            # The unassigned vertex must be color 1
                            mask = edge_masks[i]
                            unmask = mask & ~(color0 | color1)
                            if unmask:
                                if unmask & color1:
                                    # Already assigned, no change
                                    pass
                                elif unmask & color0:
                                    # Conflict
                                    return (False, color0, color1, edge_state)
                                else:
                                    color1 |= unmask
                                    edge_state[i] = [c0, c1 + 1, 0]
                                    changed = True
                        elif c1 == 4 and un == 1:
                            # The unassigned vertex must be color 0
                            mask = edge_masks[i]
                            unmask = mask & ~(color0 | color1)
                            if unmask:
                                if unmask & color0:
                                    # Already assigned, no change
                                    pass
                                elif unmask & color1:
                                    # Conflict
                                    return (False, color0, color1, edge_state)
                                else:
                                    color0 |= unmask
                                    edge_state[i] = [c0 + 1, c1, 0]
                                    changed = True
                return (True, color0, color1, edge_state)
            
            # Backtracking search
            def backtrack(color0, color1, edge_state):
                # Propagate
                success, color0, color1, edge_state = propagate(edge_state, color0, color1)
                if not success:
                    return False
                
                # Check if all vertices are assigned
                unassigned = ~(color0 | color1) & full_mask
                if unassigned == 0:
                    # All assigned, verify no edge is monochromatic
                    for i in range(len(edge_state)):
                        c0, c1, un = edge_state[i]
                        if c0 == 5 or c1 == 5:
                            return False
                    return True
                
                # Pick the most constrained unassigned vertex
                # Heuristic: pick vertex that appears in the most "critical" edges
                # (edges with 4 of one color and 1 unassigned)
                
                # Simple heuristic: pick first unassigned
                v = (unassigned & -unassigned).bit_length() - 1
                v_bit = 1 << v
                
                # Try color 0
                new_edge_state = [list(s) for s in edge_state]
                for i in range(len(new_edge_state)):
                    if edge_masks[i] & v_bit:
                        if new_edge_state[i][2] > 0:  # vertex was unassigned in this edge
                            new_edge_state[i][0] += 1
                            new_edge_state[i][2] -= 1
                
                if backtrack(color0 | v_bit, color1, new_edge_state):
                    return True
                
                # Try color 1
                new_edge_state = [list(s) for s in edge_state]
                for i in range(len(new_edge_state)):
                    if edge_masks[i] & v_bit:
                        if new_edge_state[i][2] > 0:  # vertex was unassigned in this edge
                            new_edge_state[i][1] += 1
                            new_edge_state[i][2] -= 1
                
                if backtrack(color0, color1 | v_bit, new_edge_state):
                    return True
                
                return False
            
            return backtrack(color0, color1, edge_state)
        
        return not check(0, 0)
    
    # Start with all edges
    best_edges = all_edges[:]
    best_len = len(best_edges)
    
    # Randomized Greedy Minimization with multiple restarts
    max_restarts = 50
    restart = 0
    while time.time() < deadline and restart < max_restarts:
        restart += 1
        
        current_edges = all_edges[:]
        random.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges) and time.time() < deadline:
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if is_not_2_colorable(test_edges):
                current_edges = test_edges
                # Do not increment i
            else:
                i += 1
        
        if len(current_edges) < best_len:
            best_edges = current_edges[:]
            best_len = len(best_edges)
        
        # Early termination if we hit a good bound
        if best_len <= 28:
            break
    
    return {'n': n, 'edges': best_edges}
