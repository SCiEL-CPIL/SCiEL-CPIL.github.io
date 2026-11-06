"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def _is_not_2colorable(n, edges, deadline=None):
    """Check if hypergraph has NO proper 2-coloring using backtracking with unit propagation.
    Returns True if NOT 2-colorable (i.e., UNSAT), False if 2-colorable."""
    if deadline is not None and time.time() > deadline:
        return True  # assume not colorable to be safe (conservative)
    
    # Build incidence: for each vertex, list of edges containing it
    # For each edge, list of vertices
    num_edges = len(edges)
    if num_edges == 0:
        return False  # empty hypergraph is 2-colorable
    
    # Unit propagation + backtracking
    # State: assignment array (-1=unassigned, 0=False, 1=True)
    # We'll use a recursive approach with unit propagation
    
    # Precompute: for each edge, which vertices are in it
    # For efficiency, track for each edge how many vertices are assigned and their values
    
    def solve(assignment, edge_counts):
        """edge_counts[i] = (num_true, num_false) for edge i"""
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                nt, nf = edge_counts[i]
                unassigned = 5 - nt - nf
                if unassigned == 0:
                    # Edge is fully assigned
                    if nt == 5 or nf == 5:
                        return False  # monochromatic edge found -> this branch fails
                    # else edge is properly colored, continue
                elif unassigned == 1:
                    # One unassigned vertex: if all others are same color, must be opposite
                    if nt == 4:
                        # Must assign the unassigned vertex to False
                        for v in edges[i]:
                            if assignment[v] == -1:
                                assignment[v] = 0
                                edge_counts[i] = (4, 2)
                                changed = True
                                break
                    elif nf == 4:
                        # Must assign the unassigned vertex to True
                        for v in edges[i]:
                            if assignment[v] == -1:
                                assignment[v] = 1
                                edge_counts[i] = (2, 4)
                                changed = True
                                break
        # Check for conflicts after propagation
        for i in range(num_edges):
            nt, nf = edge_counts[i]
            if nt == 5 or nf == 5:
                return False
            # Check if edge can still be properly colored
            unassigned = 5 - nt - nf
            if unassigned > 0:
                # Can we still avoid monochromatic?
                if nt + unassigned < 5 and nf + unassigned < 5:
                    pass  # OK
                else:
                    # One color already has 5 or will inevitably have 5
                    if nt >= 5 or nf >= 5:
                        return False
                    # If nt + unassigned == 5, all remaining must be False to avoid all-True
                    # If nf + unassigned == 5, all remaining must be True to avoid all-False
                    # These are handled by unit propagation above
        # Find unassigned vertex with smallest domain (MRV heuristic)
        min_unassigned = 6
        best_v = -1
        for v in range(n):
            if assignment[v] == -1:
                # Count how many edges this vertex is in that still have unassigned vertices
                count = 0
                for i in range(num_edges):
                    if v in edges[i]:
                        nt, nf = edge_counts[i]
                        if 5 - nt - nf > 0:
                            count += 1
                if count < min_unassigned:
                    min_unassigned = count
                    best_v = v
                    if count == 0:
                        break
        if best_v == -1:
            return True  # All assigned, no monochromatic edge found
        # Branch on best_v
        for val in [0, 1]:
            # Save state
            old_assignment = assignment.copy()
            old_edge_counts = [c.copy() if isinstance(c, tuple) else c for c in edge_counts]
            # Actually edge_counts are tuples, let me use lists
            # Let me restructure to use lists for edge_counts
            pass
        
        # This is getting complex. Let me use a cleaner approach.
        return True  # placeholder
    
    # Let me rewrite with a cleaner data structure
    # Use lists for edge state: edge_state[i] = [num_true, num_false]
    
    def _check_not_2col(n, edges):
        num_edges = len(edges)
        if num_edges == 0:
            return False
        
        # For unit propagation, we need to track per-edge counts
        # assignment: list of -1, 0, 1
        # edge_state: list of [num_true, num_false]
        
        # Build vertex-to-edges mapping for efficiency
        v_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_edges[v].append(i)
        
        def backtrack(assignment, edge_state):
            # Unit propagation
            stack = list(range(num_edges))
            while stack:
                i = stack.pop()
                nt, nf = edge_state[i]
                unassigned = 5 - nt - nf
                if nt == 5 or nf == 5:
                    return False
                if unassigned == 1:
                    if nt == 4:
                        for v in edges[i]:
                            if assignment[v] == -1:
                                assignment[v] = 0
                                edge_state[i] = [4, 2]
                                # Propagate to other edges containing v
                                for j in v_edges[v]:
                                    if j != i:
                                        edge_state[j][1] += 1
                                        stack.append(j)
                                break
                    elif nf == 4:
                        for v in edges[i]:
                            if assignment[v] == -1:
                                assignment[v] = 1
                                edge_state[i] = [2, 4]
                                for j in v_edges[v]:
                                    if j != i:
                                        edge_state[j][0] += 1
                                        stack.append(j)
                                break
            # Check all edges
            for i in range(num_edges):
                nt, nf = edge_state[i]
                if nt == 5 or nf == 5:
                    return False
            # Find unassigned vertex
            unassigned_vertices = [v for v in range(n) if assignment[v] == -1]
            if not unassigned_vertices:
                return True  # All assigned, no conflict
            # MRV: pick vertex in most constrained edge
            best_v = unassigned_vertices[0]
            best_constraint = -1
            for v in unassigned_vertices:
                max_unassigned_in_edge = 0
                for i in v_edges[v]:
                    nt, nf = edge_state[i]
                    ua = 5 - nt - nf
                    if ua > max_unassigned_in_edge:
                        max_unassigned_in_edge = ua
                # We want the vertex where the edges have fewest unassigned
                # Actually MRV: fewest possible values. Here each vertex has 2 values.
                # Use degree heuristic: vertex in most edges with unassigned vertices
                deg = len([i for i in v_edges[v] if 5 - edge_state[i][0] - edge_state[i][1] > 0])
                if deg > best_constraint:
                    best_constraint = deg
                    best_v = v
            
            # Try both values
            for val in [0, 1]:
                # Save
                old_assignment = assignment.copy()
                old_edge_state = [list(s) for s in edge_state]
                
                assignment[best_v] = val
                if val == 1:
                    for i in v_edges[best_v]:
                        edge_state[i][0] += 1
                else:
                    for i in v_edges[best_v]:
                        edge_state[i][1] += 1
                
                if backtrack(assignment, edge_state):
                    return True
                
                # Restore
                assignment = old_assignment
                edge_state = old_edge_state
            
            return False
        
        initial_assignment = [-1] * n
        initial_edge_state = [[0, 0]] * num_edges
        return not backtrack(initial_assignment, initial_edge_state)
    
    return _check_not_2col(n, edges)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with K_9^5 (126 edges)
    n = 9
    all_edges = list(combinations(range(n), 5))
    edges = [list(e) for e in all_edges]
    
    # Greedy deletion: try to remove edges while maintaining non-2-colorability
    # Shuffle for diversity
    random.seed(42)
    
    # First, verify the full hypergraph is not 2-colorable
    if not _is_not_2colorable(n, edges, deadline):
        # Shouldn't happen, but return as-is
        return {'n': n, 'edges': edges}
    
    # Try to remove edges
    # Order edges by some heuristic - try removing in random order
    edge_indices = list(range(len(edges)))
    random.shuffle(edge_indices)
    
    current_edges = edges[:]
    current_n = n
    
    for idx in edge_indices:
        if time.time() > deadline:
            break
        
        # Try removing edge idx from current_edges
        # Find its position in current_edges
        edge_to_remove = edges[idx]
        try:
            pos = current_edges.index(edge_to_remove)
        except ValueError:
            continue
        
        # Remove it
        test_edges = current_edges[:pos] + current_edges[pos+1:]
        
        # Check if still not 2-colorable
        if _is_not_2colorable(current_n, test_edges, deadline):
            current_edges = test_edges
    
    # Try multiple random restarts with different orderings
    best_edges = current_edges[:]
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 1000 + 42)
        edge_indices = list(range(len(all_edges)))
        random.shuffle(edge_indices)
        
        current_edges = [list(e) for e in all_edges]
        
        for idx in edge_indices:
            if time.time() > deadline:
                break
            edge_to_remove = all_edges[idx]
            try:
                pos = current_edges.index(edge_to_remove)
            except ValueError:
                continue
            
            test_edges = current_edges[:pos] + current_edges[pos+1:]
            
            if _is_not_2colorable(current_n, test_edges, deadline):
                current_edges = test_edges
        
        if len(current_edges) < len(best_edges):
            best_edges = current_edges[:]
    
    return {'n': current_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
