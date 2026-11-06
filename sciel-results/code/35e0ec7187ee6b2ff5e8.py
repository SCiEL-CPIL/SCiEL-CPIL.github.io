import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph (n, edges) has a proper 2-coloring.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Pre-process edges
    edge_tuples = [tuple(e) for e in edges]
    
    # For speed, we can map edges to indices and track unassigned vertices
    # But for n <= 12, simple backtracking with UP is enough.
    
    # We will use a recursive function with a trail.
    # assignment: -1 for unassigned, 0 or 1 for assigned.
    assignment = [-1] * n
    
    # Count how many vertices in each edge are unassigned
    unassigned_count = [5] * len(edge_tuples)
    
    # For each vertex, list of edges it belongs to
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_tuples):
        for v in e:
            vertex_edges[v].append(i)
            
    # Unit Propagation
    def propagate():
        # Returns False if contradiction, True if consistent
        # We use a stack for unit propagation
        stack = []
        # Check all edges for unit constraints
        # An edge is unit if it has 1 unassigned vertex and the other 4 are same color.
        # Or if it has 0 unassigned and is monochromatic (contradiction).
        
        # To be efficient, we only check edges affected by changes.
        # But for simplicity and correctness in a small n, we can scan all edges 
        # or maintain a queue.
        # Given n is small, scanning all edges in each step of backtracking is O(E).
        # Backtracking depth is N. Total O(N*E).
        
        # Let's implement a simple propagation that returns the list of forced assignments
        # or None if contradiction.
        
        # We need to be careful: propagate should be called after each assignment.
        # We will implement a function that takes the current assignment and performs UP.
        # Actually, it's easier to integrate UP into the backtrack step.
        
        pass # Placeholder for logic below

    def backtrack(idx):
        # Find the next unassigned vertex
        # Heuristic: Pick vertex with highest degree or involved in most constrained edges?
        # Simple: Next index.
        
        # Check for contradictions in fully assigned edges
        # This is inefficient. We should check as we go.
        
        # Let's use a standard DPLL approach.
        
        # 1. Unit Propagation
        #    Scan all edges.
        #    If an edge has 0 unassigned and is mono -> Contradiction (return False)
        #    If an edge has 1 unassigned (v) and other 4 are c -> assignment[v] = 1-c
        #    If this causes conflict (v already assigned to c) -> Contradiction
        #    Repeat until no more units.
        
        # 2. If all assigned -> Success (return True)
        
        # 3. Pick a variable (vertex) to branch on.
        #    Branch on 0 and 1.
        
        # This is complex to implement correctly with state restoration in Python quickly.
        # Given the small N (<=12), a simpler optimized backtracking might suffice.
        
        # Let's try a different optimization:
        # Bitmask representation?
        # N=12, 2^12 = 4096. We can precompute all 2-colorings?
        # No, we need to check if ANY coloring works.
        
        # Let's stick to the recursive checker but optimize the consistency check.
        
        if idx == n:
            return True
            
        # Try 0
        assignment[idx] = 0
        if check_consistency(idx):
            if backtrack(idx + 1):
                return True
                
        # Try 1
        assignment[idx] = 1
        if check_consistency(idx):
            if backtrack(idx + 1):
                return True
                
        assignment[idx] = -1
        return False

    def check_consistency(idx):
        # Check edges where idx is the max vertex? No, any edge where all are assigned.
        # Optimization: Only check edges that include idx?
        # If an edge doesn't include idx, and was consistent before, it remains consistent.
        # So we only need to check edges containing idx that are now fully assigned.
        
        for e_idx in vertex_edges[idx]:
            e = edge_tuples[e_idx]
            # Check if all vertices in e are assigned
            # Since we assign sequentially 0..idx, and e contains idx,
            # if all other vertices in e are < idx and assigned, then e is fully assigned.
            # Actually, if e contains a vertex > idx, it's not fully assigned yet.
            
            # So, check if all v in e are assigned.
            is_full = True
            colors = []
            for v in e:
                if assignment[v] == -1:
                    is_full = False
                    break
                colors.append(assignment[v])
                
            if is_full:
                if len(set(colors)) == 1:
                    return False
        return True

    return backtrack(0)

def construct():
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Try n from 9 to 11
    for n in range(9, 12):
        # Generate complete hypergraph
        all_edges = [tuple(e) for e in combinations(range(n), 5)]
        
        # Shuffle
        random.seed(n) # Different seed for each n
        edges_to_try = all_edges.copy()
        random.shuffle(edges_to_try)
        
        current_edges = set(all_edges)
        
        # Greedy Pruning
        for edge in edges_to_try:
            if edge not in current_edges:
                continue
            
            # Temporarily remove
            current_edges.remove(edge)
            
            # Check
            if not is_2_colorable(n, list(current_edges)):
                # Keep removed
                pass
            else:
                # Add back
                current_edges.add(edge)
                
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_n = n
            best_edges = list(current_edges)
            
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
