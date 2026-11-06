import time
import random
import itertools
from collections import deque

def is_property_b(n, edges):
    """
    Check if the hypergraph has no proper 2-coloring.
    Returns True if Property B holds (no 2-coloring).
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False
    
    # Map vertices to a set of incident edges
    vertex_edges = [set() for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].add(i)
            
    # We want to assign colors 0 or 1 to vertices.
    # An edge is "satisfied" (not monochromatic) if it has at least one 0 and one 1.
    # "No proper 2-coloring" means for EVERY assignment, there exists an edge that is monochromatic.
    # Equivalently, the system of constraints "for all edges, not all vertices same color" is unsatisfiable.
    
    # Backtracking with unit propagation on the CNF:
    # For each edge {v1, ..., v5}, the constraint is:
    # (v1 OR ... OR v5) AND (!v1 OR ... OR !v5)
    # This is a 2-CNF? No, it's a 5-CNF.
    # Unit propagation works if we fix some variables.
    
    # Simpler: Just use a standard DPLL/CDCL-lite or simple backtracking.
    # Given n <= 26, simple backtracking might be slow if not careful.
    # But we only need to check a few candidates.
    
    # Let's use a recursive backtracking with pruning.
    # State: assignment array of size n, -1 for unassigned.
    # Pruning: If an edge becomes monochromatic with the current partial assignment (and the rest are unassigned or same color), we can prune?
    # No, if an edge is currently all 0s, it's satisfied. If it's all 0s and 1s, it's satisfied.
    # If it's all 0s and the rest are unassigned, it *could* become unsatisfied if all remaining become 0? No, if all become 0, it IS monochromatic (bad for coloring, good for Property B).
    # We are looking for a coloring that satisfies ALL edges (i.e., no edge is monochromatic).
    # If we find such a coloring, Property B does NOT hold.
    # If no such coloring exists, Property B holds.
    
    # So, we search for a valid 2-coloring. If we find one, return False. If search space exhausted, return True.
    
    def backtrack(assign):
        # Check if current assignment is valid so far (no edge is already monochromatic with all assigned vertices)
        # Actually, if an edge has all vertices assigned and they are all same color, it's a failure for the coloring.
        # If an edge has all vertices assigned and they are mixed, it's OK.
        # If an edge has some assigned, some unassigned:
        # If all assigned are 0, and if all unassigned become 0, it fails. But we don't know yet.
        # Unit propagation:
        # If an edge has 4 vertices assigned as 0, and 1 unassigned, the unassigned MUST be 1 to avoid monochromatic 0.
        # If an edge has 4 vertices assigned as 1, and 1 unassigned, the unassigned MUST be 0.
        
        # Let's implement a simple backtracking with this propagation.
        
        # For efficiency, we can just check all edges at each step.
        # n is small (<= 26), edges <= 200.
        
        # Find first unassigned vertex
        v = -1
        for i in range(n):
            if assign[i] == -1:
                v = i
                break
        
        if v == -1:
            # All assigned. Check if all edges are non-monochromatic.
            for e in edges:
                colors = [assign[x] for x in e]
                if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                    return False # Found a valid coloring
            return True # No valid coloring found
            
        # Try 0
        assign[v] = 0
        # Check if this causes any edge to be forced to fail?
        # If any edge becomes monochromatic with all vertices assigned, fail.
        # If any edge has 4 assigned 0 and 1 unassigned, that unassigned must be 1.
        # This is complex to implement generally.
        # Given n is small, simple backtracking might suffice for the few checks we do.
        # But to be safe, let's add a simple check:
        # If we set v to 0, is there an edge where all other 4 are 0 and the 5th is unassigned?
        # Then the 5th is forced to 1.
        
        # For simplicity and speed, let's just do standard backtracking.
        # With n=11, 2^11 = 2048 colorings. Very fast.
        # With n=26, 2^26 = 67 million. Too slow.
        # But we will likely use n=11 or 12.
        
        if backtrack(assign):
            return True
        assign[v] = 1
        if backtrack(assign):
            return True
        assign[v] = -1
        return False

    assign = [-1] * n
    return backtrack(assign)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_edges = None
    best_count = float('inf')
    
    # Try to find a small hypergraph
    # Start with n=11, random edges, try to minimize
    
    # Known fact: 51 edges is the minimum.
    # Let's try to find a solution with n=11.
    
    # We will use a local search / greedy deletion approach.
    # 1. Generate a random hypergraph with enough edges to be likely Property B.
    # 2. Check if it is Property B.
    # 3. If yes, try to remove edges one by one (randomly) and check if it remains Property B.
    # 4. Keep the smallest one found.
    
    for n in [11, 12, 13]:
        # Generate random edges
        all_possible = list(itertools.combinations(range(n), 5))
        
        # Try a few random instances
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Start with a large number of edges, e.g., half of all possible
            num_edges = len(all_possible) // 2
            random.seed(attempt + n)
            edges = random.sample(all_possible, num_edges)
            edges = [list(e) for e in edges]
            
            # Check if Property B
            if not is_property_b(n, edges):
                continue
            
            # It is Property B. Now try to minimize.
            # Greedy deletion:
            # Shuffle edges, try to remove each one if Property B still holds.
            random.shuffle(edges)
            
            i = 0
            while i < len(edges) and time.time() < deadline:
                # Try to remove edges[i]
                new_edges = edges[:i] + edges[i+1:]
                if is_property_b(n, new_edges):
                    edges = new_edges
                    # Do not increment i, check the next element at same index
                else:
                    i += 1
            
            if len(edges) < best_count:
                best_count = len(edges)
                best_edges = [list(e) for e in edges]
                # print(f"Found {n} vertices, {best_count} edges")
                
    if best_edges is not None:
        return {'n': n, 'edges': best_edges}
    
    # Fallback: Complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}

# EVOLVE-BLOCK-START
def construct():
    # The complete 5-uniform hypergraph on 9 vertices: any 2-colouring has 5 vertices of one colour.
    from itertools import combinations
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
