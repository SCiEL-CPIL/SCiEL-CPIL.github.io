"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
from itertools import combinations

def _is_2_colorable(n, edges, deadline):
    """Check if hypergraph is 2-colorable via backtracking with unit propagation."""
    if time.time() > deadline:
        return True  # assume colorable to be safe (we'll verify separately if needed)
    
    # Precompute: for each edge, the set of vertices
    edge_sets = [frozenset(e) for e in edges]
    edge_idx = list(range(len(edges)))
    
    # For unit propagation: for each vertex, which edges contain it
    vert_to_edges = {v: [] for v in range(n)}
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
    
    # State: color[v] = -1 (unassigned), 0, or 1
    color = [-1] * n
    # Active edges: those not yet satisfied (not monochromatic)
    # An edge is satisfied if all its vertices have the same color
    # We need to find a coloring where NO edge is monochromatic
    
    # Use iterative deepening / backtracking with unit propagation
    # Unit propagation: if an edge has 4 vertices of color 0 and 1 unassigned, 
    # the unassigned must be 1. If 4 are 1, unassigned must be 0.
    # If an edge has all 5 same color -> contradiction.
    
    def propagate(color, active_edges):
        """Apply unit propagation. Returns (success, new_color, new_active_edges) or (False, None, None)."""
        changed = True
        new_color = color[:]
        new_active = set(active_edges)
        
        while changed:
            changed = False
            for i in list(new_active):
                e = edge_sets[i]
                counts = [0, 0]
                unassigned = None
                for v in e:
                    if new_color[v] == -1:
                        if unassigned is not None:
                            # More than one unassigned, can't propagate
                            break
                        unassigned = v
                    else:
                        counts[new_color[v]] += 1
                else:
                    if unassigned is not None:
                        if counts[0] == 4:
                            # Must be color 1
                            new_color[unassigned] = 1
                            changed = True
                        elif counts[1] == 4:
                            # Must be color 0
                            new_color[unassigned] = 0
                            changed = True
                    else:
                        # All assigned
                        if counts[0] == 5 or counts[1] == 5:
                            return (False, None, None)
                        else:
                            # Edge is satisfied (has both colors), remove from active
                            new_active.discard(i)
                            changed = True
        
        return (True, new_color, new_active)
    
    # Find the first unassigned vertex with fewest choices
    def select_vertex(color, active_edges):
        best_v = -1
        best_score = float('inf')
        for v in range(n):
            if color[v] == -1:
                # Count how many active edges this vertex is in
                score = len(vert_to_edges[v])  # approximate
                if score < best_score:
                    best_score = score
                    best_v = v
        return best_v
    
    # Backtracking
    def backtrack(color, active_edges):
        if time.time() > deadline:
            return None  # timeout, assume colorable
        
        # Check if all edges are satisfied
        if not active_edges:
            return color[:]
        
        # Unit propagation
        success, new_color, new_active = propagate(color, active_edges)
        if not success:
            return None
        if new_color != color or new_active != active_edges:
            return backtrack(new_color, new_active)
        
        # Select vertex
        v = select_vertex(color, active_edges)
        if v == -1:
            # All vertices assigned but some edges still active -> contradiction
            return None
        
        # Try both colors
        for c in [0, 1]:
            trial_color = color[:]
            trial_color[v] = c
            result = backtrack(trial_color, active_edges)
            if result is not None:
                return result
        return None
    
    initial_active = set(range(len(edges)))
    result = backtrack(color, initial_active)
    return result is not None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Best known: 51 edges construction for m(5)
    # We'll use a known construction or search for a small one
    
    # Known result: m(5) = 51 (or close to it)
    # Let's try to construct a 51-edge 5-uniform hypergraph with no 2-coloring
    
    # One approach: use the known construction from literature
    # Another: systematic search
    
    # For now, use a well-known small construction
    # The complete 5-uniform hypergraph on 9 vertices has C(9,5) = 126 edges
    # We need something much smaller
    
    # Let's try a systematic approach: start with a candidate and verify
    # Known: there exists a 5-uniform hypergraph on ~15 vertices with ~51 edges that is not 2-colorable
    
    # Use a greedy/random search to find a small non-2-colorable hypergraph
    # Start with a base and add edges until it becomes non-2-colorable
    
    best_result = None
    best_edges = None
    
    # Try several random constructions
    rng = random.Random(42)
    
    for n in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline:
            break
        
        # Generate random 5-uniform hypergraphs and check
        # Start with fewer edges and add until non-2-colorable
        
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Generate a random set of edges
            all_possible = list(combinations(range(n), 5))
            rng.shuffle(all_possible)
            
            edges = []
            # Start with a small number and grow
            for num_edges in range(20, 80, 2):
                if time.time() > deadline:
                    break
                
                if len(edges) < num_edges:
                    edges = all_possible[:num_edges]
                
                # Check if 2-colorable
                if not _is_2_colorable(n, edges, deadline):
                    # Found a non-2-colorable hypergraph
                    if best_result is None or len(edges) < len(best_result['edges']):
                        best_result = {'n': n, 'edges': [list(e) for e in edges]}
                        best_edges = edges
                    break
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete 5-uniform on 9 vertices (126 edges)
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
