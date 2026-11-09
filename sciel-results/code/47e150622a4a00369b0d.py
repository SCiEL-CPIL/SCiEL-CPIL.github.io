<<<<<<< SEARCH
import time
import random
import itertools

def construct():
    n = 11
    vertices = list(range(n))
    all_edges = list(itertools.combinations(vertices, 5))
    num_edges = len(all_edges)
    num_colorings = 1 << n
    
    # Precompute masks for each edge
    # mask[i] is a bitset where bit j is 1 if coloring j makes edge i monochromatic
    masks = [0] * num_edges
    
    for i, e in enumerate(all_edges):
        # Calculate the bitmask for the vertices in this edge
        edge_bitmask = 0
        for v in e:
            edge_bitmask |= (1 << v)
        
        m = 0
        for c in range(num_colorings):
            val = c & edge_bitmask
            if val == 0 or val == edge_bitmask:
                m |= (1 << c)
        masks[i] = m

    best_len = float('inf')
    best_edges = []
    
    # Deadline guard: 100 seconds to be safe within 120s limit
    deadline = time.time() + 100.0
    
    while time.time() < deadline:
        active = (1 << num_colorings) - 1
        selected_indices = []
        
        while active != 0:
            # Find edges that cover at least one active coloring
            candidates = []
            for i in range(num_edges):
                # If the edge mask has no overlap with active, skip
                if masks[i] & active:
                    cnt = (masks[i] & active).bit_count()
                    candidates.append((cnt, i))
            
            if not candidates:
                # This should not happen for n=11, but break to avoid infinite loop
                break
            
            # Sort by count descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Pick randomly from top 10 to diversify search
            k = min(10, len(candidates))
            _, idx = random.choice(candidates[:k])
            
            selected_indices.append(idx)
            active &= ~masks[idx]
            
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_edges = [all_edges[i] for i in selected_indices]
            
            # If we found a very good solution, we can stop or continue to see if we can do better
            if best_len <= 51:
                break

    return {'n': n, 'edges': [list(e) for e in best_edges]}
=======
import time
import random
import itertools

def is_non_2_colorable(n, edges):
    """Check if hypergraph has no proper 2-coloring using backtracking with unit propagation."""
    if not edges:
        return False
    
    # Build adjacency: for each vertex, which edges contain it
    vertex_to_edges = {v: [] for v in range(n)}
    for idx, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(idx)
    
    # State: assignment of vertices to colors (-1=unassigned, 0/1=assigned)
    # Use recursive backtracking with unit propagation
    
    def backtrack(assign, edge_satisfied):
        # edge_satisfied: set of edge indices that are already monochromatic (violation)
        # Actually, we track edges that are NOT yet monochromatic
        # An edge is violated if all its vertices are assigned the same color
        
        # Unit propagation: if an edge has all but one vertex assigned the same color,
        # the last vertex must be assigned the opposite color (or if all assigned same, violation)
        
        # Check for violations
        for idx in range(len(edges)):
            e = edges[idx]
            colors = [assign[v] for v in e]
            if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                return False  # violation
            # If all but one are assigned and same color, force the last one
            unassigned = [v for v in e if assign[v] == -1]
            assigned = [assign[v] for v in e if assign[v] != -1]
            if len(unassigned) == 1 and len(assigned) == 4:
                if len(set(assigned)) == 1:
                    forced_color = 1 - assigned[0]
                    if assign[unassigned[0]] != -1 and assign[unassigned[0]] != forced_color:
                        return False
                    assign[unassigned[0]] = forced_color
        
        # Find unassigned vertex with smallest domain (heuristic)
        unassigned_vertices = [v for v in range(n) if assign[v] == -1]
        if not unassigned_vertices:
            # All assigned, check all edges
            for idx in range(len(edges)):
                e = edges[idx]
                colors = [assign[v] for v in e]
                if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                    return False
            return True
        
        # Pick vertex with most constraints
        v = min(unassigned_vertices, key=lambda x: len(vertex_to_edges[x]))
        
        for color in [0, 1]:
            assign[v] = color
            if backtrack(assign, None):
                return True
            assign[v] = -1
        
        return False
    
    assign = [-1] * n
    return not backtrack(assign, None)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 11
    all_edges = list(itertools.combinations(range(n), 5))
    
    # Start with a known large non-2-colorable hypergraph: all edges
    # Then try to remove edges while maintaining non-2-colorability
    
    # Better: start with a random subset and grow, or start full and shrink
    # Let's start with all edges (definitely non-2-colorable) and greedily remove
    
    # First, verify that all edges is non-2-colorable
    # For 11 vertices, any 2-coloring has at least 6 of one color, so C(6,5)=6 monochromatic edges
    # So yes, K_11^5 is non-2-colorable
    
    current_edges = set(all_edges)
    
    # Greedy removal: try to remove each edge, check if still non-2-colorable
    # This is expensive, so we'll do it with a time limit
    
    random.seed(42)
    edge_list = list(all_edges)
    random.shuffle(edge_list)
    
    # Try to remove edges in random order
    removed = set()
    for i, e in enumerate(edge_list):
        if time.time() > deadline - 10:
            break
        if e in removed:
            continue
        
        # Try removing this edge
        trial_edges = [ed for ed in current_edges if ed != e]
        
        # Quick check: if number of edges is too small, it might be 2-colorable
        # But we need to actually check
        
        if is_non_2_colorable(n, trial_edges):
            current_edges = set(trial_edges)
            removed.add(e)
    
    # Convert to list
    final_edges = list(current_edges)
    
    # If we didn't get a good result, try a different approach:
    # Use a constructive method based on specific patterns
    
    if len(final_edges) > 51:
        # Try a specific known construction
        # Use edges based on arithmetic progressions mod 11
        # or a fixed set that is known to be non-2-colorable
        
        # For now, return what we have
        pass
    
    return {'n': n, 'edges': [list(e) for e in final_edges]}
>>>>>>> REPLACE

Wait, the backtracking checker is too slow for 11 vertices with many edges. Let me use a faster SAT-like approach or the bitmask method but only for verification of smaller candidates.

Let me reconsider. The bitmask approach is actually fast for n=11 since there are only 2048 colorings. The issue is the search strategy. Let me use a much better local search: start from a random non-2-colorable set, then do add/remove local search to minimize edges.

