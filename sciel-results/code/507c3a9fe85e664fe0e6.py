"""Rewritten program: SAT-based greedy edge selection for m(5) hypergraph."""
import random
from itertools import combinations

def _backtrack_2colorable(n, edges, timeout_s=10.0):
    """
    Check if the 5-uniform hypergraph on n vertices is 2-colorable.
    Returns True if a proper 2-coloring exists (i.e., not all edges are monochromatic).
    Uses backtracking with unit propagation (forced assignments from edges with 4 same-color vertices).
    """
    import time
    start_time = time.time()
    
    # Preprocess edges for fast lookup
    # For each edge, store as tuple of sorted vertices
    edge_tuples = [tuple(sorted(e)) for e in edges]
    
    # Build adjacency: for each vertex, list of edge indices containing it
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_tuples):
        for v in e:
            vertex_to_edges[v].append(i)
    
    # For unit propagation: if 4 vertices in an edge have same color, the 5th is forced
    # We'll track assignments: -1 unassigned, 0 or 1 color
    
    def check_satisfiable(assignments):
        """
        assignments: list of -1, 0, 1 for each vertex
        Returns: True if satisfiable (can extend to full proper coloring), False if contradiction
        """
        # Unit propagation: repeatedly find edges with 4 same-colored vertices
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_tuples):
                colors = [assignments[v] for v in e]
                # Count 0s and 1s
                count_0 = colors.count(0)
                count_1 = colors.count(1)
                # If 4 are 0, the remaining must be 1 (to avoid all-0)
                if count_0 == 4:
                    for v in e:
                        if assignments[v] == -1:
                            assignments[v] = 1
                            changed = True
                            break
                elif count_1 == 4:
                    for v in e:
                        if assignments[v] == -1:
                            assignments[v] = 0
                            changed = True
                            break
                # Contradiction: all 5 same color
                if count_0 == 5 or count_1 == 5:
                    return False
        # Check for any edge that is fully assigned and monochromatic
        for i, e in enumerate(edge_tuples):
            colors = [assignments[v] for v in e]
            if -1 not in colors:
                if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                    return False
        return True
    
    def backtrack(assignments, unassigned_count):
        if time.time() - start_time > timeout_s:
            return True  # Assume satisfiable on timeout to be safe for "not 2-colorable" check
        
        if unassigned_count == 0:
            return check_satisfiable(assignments)
        
        # Find most constrained unassigned vertex
        best_v = -1
        best_score = -1
        for v in range(n):
            if assignments[v] != -1:
                continue
            # Count how many edges this vertex is in that have partial assignments
            score = 0
            for i in vertex_to_edges[v]:
                e = edge_tuples[i]
                colors = [assignments[u] for u in e]
                if -1 in colors:
                    score += 1
                else:
                    # Fully assigned edge
                    if all(c == 0 for c in colors) or all(c == 1 for c in colors):
                        return False
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v == -1:
            return check_satisfiable(assignments)
        
        # Try assigning 0 first, then 1
        for color in [0, 1]:
            assignments[best_v] = color
            if check_satisfiable(assignments):
                if backtrack(assignments, unassigned_count - 1):
                    return True
            assignments[best_v] = -1
        
        return False
    
    assignments = [-1] * n
    return backtrack(assignments, n)

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Uses greedy SAT-based heuristic: start with random vertex set, add edges that
    maximally reduce the solution space until unsatisfiable.
    """
    import time
    start_time = time.time()
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try different n values
    for n in range(9, 15):
        if time.time() - start_time > 100:
            break
        
        # Try multiple random seeds for diversity
        for seed in range(20):
            if time.time() - start_time > 105:
                break
            
            rng = random.Random(seed)
            edges = []
            edge_set = set()
            
            # Start with a few random edges to break symmetry
            all_possible = list(combinations(range(n), 5))
            rng.shuffle(all_possible)
            
            # Add initial random edges
            initial_count = min(10, len(all_possible))
            for e in all_possible[:initial_count]:
                edges.append(list(e))
                edge_set.add(tuple(e))
            
            # Greedy: repeatedly add the edge that most reduces solution space
            max_iter = 200
            for _ in range(max_iter):
                if time.time() - start_time > 108:
                    break
                
                # Check if current hypergraph is already non-2-colorable
                if not _backtrack_2colorable(n, edges, timeout_s=2.0):
                    if len(edges) < best_count:
                        best_count = len(edges)
                        best_edges = [list(e) for e in edges]
                        best_n = n
                    break
                
                # Find the best edge to add
                best_edge = None
                best_score = -1
                
                # Sample a subset of candidate edges for efficiency
                candidates = [e for e in all_possible if tuple(e) not in edge_set]
                if not candidates:
                    break
                
                # Limit candidates for speed
                if len(candidates) > 500:
                    rng.shuffle(candidates)
                    candidates = candidates[:500]
                
                for e in candidates:
                    test_edges = edges + [list(e)]
                    # Quick check: does adding this edge make it non-2-colorable?
                    if not _backtrack_2colorable(n, test_edges, timeout_s=0.5):
                        best_edge = e
                        best_score = float('inf')
                        break
                    
                    # Otherwise, score by how much it reduces solution space
                    # Approximate: count number of satisfying assignments before/after
                    # This is expensive, so use a simpler heuristic:
                    # Prefer edges that create more unit propagation chains
                    score = 0
                    for v in e:
                        for i, existing_e in enumerate(edges):
                            common = len(set(e) & set(existing_e))
                            if common >= 3:
                                score += common
                    if score > best_score:
                        best_score = score
                        best_edge = e
                
                if best_edge is None:
                    break
                
                edges.append(list(best_edge))
                edge_set.add(tuple(best_edge))
        
        # After trying all seeds for this n, check if we found something
        if best_edges is not None and best_n == n:
            break
    
    # Fallback to known construction if greedy didn't find anything good
    if best_edges is None or best_count > 51:
        # Use the complete 5-uniform hypergraph on 9 vertices as fallback
        # But try to find a smaller one first
        # Known: there exist constructions with fewer edges
        # For now, use the seed program's approach as fallback
        from itertools import combinations
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': best_n, 'edges': best_edges}
