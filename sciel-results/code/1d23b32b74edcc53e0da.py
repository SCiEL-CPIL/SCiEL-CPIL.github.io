"""Construct a minimal 5-uniform hypergraph without Property B."""
import time
import random
from itertools import combinations

def _check_2colorable(n, edges):
    """Check if hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False if not (i.e., no proper 2-colouring exists).
    """
    if not edges:
        return True
    
    # Build adjacency: for each vertex, which edges contain it
    # For unit propagation: if an edge has 4 vertices of same colour, the 5th is forced
    
    # Represent colouring as list, -1 = unassigned, 0/1 = colour
    # Edge is violated if all 5 same colour
    # We want to find a colouring where NO edge is monochromatic
    
    # Convert edges to tuples for fast lookup
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # For each edge, track which vertices are assigned and their colours
    # Unit propagation: if 4 vertices in an edge have colour c, the 5th must be 1-c
    
    def propagate(assignment, edge_info):
        """Apply unit propagation. Returns (assignment, edge_info, contradiction)"""
        changed = True
        while changed:
            changed = False
            for idx, (edge, counts) in enumerate(edge_info):
                # counts[c] = number of vertices in edge with colour c
                # unassigned = number unassigned
                unassigned = 5 - counts[0] - counts[1]
                if unassigned == 0:
                    # All assigned; check if monochromatic
                    if counts[0] == 5 or counts[1] == 5:
                        return assignment, edge_info, True  # Contradiction
                    continue
                if counts[0] == 5 - unassigned and unassigned > 0:
                    # All unassigned must be 1 (to avoid all-0)
                    # Wait, this isn't quite right for unit propagation
                    pass
                # Proper unit propagation for "not all 0" and "not all 1":
                # Clause "at least one 1 in edge": if 4 are 0, the 5th must be 1
                # Clause "at least one 0 in edge": if 4 are 1, the 5th must be 0
                if counts[0] == 4 and unassigned == 1:
                    # The unassigned vertex must be 1
                    for v in edge:
                        if assignment[v] == -1:
                            assignment[v] = 1
                            changed = True
                            break
                    # Update edge_info
                    edge_info[idx] = (edge, [4, 1])
                elif counts[1] == 4 and unassigned == 1:
                    # The unassigned vertex must be 0
                    for v in edge:
                        if assignment[v] == -1:
                            assignment[v] = 0
                            changed = True
                            break
                    edge_info[idx] = (edge, [1, 4])
        return assignment, edge_info, False
    
    # Initialize
    assignment = [-1] * n
    # For each edge, track count of 0s and 1s
    edge_info = []
    for e in edges:
        edge_info.append((tuple(e), [0, 0]))
    
    # We'll do backtracking
    # Choose vertex with smallest domain (MRV heuristic)
    
    def backtrack(assignment, edge_info):
        # Check for contradiction via propagation
        assignment, edge_info, contradiction = propagate(assignment, edge_info)
        if contradiction:
            return False
        
        # Find unassigned vertex
        unassigned = [i for i in range(n) if assignment[i] == -1]
        if not unassigned:
            # All assigned; verify no monochromatic edge
            for e, counts in edge_info:
                if counts[0] == 5 or counts[1] == 5:
                    return False
            return True
        
        # MRV: pick vertex involved in most constrained edges
        # For simplicity, pick first unassigned
        v = unassigned[0]
        
        # Try colour 0
        assignment[v] = 0
        # Update edge_info
        new_edge_info = []
        for e, counts in edge_info:
            if v in e:
                new_edge_info.append((e, [counts[0]+1, counts[1]]))
            else:
                new_edge_info.append((e, counts[:]))
        
        if backtrack(assignment, new_edge_info):
            return True
        
        # Try colour 1
        assignment[v] = 1
        new_edge_info = []
        for e, counts in edge_info:
            if v in e:
                new_edge_info.append((e, [counts[0], counts[1]+1]))
            else:
                new_edge_info.append((e, counts[:]))
        
        if backtrack(assignment, new_edge_info):
            return True
        
        assignment[v] = -1
        return False
    
    return backtrack(assignment, edge_info)


def construct():
    """Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # We'll search over different n values and use a greedy+verification approach
    # Target: find a hypergraph with ~51 or fewer edges
    
    best_result = None
    best_edge_count = float('inf')
    
    # Try various n values
    for n in range(10, 15):
        if time.time() > deadline - 30:
            break
        
        # Generate all possible 5-edges
        all_edges = list(combinations(range(n), 5))
        
        # Randomly shuffle and greedily add edges, checking for non-2-colourability
        # Strategy: start with random subset, check if non-2-colourable, then minimize
        
        attempts = 0
        max_attempts = 500
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            # Start with a random set of edges
            # Use a heuristic: prefer edges that are "overlapping" to create conflicts
            random.seed(attempts * 7 + n * 13)
            
            # Try different sizes
            target_size = random.randint(45, 60)
            
            # Greedy construction: add edges one by one
            # Start with some random edges
            edge_set = set()
            remaining = all_edges[:]
            random.shuffle(remaining)
            
            # Add edges, checking periodically if we've achieved non-2-colourability
            for edge in remaining:
                if len(edge_set) >= target_size:
                    break
                edge_set.add(edge)
                
                # Check every few edges
                if len(edge_set) % 5 == 0 and len(edge_set) >= 40:
                    if time.time() > deadline - 20:
                        break
                    edges_list = [list(e) for e in edge_set]
                    if not _check_2colorable(n, edges_list):
                        # We found a non-2-colourable hypergraph!
                        # Now try to minimize by removing edges
                        current_edges = list(edge_set)
                        minimized = _minimize_edges(n, current_edges, deadline)
                        if minimized is not None:
                            count = len(minimized)
                            if count < best_edge_count:
                                best_edge_count = count
                                best_result = {'n': n, 'edges': minimized}
                        break
            
            if time.time() > deadline - 20:
                break
        
        if time.time() > deadline - 30:
            break
    
    # If we didn't find anything, return a known valid construction
    if best_result is None:
        # Fallback: use a denser construction that we know works
        # Complete 5-uniform on 9 vertices (126 edges) - valid but not minimal
        # Better: try a structured approach
        n = 12
        # Use a random dense construction and verify
        random.seed(42)
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Take enough edges to ensure non-2-colourability
        # With n=12, C(12,5)=792 edges. We need enough to block all colourings.
        # Let's try a structured approach: take all edges from a random subset
        edges_list = [list(e) for e in all_edges[:80]]
        
        # Verify and try to minimize
        if not _check_2colorable(n, edges_list):
            minimized = _minimize_edges(n, edges_list, deadline)
            if minimized is not None:
                return {'n': n, 'edges': minimized}
        
        # Just return a valid (if not minimal) construction
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return best_result


def _minimize_edges(n, edges, deadline):
    """Try to minimize the number of edges while preserving non-2-colourability."""
    edges = list(edges)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        # Try removing each edge
        for i in range(len(edges)):
            if time.time() > deadline - 5:
                break
            # Temporarily remove edge i
            candidate = edges[:i] + edges[i+1:]
            if not _check_2colorable(n, candidate):
                # Still non-2-colourable, keep the removal
                edges = candidate
                changed = True
                break  # Restart from beginning
    return edges
