"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

DEADLINE_SECONDS = 115.0


def _unit_propagate(n, edges, color):
    """
    color: list of length n with values in {-1, 0, 1}
    0 = uncolored, -1 = color A, 1 = color B
    Returns (ok, new_color, forced_count)
    """
    # We maintain a queue of edges to check.
    # For each edge, count how many are uncolored and what colors are present.
    # If an edge has all but one vertex colored and those are all the same,
    # the last one must be the other color.
    changed = True
    queue = list(range(len(edges)))
    while queue and changed:
        changed = False
        # Process from end for efficiency
        i = len(queue) - 1
        while i >= 0:
            idx = queue[i]
            edge = edges[idx]
            # Count colors
            counts = [0, 0]  # A, B
            uncolored_pos = -1
            uncolored_count = 0
            for pos, v in enumerate(edge):
                c = color[v]
                if c == 0:
                    uncolored_count += 1
                    if uncolored_count > 1:
                        break
                    uncolored_pos = pos
                elif c == -1:
                    counts[0] += 1
                else:
                    counts[1] += 1
            
            if uncolored_count == 1:
                if counts[0] == 4:
                    # Must be B
                    v = edge[uncolored_pos]
                    if color[v] == 0:
                        color[v] = 1
                        changed = True
                    # If already 1, fine. If -1, conflict.
                    if color[v] == -1:
                        return False, color, 0
                elif counts[1] == 4:
                    # Must be A
                    v = edge[uncolored_pos]
                    if color[v] == 0:
                        color[v] = -1
                        changed = True
                    if color[v] == 1:
                        return False, color, 0
                # If counts[0]+counts[1] < 4, no constraint yet (e.g. 3 of one, 1 of other, 1 uncolored -> uncolored can be either? No, if 3 A, 1 B, 1 uncolored, uncolored can be A or B? If A -> 4A,1B valid. If B -> 3A,2B valid. So no force.)
            # If uncolored_count == 0:
            # If all same color -> conflict
            if uncolored_count == 0:
                if counts[0] == 5 or counts[1] == 5:
                    return False, color, 0
            
            if changed:
                # This edge might affect others, but we'll just re-scan all? 
                # For simplicity and correctness, if any change, reset queue to all edges.
                # This is O(E) per change, which is fine for small E.
                queue = list(range(len(edges)))
                break
            i -= 1
    return True, color, 0


def _is_2_colorable(n, edges, time_limit=0.1):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Symmetry breaking: vertex 0 is color A (-1)
    color = [0] * n
    color[0] = -1
    
    # Try to propagate initially
    ok, color, _ = _unit_propagate(n, edges, color)
    if not ok:
        return False
    
    # If all colored, check if valid (no monochromatic edge)
    # Actually _unit_propagate returns False if conflict.
    # If all colored and no conflict, it's a valid coloring?
    # Wait, if all colored and no conflict, it means no edge is monochromatic.
    # So it is 2-colorable.
    if 0 not in color:
        return True

    start_time = time.time()
    
    # Find uncolored vertex with highest degree (or just first)
    # For speed, just pick first uncolored.
    
    # Recursive backtracking
    def backtrack(color):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        
        # Unit propagate
        ok, color, _ = _unit_propagate(n, edges, color)
        if not ok:
            return False
        
        if 0 not in color:
            # All colored. Check for monochromatic edges.
            # _unit_propagate should have caught monochromatic edges if they formed.
            # But let's be safe.
            for e in edges:
                c0 = color[e[0]]
                mono = True
                for v in e[1:]:
                    if color[v] != c0:
                        mono = False
                        break
                if mono:
                    return False
            return True
        
        # Pick a vertex to branch on
        # Heuristic: vertex in most "tight" edges?
        # Simple: first uncolored vertex.
        v = color.index(0)
        
        # Try color A
        color[v] = -1
        if backtrack(color):
            return True
        color[v] = 0
        
        # Try color B
        color[v] = 1
        if backtrack(color):
            return True
        color[v] = 0
        
        return False

    try:
        return backtrack(color)
    except TimeoutError:
        # If timeout, we don't know. 
        # For the purpose of finding non-2-colorable graphs, we only accept if we prove it's NOT 2-colorable.
        # If we timeout, we assume it might be 2-colorable (conservative for "finding non-2-colorable").
        # So return True (2-colorable) to reject the candidate.
        return True


def construct():
    start_time = time.time()
    best_edges = None
    best_n = 9
    # Start with the known 51-edge construction if we have it? 
    # We don't have the explicit 51-edge list hardcoded.
    # We will search for one.
    # Random search for small n.
    
    # Known lower bound: 29 edges might be non-2-colorable? 
    # Actually, "at most 28 edges is 2-colourable" means 29+ might be non-2-colorable.
    # Best known is 51.
    # Let's try to find a non-2-colorable graph with < 51 edges.
    
    # Strategy:
    # 1. Try n=9, 10, ...
    # 2. Randomly generate edges.
    # 3. Check if non-2-colorable.
    # 4. Keep best.
    
    # To be efficient, we should bias edge generation.
    # A random 5-uniform hypergraph on n vertices needs many edges to be non-2-colorable.
    # Maybe start with a base structure?
    
    # Let's try a greedy approach:
    # Start with no edges.
    # Add edges one by one.
    # Check if non-2-colorable after each addition? Too slow.
    
    # Better: Generate random sets of edges of size k.
    # Check if non-2-colorable.
    # If yes, try to remove edges (local search) to minimize count.
    
    # Given time limit 115s, we can do a few thousand checks.
    
    # Let's iterate over n from 9 to 15.
    # For each n, try random graphs with increasing edge counts.
    # Start with low edge counts (e.g. 30) and go up.
    # If we find one, try to minimize.
    
    # Actually, the seed program returns 126 edges (complete 9-vertex).
    # We want < 51.
    
    # Let's use a random restart local search.
    # 1. Generate random hypergraph H with n vertices and E edges.
    # 2. If H is non-2-colorable, try to remove edges one by one.
    #    For each edge, remove it, check if still non-2-colorable.
    #    If yes, keep removed.
    # 3. Repeat until no more edges can be removed.
    # 4. Record size.
    
    # To find a good starting point, we need H to be non-2-colorable.
    # Random hypergraphs become non-2-colorable at a threshold.
    # For 5-uniform, threshold is around ... ?
    
    # Let's just try random graphs with E in range [30, 60].
    
    rng = random.Random(42)
    
    for n in range(9, 16):
        # Max edges is C(n, 5)
        max_edges = len(list(itertools.combinations(range(n), 5)))
        if max_edges < 29:
            continue
            
        # Try different edge counts
        # We want to find the minimum.
        # Let's scan edge counts from 29 up to 50.
        # For each count, do a few random trials.
        
        for e in range(29, 51):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
                
            # Try a few random graphs with e edges
            trials = 10
            for t in range(trials):
                if time.time() - start_time > DEADLINE_SECONDS - 5:
                    break
                    
                # Generate random e edges
                all_edges = list(itertools.combinations(range(n), 5))
                rng.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:e]]
                
                # Check if non-2-colorable
                # Use a short time limit for the check
                colorable = _is_2_colorable(n, edges, time_limit=0.5)
                
                if not colorable:
                    # We found a non-2-colorable graph with e edges!
                    # Try to minimize
                    current_edges = edges
                    current_n = n
                    
                    # Local search: remove edges
                    improved = True
                    while improved:
                        improved = False
                        for i in range(len(current_edges)):
                            if time.time() - start_time > DEADLINE_SECONDS - 2:
                                break
                            
                            # Try removing edge i
                            test_edges = current_edges[:i] + current_edges[i+1:]
                            # Check if still non-2-colorable
                            # Use very short time limit for this inner loop
                            colorable = _is_2_colorable(current_n, test_edges, time_limit=0.2)
                            
                            if not colorable:
                                # Can remove
                                current_edges = test_edges
                                improved = True
                                break # Restart loop with new list
                        if time.time() - start_time > DEADLINE_SECONDS - 2:
                            break
                    
                    # Update best
                    if best_edges is None or len(current_edges) < len(best_edges):
                        best_edges = current_edges
                        best_n = current_n
                        
                    # If we found a very small one, we can stop
                    if len(best_edges) <= 29:
                        return {'n': best_n, 'edges': best_edges}
                        
    # If we found something better than the seed (126), return it
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: return the seed construction (complete 9-vertex)
    # This is valid, though not minimal.
    return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
