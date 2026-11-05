"""Few edges in 5-uniform hypergraphs without Property B (m(5))"""
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills after 150s, keep safety margin
_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _check_time():
    return _elapsed() > _DEADLINE

def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph has a proper 2-coloring
    (no edge is monochromatic) using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    # Precompute edge membership
    # For unit propagation: if an edge has 4 vertices of same color, the 5th must be opposite
    # If an edge has all 5 same color, contradiction
    
    # Represent colors: -1 = unassigned, 0 = color A, 1 = color B
    colors = [-1] * n
    assignments = []
    
    # Build index: for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # For unit propagation, we need to track for each edge how many of each color
    # edge_color_count[i] = [count_color0, count_color1, count_unassigned]
    
    def propagate(start_idx=0):
        """Apply unit propagation. Returns False on contradiction, True otherwise."""
        # Use a queue of newly assigned vertices
        queue = list(range(start_idx, n)) if start_idx < n else []
        # Actually, we need to be more careful. Let's use a worklist approach.
        
        # Simpler: iterate until fixpoint
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                # Contradiction: all 5 same color
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit propagation: 4 of one color forces the 5th
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 1
                        changed = True
                    elif colors[v] == 0:
                        return False
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == -1:
                        colors[v] = 0
                        changed = True
                    elif colors[v] == 1:
                        return False
        return True
    
    def backtrack(idx):
        """Try to assign vertex idx and beyond. Returns True if satisfiable."""
        if idx >= n:
            # Check all edges are not monochromatic
            for e in edges:
                c0 = sum(1 for v in e if colors[v] == 0)
                c1 = sum(1 for v in e if colors[v] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True
        
        # Try both colors
        for c in (0, 1):
            colors[idx] = c
            # Save state for undo
            saved = colors[idx:]
            if propagate(idx + 1):
                if backtrack(idx + 1):
                    return True
            # Restore
            colors[idx:] = saved
            if _check_time():
                return True  # Assume satisfiable to avoid timeout issues
        
        return False
    
    return backtrack(0)

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-coloring, minimizing edges.
    Returns {'n': n, 'edges': [[v1,...,v5], ...]}
    """
    best = None
    best_edges = None
    
    # We know the complete 5-uniform hypergraph on 9 vertices works (126 edges)
    # But we want fewer. Let's try to find smaller constructions.
    
    # Strategy: Try various n and use random search / greedy to find non-2-colorable
    # hypergraphs with few edges.
    
    # Known: m(5) is between 29 and 51.
    # Let's try to find constructions with as few edges as possible.
    
    # Approach: Start with random edge sets and check if they're non-2-colorable.
    # For efficiency, use a fast 2-colorability check.
    
    # For small n, we can enumerate more carefully.
    
    # Let's try n from 9 to 16, with increasing edge counts
    # For each (n, m), generate random m-edge 5-uniform hypergraphs and check
    
    # First, try to find a good construction
    # Use a greedy approach: add edges that make the hypergraph harder to 2-color
    
    # Actually, let's use a smarter approach:
    # 1. Start with a known non-2-colorable hypergraph
    # 2. Try to remove edges while maintaining non-2-colorability
    
    # The complete 5-uniform on 9 vertices is non-2-colorable (126 edges)
    # Let's try to reduce it.
    
    # Alternative: use the known construction with 51 edges if we can't do better
    # But let's try to find something with fewer edges.
    
    # For n=9, we have C(9,5)=126 possible edges
    # For n=10, C(10,5)=252
    # etc.
    
    # Let's try a local search approach:
    # 1. Generate a random 5-uniform hypergraph with m edges
    # 2. Check if it's non-2-colorable
    # 3. If not, add edges strategically
    # 4. If yes, try to remove edges
    
    # For speed, let's use a timeout on the 2-colorability check
    
    # Let's try n=9 first since it has the most symmetry
    for n in range(9, 17):
        if _check_time():
            break
        
        # Max edges for this n
        max_edges = min(200, len(list(combinations(range(n), 5))))
        
        # Try different edge counts from low to high
        # Start with a reasonable lower bound
        for m in range(20, max_edges + 1, 2):
            if _check_time():
                break
            
            # Try several random instances
            attempts = 0
            max_attempts = 50 if m < 40 else 20
            
            while attempts < max_attempts and not _check_time():
                attempts += 1
                edges = random.sample(list(combinations(range(n), 5)), m)
                edges = [list(e) for e in edges]
                
                # Check if non-2-colorable
                if not _is_2_colorable(n, edges):
                    if best is None or m < best:
                        best = m
                        best_edges = edges
                        # Try to reduce
                        # For now, just record and continue searching
                        break
    
    # If we found something, try to reduce it further
    if best_edges is not None:
        # Try greedy edge removal
        current_edges = list(best_edges)
        improved = True
        while improved and not _check_time():
            improved = False
            for i in range(len(current_edges)):
                if _check_time():
                    break
                # Try removing edge i
                candidate = current_edges[:i] + current_edges[i+1:]
                if not _is_2_colorable(len(current_edges[0]) and 0 or 9, candidate):
                    # Wait, we need n, not derived from edges
                    pass
                if not _is_2_colorable(n, candidate):
                    current_edges = candidate
                    improved = True
                    break
        
        if current_edges is not None:
            best = len(current_edges)
            best_edges = current_edges
    
    # If we couldn't find anything better, use the complete hypergraph on 9 vertices
    if best_edges is None:
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best = len(best_edges)
    else:
        best = len(best_edges)
    
    return {'n': n if best_edges else 9, 'edges': best_edges}
