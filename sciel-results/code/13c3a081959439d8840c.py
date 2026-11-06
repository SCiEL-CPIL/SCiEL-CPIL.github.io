"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills after 150s; we target finishing well before that.
_DEADLINE = 150.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_remaining():
    return _DEADLINE - 15.0 - _elapsed()  # 15s safety margin

def is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph on n vertices is 2-colorable.
    Uses backtracking with unit propagation (constraint propagation).
    Returns True if a proper 2-coloring exists, False otherwise.
    """
    if not edges:
        return True
    
    # Build adjacency: for each vertex, which edges contain it
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
    
    # For each edge, track how many vertices are unassigned, and the colors assigned
    edge_unassigned = [5] * len(edges)
    edge_color_count = [[0, 0] for _ in range(edges)]  # [count_color0, count_color1]
    
    # Assignment: -1 = unassigned, 0 = color0, 1 = color1
    assignment = [-1] * n
    
    # Unit propagation queue
    import collections
    queue = collections.deque()
    
    def propagate():
        """
        Unit propagation: if an edge has 4 vertices of one color, the 5th must be the other.
        Also detects contradictions (an edge has all 5 of one color).
        Returns (success, list_of_newly_assigned_vertices)
        """
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edges):
                unassigned = edge_unassigned[i]
                if unassigned == 0:
                    continue
                c0 = edge_color_count[i][0]
                c1 = edge_color_count[i][1]
                # If 4 are color0, the remaining must be color1
                if c0 == 4 and unassigned == 1:
                    # Find the unassigned vertex in this edge
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 1
                            edge_color_count[i][1] += 1
                            edge_unassigned[i] -= 1
                            changed = True
                            break
                    # Check for contradiction: now c1=5
                    if edge_color_count[i][1] == 5:
                        return False, []
                elif c1 == 4 and unassigned == 1:
                    for v in e:
                        if assignment[v] == -1:
                            assignment[v] = 0
                            edge_color_count[i][0] += 1
                            edge_unassigned[i] -= 1
                            changed = True
                            break
                    if edge_color_count[i][0] == 5:
                        return False, []
                # Also: if unassigned == 0 and (c0==5 or c1==5), contradiction
                elif unassigned == 0:
                    if c0 == 5 or c1 == 5:
                        return False, []
        return True, []
    
    # Initial propagation
    ok, _ = propagate()
    if not ok:
        return False
    
    # Backtracking search
    # Heuristic: pick the vertex that appears in the most edges with fewest unassigned
    def get_candidates():
        """Find unassigned vertices, ordered by a heuristic."""
        candidates = []
        for v in range(n):
            if assignment[v] == -1:
                # Count how many edges are "critical" (have 4 assigned of one color)
                critical_count = 0
                for ei in vertex_to_edges[v]:
                    if edge_unassigned[ei] == 1:
                        c0 = edge_color_count[ei][0]
                        c1 = edge_color_count[ei][1]
                        if c0 == 4 or c1 == 4:
                            critical_count += 1
                candidates.append((critical_count, -len(vertex_to_edges[v]), v))
        candidates.sort(reverse=True)
        return [v for _, _, v in candidates]
    
    def backtrack(depth):
        if depth > 200000:
            return True  # Safety: assume colorable if search is too deep
        if _time_remaining() < 0:
            return True  # Time out: assume colorable
        
        candidates = get_candidates()
        if not candidates:
            # All assigned; check no edge is monochromatic
            for i, e in enumerate(edges):
                if edge_color_count[i][0] == 5 or edge_color_count[i][1] == 5:
                    return False
            return True
        
        v = candidates[0]
        
        for color in [0, 1]:
            # Try assigning color to v
            assignment[v] = color
            # Update edge counts
            affected_edges = vertex_to_edges[v]
            for ei in affected_edges:
                edge_color_count[ei][color] += 1
                edge_unassigned[ei] -= 1
            
            # Propagate
            ok, _ = propagate()
            if ok:
                if backtrack(depth + 1):
                    return True
            
            # Undo propagation: this is tricky. We need to undo all changes.
            # Simpler approach: save state before trying.
            # Let's restructure: save/restore.
            
            # Undo
            for ei in affected_edges:
                edge_color_count[ei][color] -= 1
                edge_unassigned[ei] += 1
            assignment[v] = -1
        
        return False
    
    # The above backtrack has an issue with undoing propagation.
    # Let's use a cleaner save/restore approach.
    
    def backtrack2():
        if _time_remaining() < 0:
            return True
        
        candidates = get_candidates()
        if not candidates:
            for i, e in enumerate(edges):
                if edge_color_count[i][0] == 5 or edge_color_count[i][1] == 5:
                    return False
            return True
        
        v = candidates[0]
        
        # Save state
        saved_assignment = assignment[:]
        saved_edge_unassigned = edge_unassigned[:]
        saved_edge_color_count = [row[:] for row in edge_color_count]
        
        for color in [0, 1]:
            assignment[v] = color
            for ei in vertex_to_edges[v]:
                edge_color_count[ei][color] += 1
                edge_unassigned[ei] -= 1
            
            ok, _ = propagate()
            if ok:
                if backtrack2():
                    return True
            
            # Restore state
            assignment[:] = saved_assignment
            edge_unassigned[:] = saved_edge_unassigned
            for i in range(len(edges)):
                edge_color_count[i][0] = saved_edge_color_count[i][0]
                edge_color_count[i][1] = saved_edge_color_count[i][1]
        
        return False
    
    return backtrack2()


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Known small non-2-colorable 5-uniform hypergraphs
    # We'll try to find a minimal one by starting from a known construction
    # and pruning edges.
    
    # Start with the complete 5-uniform hypergraph on 9 vertices (126 edges)
    # and try to prune it down.
    
    # Actually, let's use a smarter approach: start with a random hypergraph
    # that is likely non-2-colorable, then prune.
    
    # A known fact: the 5-uniform hypergraph on 9 vertices with all C(9,5)=126 edges
    # is not 2-colorable. But we want fewer edges.
    
    # Let's try to construct a smaller one. The best known is 51 edges.
    # We'll use a random initialization and iterative pruning.
    
    random.seed(42)
    
    def try_prune(n, initial_edges, max_time=120.0):
        """
        Start with initial_edges and iteratively remove edges that don't
        break 2-colorability. Return the pruned hypergraph.
        """
        edges = list(initial_edges)
        start_time = time.time()
        
        # Shuffle edges to try different orders
        random.shuffle(edges)
        
        i = 0
        while i < len(edges) and (time.time() - start_time) < max_time:
            if _time_remaining() < 20:
                break
            
            # Try removing edge i
            if len(edges) <= 1:
                break
                
            edges_without = edges[:i] + edges[i+1:]
            
            # Check if still non-2-colorable
            if not is_2_colorable(n, edges_without):
                # Remove it
                edges = edges_without
                # Don't increment i, since the list shifted
            else:
                i += 1
        
        return edges
    
    # Strategy: try multiple random starting points and pruning
    best_edges = None
    best_n = None
    
    # First, try the complete hypergraph on 9 vertices
    n = 9
    initial = list(combinations(range(n), 5))
    pruned = try_prune(n, initial, max_time=60.0)
    
    if len(pruned) < (best_edges is None or len(best_edges)):
        best_edges = pruned
        best_n = n
    
    # Try other vertex counts
    for n_try in [8, 10, 11, 12]:
        if _time_remaining() < 20:
            break
        
        # Random hypergraph with enough edges to likely be non-2-colorable
        # For 5-uniform, we need roughly n * log(n) / 5 or so edges
        # Let's try a few random instances
        
        for attempt in range(3):
            if _time_remaining() < 20:
                break
            
            num_edges = int(n_try * n_try * 0.3)  # Rough heuristic
            num_edges = min(num_edges, len(list(combinations(range(n_try), 5))))
            
            all_edges = list(combinations(range(n_try), 5))
            random.shuffle(all_edges)
            initial = all_edges[:num_edges]
            
            # Check if non-2-colorable
            if not is_2_colorable(n_try, initial):
                pruned = try_prune(n_try, initial, max_time=30.0)
                if len(pruned) < (best_edges is None or len(best_edges)):
                    best_edges = pruned
                    best_n = n_try
    
    # If we didn't find anything, fall back to the complete hypergraph on 9
    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
    else:
        best_edges = [list(e) for e in best_edges]
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
