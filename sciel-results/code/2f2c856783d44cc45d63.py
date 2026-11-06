"""
Construct a 5-uniform hypergraph with no proper 2-coloring (Property B fails).
Goal: minimize number of edges. Best known: 51 edges.

Strategy:
1. Hard-code the known 51-edge construction if we can verify it.
2. Use bitmask-based backtracking with unit propagation for fast checking.
3. Use local search / greedy edge removal to try to find smaller examples.
4. If nothing better found within deadline, return the 51-edge construction.
"""
import time
import itertools
import random

DEADLINE = 140.0  # safety margin before 150s kill

def _now():
    return time.time()

def _check_not_2_colorable(edges, n):
    """
    Bitmask-based backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable (no proper 2-coloring exists).
    """
    # Precompute for each vertex, the list of edge indices containing it
    vert_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vert_edges[v].append(i)
    
    e = len(edges)
    
    # Use bitmasks for colors: color_mask[v] = 0 for uncolored, 1 for color 0, 2 for color 1
    # Actually, let's use a different approach:
    # For each edge, track counts of color 0 and color 1.
    
    # We'll use recursive backtracking with unit propagation.
    # State: colors array, edge_counts array (for each edge, [count0, count1])
    
    # For efficiency with small n and e, we can use a simpler approach:
    # Try all 2^n colorings? No, n can be up to 26, so 2^26 is too large.
    # But for n <= 20, 2^20 = 1M is feasible.
    
    # Let's use backtracking with unit propagation.
    
    def solve(colors, edge_c0, edge_c1):
        """
        colors: list of -1 (uncolored), 0, 1
        edge_c0: list of count of 0s in each edge
        edge_c1: list of count of 1s in each edge
        """
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for i, edge in enumerate(edges):
                c0, c1 = edge_c0[i], edge_c1[i]
                if c0 == 5:
                    return False  # conflict
                if c1 == 5:
                    return False  # conflict
                # Check if we can force a color
                uncolored_in_edge = [v for v in edge if colors[v] == -1]
                if len(uncolored_in_edge) == 1:
                    v = uncolored_in_edge[0]
                    if c0 == 4:
                        # Must be color 1
                        colors[v] = 1
                        edge_c1[i] += 1
                        for j in vert_edges[v]:
                            edge_c1[j] += 1
                        changed = True
                    elif c1 == 4:
                        # Must be color 0
                        colors[v] = 0
                        edge_c0[i] += 1
                        for j in vert_edges[v]:
                            edge_c0[j] += 1
                        changed = True
                # Check for new conflicts after propagation
                if edge_c0[i] == 5 or edge_c1[i] == 5:
                    return False
        
        # Find first uncolored vertex
        for v in range(n):
            if colors[v] == -1:
                break
        else:
            return True  # all colored, no conflict
        
        # Try both colors
        for color in [0, 1]:
            new_colors = colors[:]
            new_edge_c0 = edge_c0[:]
            new_edge_c1 = edge_c1[:]
            
            new_colors[v] = color
            if color == 0:
                for j in vert_edges[v]:
                    new_edge_c0[j] += 1
            else:
                for j in vert_edges[v]:
                    new_edge_c1[j] += 1
            
            if solve(new_colors, new_edge_c0, new_edge_c1):
                return True
        
        return False
    
    colors = [-1] * n
    edge_c0 = [0] * e
    edge_c1 = [0] * e
    
    # If solve returns True, a valid coloring exists -> hypergraph IS 2-colorable
    # We want NOT 2-colorable, so return not solve(...)
    return not solve(colors, edge_c0, edge_c1)

def _known_51_edges():
    """
    Return a known 51-edge 5-uniform hypergraph that is not 2-colorable.
    Based on the construction from the literature.
    
    One known construction: use the 20 vertices of the 5-dimensional hypercube?
    No, that's 32 vertices.
    
    Let me think of a specific construction.
    
    Actually, a well-known small example is based on the Petersen graph or similar.
    But for 5-uniform, let's use a construction from the paper.
    
    For now, let's construct one using the following idea:
    Take vertices 0..19 (20 vertices).
    Use edges that are "balanced" in some way.
    
    Actually, let's just return a valid 51-edge hypergraph.
    I'll use a construction based on the 51 edges from the known result.
    
    Since I don't have the exact 51 edges memorized, let me use a different approach:
    Return the complete 5-uniform hypergraph on 9 vertices (126 edges) as fallback,
    but try to find a smaller one via search.
    
    Wait, the task says best known is 51. Let me try to construct one.
    
    A known construction: 
    - Take the 20 vertices as pairs (i, j) where i in {0,1,2,3,4}, j in {0,1,2,3,4}
    - No, that's 25 vertices.
    
    Let me use a simpler approach: hard-code a 51-edge hypergraph that I can verify.
    If I can't, I'll use search.
    """
    # For now, return None to indicate we need to search
    return None

def construct():
    start = _now()
    
    # First, try to find a small non-2-colorable hypergraph via random search + local improvement
    
    best_edges = None
    best_e = 999
    
    rng = random.Random(12345)
    
    # Try random search for small n and e
    for n in range(8, 21):
        if _now() > DEADLINE:
            break
        
        max_e_to_try = min(60, 200)
        for e in range(10, max_e_to_try + 1):
            if _now() > DEADLINE:
                break
            
            all_edges = list(itertools.combinations(range(n), 5))
            if len(all_edges) < e:
                continue
            
            # Try a few random samples
            n_trials = 20 if e <= 30 else 5
            for _ in range(n_trials):
                if _now() > DEADLINE:
                    break
                
                edges = [list(edge) for edge in rng.sample(all_edges, e)]
                
                if _check_not_2_colorable(edges, n):
                    if e < best_e:
                        best_e = e
                        best_edges = edges
                        print(f"Found non-2-colorable with n={n}, e={e}")
    
    # If we found something with e <= 51, try to reduce it further via edge removal
    if best_edges is not None and best_e <= 51:
        n = len(range(best_e))  # actually n is the number of vertices
        # We need to recover n
        max_v = max(max(edge) for edge in best_edges)
        n = max_v + 1
        
        # Try removing edges one by one
        improved = True
        while improved:
            improved = False
            for i in range(len(best_edges)):
                if _now() > DEADLINE:
                    break
                trial_edges = best_edges[:i] + best_edges[i+1:]
                if _check_not_2_colorable(trial_edges, n):
                    best_edges = trial_edges
                    best_e -= 1
                    improved = True
                    print(f"Reduced to e={best_e}")
    
    # If we have a good result, return it
    if best_edges is not None and best_e <= 51:
        max_v = max(max(edge) for edge in best_edges)
        n = max_v + 1
        return {'n': n, 'edges': best_edges}
    
    # Fallback: return a known construction or the complete hypergraph on 9 vertices
    # The complete 5-uniform hypergraph on 9 vertices has C(9,5) = 126 edges
    # It is definitely not 2-colorable (any 2-coloring of 9 vertices has at least
    # one monochromatic 5-subset by pigeonhole: if 5 or more are same color, done;
    # if at most 4 of each, then 4+4=8, but we have 9 vertices, so one color has at least 5)
    
    # Actually, let's verify: with 9 vertices and 2 colors, by pigeonhole, one color has at least 5 vertices.
    # Any 5-subset of those 5+ vertices is monochromatic.
    # So the complete 5-uniform hypergraph on 9 vertices is not 2-colorable.
    
    edges = [list(edge) for edge in itertools.combinations(range(9), 5)]
    return {'n': 9, 'edges': edges}
