"""Search for a small non-2-colorable 5-uniform hypergraph."""
import random
import time
from itertools import combinations

DEADLINE = 150.0
START = time.time()

def _is_non_2colorable(n, edges):
    """Fast backtracking with unit propagation using bitmasks.
    Returns True if NOT 2-colorable (i.e., no valid 2-coloring exists)."""
    m = len(edges)
    if m == 0:
        return False
    
    # Build edge masks and vertex-edge adjacency
    edge_masks = []
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        for v in e:
            vert_edges[v].append(i)
    
    # State: for each edge, track count of color-0 and color-1 vertices
    # assign: -1 unassigned, 0 or 1
    assign = [-1] * n
    e_c0 = [0] * m
    e_c1 = [0] * m
    
    # Worklist: edges that need re-checking
    def propagate():
        """Unit propagation. Returns False on contradiction."""
        worklist = list(range(m))
        in_work = [False] * m
        for ei in range(m):
            in_work[ei] = True
        
        while worklist:
            ei = worklist.pop()
            in_work[ei] = False
            c0, c1 = e_c0[ei], e_c1[ei]
            unassigned = 5 - c0 - c1
            
            if c0 == 5 or c1 == 5:
                return False  # monochromatic edge
            
            forced_color = -1
            if c0 == 4 and unassigned == 1:
                forced_color = 1
            elif c1 == 4 and unassigned == 1:
                forced_color = 0
            
            if forced_color != -1:
                # Find the unassigned vertex in this edge
                for v in edges[ei]:
                    if assign[v] == -1:
                        assign[v] = forced_color
                        # Update all edges containing v
                        for ej in vert_edges[v]:
                            if forced_color == 0:
                                e_c0[ej] += 1
                            else:
                                e_c1[ej] += 1
                            if not in_work[ej] and ej != ei:
                                worklist.append(ej)
                                in_work[ej] = True
                        break
                # Re-check this edge
                if not in_work[ei]:
                    worklist.append(ei)
                    in_work[ei] = True
        
        return True
    
    def backtrack():
        """Find unassigned vertex with most constrained edges, branch."""
        # Pick vertex: prefer those in edges with fewest unassigned vertices
        best_v = -1
        best_unassigned_count = 99
        for v in range(n):
            if assign[v] != -1:
                continue
            min_unassigned = 99
            for ei in vert_edges[v]:
                c0, c1 = e_c0[ei], e_c1[ei]
                unassigned = 5 - c0 - c1
                if unassigned < min_unassigned:
                    min_unassigned = unassigned
            if min_unassigned < best_unassigned_count:
                best_unassigned_count = min_unassigned
                best_v = v
        
        if best_v == -1:
            # All assigned; verify no monochromatic edges
            for ei in range(m):
                if e_c0[ei] == 5 or e_c1[ei] == 5:
                    return True  # contradiction = not 2-colorable
            return False  # valid coloring found = 2-colorable
        
        # Save state
        old_assign = assign[:]
        old_e_c0 = e_c0[:]
        old_e_c1 = e_c1[:]
        
        for color in (0, 1):
            assign[best_v] = color
            for ei in vert_edges[best_v]:
                if color == 0:
                    e_c0[ei] += 1
                else:
                    e_c1[ei] += 1
            
            if propagate():
                if backtrack():
                    return True
            
            # Restore state
            assign[:] = old_assign
            e_c0[:] = old_e_c0
            e_c1[:] = old_e_c1
        
        return False
    
    if not propagate():
        return True
    return backtrack()


def _try_build(seed, max_edges=60, max_vertices=26):
    """Build a random hypergraph incrementally, then try to minimize edges."""
    rng = random.Random(seed)
    n_vertices = rng.randint(10, 18)
    all_possible = list(combinations(range(n_vertices), 5))
    rng.shuffle(all_possible)
    edges = []
    for e in all_possible:
        if len(edges) >= max_edges:
            break
        edges.append(list(e))
        if _is_non_2colorable(n_vertices, edges):
            # Try to reduce edges by removing redundant ones
            edges = _try_reduce(n_vertices, edges, rng)
            return n_vertices, edges
    return None


def _try_reduce(n, edges, rng, time_limit=5.0):
    """Try to remove redundant edges while maintaining non-2-colorability."""
    start_time = time.time()
    edges = [list(e) for e in edges]
    m = len(edges)
    
    # Try removing each edge in random order
    indices = list(range(m))
    rng.shuffle(indices)
    
    for idx in indices:
        if time.time() - start_time > time_limit:
            break
        if idx >= len(edges):
            continue
        # Try removing edge idx
        new_edges = edges[:idx] + edges[idx+1:]
        if _is_non_2colorable(n, new_edges):
            edges = new_edges
    
    return edges


def construct():
    best = None
    best_edges = float('inf')
    seed = 0
    while time.time() < DEADLINE - 15:
        seed += 1
        result = _try_build(seed)
        if result is not None:
            n, edges = result
            if len(edges) < best_edges:
                best_edges = len(edges)
                best = {'n': n, 'edges': edges}
                if best_edges <= 51:
                    break
    if best is None:
        # Fallback: complete 5-uniform on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    return best
