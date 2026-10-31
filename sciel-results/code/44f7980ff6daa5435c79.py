"""Kissing number in dimension 11: Maximum Independent Set on norm-4 conflict graph."""
D = 11

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _build_candidates():
    """Build all nonzero integer vectors with norm^2 = 4."""
    cands = []

    # Type 1: weight-4 vectors in {-1, 0, 1}^11 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            cands.append(v)

    # Type 2: weight-1 vectors with entry ±2 (norm^2 = 4)
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            cands.append(v)

    return np.array(cands, dtype=np.int64)


def _color_bound(vertices, adj):
    """Greedy coloring to get an upper bound on MIS size. Returns list of color classes."""
    n = len(vertices)
    if n == 0:
        return []
    
    # Greedy sequential coloring
    color_of = [-1] * n
    colors_used = []
    for i in range(n):
        v = vertices[i]
        # Find smallest color not used by neighbors
        neighbor_colors = set()
        for nb in adj[v]:
            if nb in vertices:
                idx = vertices.index(nb)
                if color_of[idx] >= 0:
                    neighbor_colors.add(color_of[idx])
        c = 0
        while c in neighbor_colors:
            c += 1
        color_of[i] = c
        if c >= len(colors_used):
            colors_used.append([])
        colors_used[c].append(v)
    
    return colors_used


def _tomita_mis(adj, n, deadline, rng):
    """Tomita-style branch-and-bound for Maximum Independent Set."""
    best_set = []
    
    # Use a stack-based approach with coloring bound
    # State: (vertices_list, current_set)
    # We'll use a recursive-like approach with explicit stack
    
    def _color_sort(vertices, adj):
        """Sort vertices by degree for better branching, and compute color bound."""
        if not vertices:
            return [], 0
        
        # Compute degrees within the subgraph
        vset = set(vertices)
        degrees = {}
        for v in vertices:
            deg = 0
            for nb in adj[v]:
                if nb in vset:
                    deg += 1
            degrees[v] = deg
        
        # Sort by degree descending
        sorted_verts = sorted(vertices, key=lambda x: -degrees[x])
        
        # Greedy coloring for upper bound
        color_of = {}
        num_colors = 0
        for v in sorted_verts:
            neighbor_colors = set()
            for nb in adj[v]:
                if nb in vset and nb in color_of:
                    neighbor_colors.add(color_of[nb])
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            if c >= num_colors:
                num_colors = c + 1
        
        return sorted_verts, num_colors
    
    # Initial coloring and sorting
    all_vertices = list(range(n))
    sorted_verts, color_bound = _color_sort(all_vertices, adj)
    
    # Branch and bound
    # We process vertices in order, deciding include or exclude
    # Use the color bound to prune
    
    stack = [(sorted_verts, 0, [])]  # (vertices, start_index, current_set)
    
    while stack and time.time() < deadline:
        verts, start, curr_set = stack.pop()
        
        if not verts:
            if len(curr_set) > len(best_set):
                best_set = list(curr_set)
            continue
        
        # If color bound + current size <= best, prune
        if len(curr_set) + len(verts) <= len(best_set):
            continue
        
        # Try to add the first vertex
        v = verts[0]
        remaining = verts[1:]
        
        # Include v: remove v and its neighbors from remaining
        v_neighbors = set(adj[v])
        new_verts = [u for u in remaining if u not in v_neighbors]
        
        # Get color bound for new_verts
        if new_verts:
            _, cb = _color_sort(new_verts, adj)
            if len(curr_set) + 1 + cb > len(best_set):
                stack.append((new_verts, 0, curr_set + [v]))
        
        # Exclude v
        if len(curr_set) + len(remaining) > len(best_set):
            stack.append((remaining, 0, curr_set))
    
    return best_set


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    candidates = _build_candidates()
    n = len(candidates)

    # Compute dot products: conflict if dot > 2
    dots = candidates @ candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Build adjacency lists
    adj = [np.where(conflicts[i])[0] for i in range(n)]

    # Convert to list of lists for faster access
    adj_list = [list(a) for a in adj]

    # Run Tomita MIS
    best_selected = _tomita_mis(adj_list, n, deadline, None)

    if not best_selected:
        # Fallback: return all weight-1 vectors
        v1 = []
        for i in range(D):
            for s in (2, -2):
                v = np.zeros(D, dtype=np.int64)
                v[i] = s
                v1.append(v)
        return np.array(v1, dtype=np.int64)

    return candidates[best_selected]
# EVOLVE-BLOCK-END
