import time
import random
from itertools import combinations

def _precompute_k32():
    """Precompute the known valid 32-vertex 4-coloring (seed construction)."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

# Precompute at module load
_K32_COLORS = _precompute_k32()

def _validate(colors, n):
    """Check that no color class contains a triangle. Returns True if valid."""
    # Build adjacency sets for each color
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1

    # For each color, check for triangles
    for c in range(4):
        # Build adjacency list
        adj_list = [[] for _ in range(n)]
        for (i, j) in adj[c]:
            adj_list[i].append(j)
            adj_list[j].append(i)

        # Check each vertex's neighbors for edges between them
        for i in range(n):
            nbrs = adj_list[i]
            for a in range(len(nbrs)):
                for b in range(a + 1, len(nbrs)):
                    if (nbrs[a], nbrs[b]) in adj[c] or (nbrs[b], nbrs[a]) in adj[c]:
                        return False
    return True

def _greedy_extend(base_colors, base_n, max_n=50, max_time=100):
    """
    Try to extend a valid coloring from base_n to larger n using greedy vertex addition.
    For each new vertex, try to assign colors to edges to existing vertices
    such that no triangle is formed in any color class.
    """
    start_time = time.time()
    deadline = start_time + max_time

    # Start with the base coloring
    current_n = base_n
    current_colors = list(base_colors)

    # For greedy extension, we need to track, for each pair of existing vertices,
    # which colors are "forbidden" for a new vertex to avoid creating triangles.
    # Specifically, if edge (u,v) has color c, then a new vertex w connected to
    # both u and v cannot have edge (w,u) and (w,v) both with color c.

    # We'll try to add vertices one at a time
    for target_n in range(base_n + 1, max_n + 1):
        if time.time() > deadline:
            break
        # Try to add vertex (target_n - 1)
        # For each existing vertex i, we need to choose a color for edge (i, target_n-1)
        # such that no triangle is formed.
        #
        # Constraint: For any pair (i, j) of existing vertices with edge color c,
        # we cannot have both (i, new) and (j, new) colored c.
        #
        # This is a CSP. Let's use a simple backtracking/greedy approach.

        new_vertex = target_n - 1
        # edges_to_assign[i] = color for edge (i, new_vertex), for i in range(current_n)
        # We need to find an assignment.

        # For each color c, and each pair (i,j) with edge(i,j)=c,
        # at most one of i or j can be connected to new_vertex with color c.
        # So for each color c, the set of vertices connected to new_vertex with color c
        # must be an independent set in the color-c subgraph.

        # Greedy: try random assignments
        found = False
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Random assignment
            assignment = [random.randint(0, 3) for _ in range(current_n)]

            # Check validity: for each color c, the set of vertices with assignment[i]=c
            # must be an independent set in the color-c graph among existing vertices.
            valid = True
            for c in range(4):
                vertices_with_c = [i for i in range(current_n) if assignment[i] == c]
                # Check no two vertices in this set are connected by color c
                for a in range(len(vertices_with_c)):
                    for b in range(a + 1, len(vertices_with_c)):
                        i, j = vertices_with_c[a], vertices_with_c[b]
                        # Get color of edge (i, j)
                        if i < j:
                            idx = i * current_n - i * (i + 1) // 2 + (j - i - 1)
                        else:
                            idx = j * current_n - j * (j + 1) // 2 + (i - j - 1)
                        if current_colors[idx] == c:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                # Found a valid extension
                # Add the new edges to current_colors
                # Edges are (i, new_vertex) for i in range(current_n), in lex order
                # Since new_vertex > all existing, edges are (i, new_vertex) for i=0..current_n-1
                current_colors.extend(assignment)
                current_n = target_n
                found = True
                break

        if not found:
            break

    return current_n, current_colors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit

    best_n = 32
    best_colors = _K32_COLORS

    # Try greedy extension from the 32-vertex base
    try:
        ext_n, ext_colors = _greedy_extend(_K32_COLORS, 32, max_n=50, max_time=90)
        if ext_n > best_n and ext_n <= 62:
            # Validate before accepting
            if _validate(ext_colors, ext_n):
                best_n = ext_n
                best_colors = ext_colors
            else:
                # Fallback to base
                best_n = 32
                best_colors = _K32_COLORS
        else:
            best_n = 32
            best_colors = _K32_COLORS
    except Exception:
        best_n = 32
        best_colors = _K32_COLORS

    # Final check: if we haven't used too much time, validate the best
    if time.time() < deadline:
        if not _validate(best_colors, best_n):
            # Shouldn't happen, but fallback to known valid
            best_n = 32
            best_colors = _K32_COLORS

    return {'n': best_n, 'colors': best_colors}
