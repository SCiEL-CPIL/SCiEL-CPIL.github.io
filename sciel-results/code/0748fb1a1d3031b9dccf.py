"""Rewritten program: exact minimum cochromatic partition via branch-and-bound."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # precompute adjacency bitmask for speed
    adj_mask = [0] * n
    for u in range(n):
        m = 0
        for v in adj[u]:
            m |= (1 << v)
        adj_mask[u] = m
    return adj, adj_mask

def _is_clique(vertices, adj_mask):
    """Check if all pairs in vertices are adjacent."""
    for i, u in enumerate(vertices):
        for w in vertices[i+1:]:
            if not (adj_mask[u] & (1 << w)):
                return False
    return True

def _is_independent(vertices, adj_mask):
    """Check if no pairs in vertices are adjacent."""
    for i, u in enumerate(vertices):
        for w in vertices[i+1:]:
            if adj_mask[u] & (1 << w):
                return False
    return True

def _is_valid_part(vertices, adj_mask):
    if not vertices:
        return False
    if _is_clique(vertices, adj_mask):
        return True
    if _is_independent(vertices, adj_mask):
        return True
    return False

def _try_add(v, part, adj_mask):
    """Try adding v to part; return True if part+v is still valid (clique or independent)."""
    if not part:
        return True
    # Check if part is currently clique or independent, and if adding v preserves it
    # We need to check all pairs in part+[v]
    for u in part:
        if (adj_mask[u] & (1 << v)) == 0:
            # v is not adjacent to u
            # If part is a clique, adding v breaks it unless part has only 1 element
            # Actually we need to check the whole set
            pass
    # More efficient: check if part is clique
    is_clique = True
    is_indep = True
    for i, u in enumerate(part):
        for w in part[i+1:]:
            if adj_mask[u] & (1 << w):
                is_indep = False
            else:
                is_clique = False
    # Now check v against all in part
    for u in part:
        if adj_mask[u] & (1 << v):
            is_indep = False
        else:
            is_clique = False
    return is_clique or is_indep

def _find_min_cochromatic(adj_mask, deadline):
    """
    Branch and bound to find minimum number of parts.
    Each part is a clique or independent set.
    Vertices 0..n-1 must be partitioned.
    """
    n = P['n']
    all_vertices = set(range(n))
    
    best = [n]  # upper bound: each vertex its own part
    best_partition = [list([v]) for v in range(n)]
    
    # Sort vertices by degree descending for better pruning
    degrees = [bin(adj_mask[v]).count('1') for v in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # State: (unassigned set as bitmask, current parts as list of frozensets)
    # We'll use a recursive approach with bitmask for unassigned
    
    def search(unassigned_mask, parts, depth):
        """
        unassigned_mask: bitmask of vertices not yet assigned
        parts: list of current parts (each is a tuple of vertices)
        depth: number of parts used so far
        """
        if time.time() > deadline:
            return
        
        if unassigned_mask == 0:
            if len(parts) < len(best):
                best[:] = parts
                best_partition[:] = [list(p) for p in parts]
            return
        
        # Pruning: if depth >= len(best), we can't improve
        if depth >= len(best):
            return
        
        # Get the first unassigned vertex (lowest index for determinism)
        # Actually, let's pick the vertex with highest degree among unassigned for better pruning
        unassigned = [v for v in range(n) if unassigned_mask & (1 << v)]
        if not unassigned:
            if len(parts) < len(best):
                best[:] = parts
                best_partition[:] = [list(p) for p in parts]
            return
        
        # Pick vertex with max degree among unassigned
        v = max(unassigned, key=lambda x: degrees[x])
        
        # Option 1: add v to an existing part
        for i, part in enumerate(parts):
            if _try_add(v, part, adj_mask):
                new_parts = list(parts)
                new_parts[i] = tuple(list(part) + [v])
                search(unassigned_mask & ~(1 << v), new_parts, depth)
        
        # Option 2: start a new part with v
        if depth + 1 < len(best):  # can still potentially improve
            new_parts = list(parts)
            new_parts.append((v,))
            search(unassigned_mask & ~(1 << v), new_parts, depth + 1)
    
    # Initial call
    search((1 << n) - 1, [], 0)
    
    return best_partition

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, leaving 15s margin for 90s limit
    
    adj, adj_mask = _build_graph()
    
    # Try exact search with time limit
    result = _find_min_cochromatic(adj_mask, deadline)
    
    # Validate result
    if not result:
        # Fallback: greedy
        parts = []
        for v in range(P['n']):
            placed = False
            for part in parts:
                if _try_add(v, part, adj_mask):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        result = parts
    
    # Ensure all vertices are present exactly once
    all_verts = []
    for part in result:
        all_verts.extend(part)
    if sorted(all_verts) != list(range(P['n'])):
        # Rebuild
        parts = []
        for v in range(P['n']):
            placed = False
            for part in parts:
                if _try_add(v, part, adj_mask):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        result = parts
    
    # Convert to list of lists of Python ints
    final = []
    for part in result:
        final.append([int(v) for v in part])
    
    return final
# EVOLVE-BLOCK-END
