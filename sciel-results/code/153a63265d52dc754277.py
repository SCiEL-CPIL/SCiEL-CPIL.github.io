"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_clique(vertices, adj):
    if len(vertices) <= 1:
        return True
    vs = list(vertices)
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if vs[j] not in adj[vs[i]]:
                return False
    return True

def _is_independent(vertices, adj):
    if len(vertices) <= 1:
        return True
    vs = list(vertices)
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if vs[j] in adj[vs[i]]:
                return False
    return True

def _is_valid_part(vertices, adj):
    return _is_clique(vertices, adj) or _is_independent(vertices, adj)

def _greedy_partition(adj):
    n = P['n']
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            new_part = part + [v]
            if _is_valid_part(new_part, adj):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

def _dfs_cochromatic(adj, unassigned, parts, max_parts, deadline):
    """
    DFS to try to assign all unassigned vertices into at most max_parts parts.
    Each part must be a clique or independent set.
    Returns a valid partition if found, else None.
    """
    if time.time() > deadline:
        return None
    
    if not unassigned:
        # All assigned, return the partition
        return [list(p) for p in parts]
    
    v = unassigned[0]
    rest = unassigned[1:]
    
    # Try to place v in an existing part
    for i, part in enumerate(parts):
        new_part = part + [v]
        if _is_valid_part(new_part, adj):
            parts[i] = new_part
            result = _dfs_cochromatic(adj, rest, parts, max_parts, deadline)
            if result is not None:
                return result
            parts[i] = part
    
    # Try to start a new part (only if we haven't used all allowed parts)
    if len(parts) < max_parts:
        new_parts = parts + [[v]]
        result = _dfs_cochromatic(adj, rest, new_parts, max_parts, deadline)
        if result is not None:
            return result
    
    return None

def _try_k_parts(adj, k, deadline):
    """Try to find a cochromatic partition with exactly k parts."""
    # Start with greedy to get an initial assignment, then try to improve
    # Use DFS with k parts
    result = _dfs_cochromatic(adj, list(range(P['n'])), [], k, deadline)
    return result

def construct():
    adj = _build_adj()
    n = P['n']
    
    # Get greedy solution as baseline
    greedy_parts = _greedy_partition(adj)
    best_parts = [list(p) for p in greedy_parts]
    best_count = len(best_parts)
    
    # Set deadline: use 55 seconds out of 60 to be safe
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Try to find solutions with fewer parts
    # Start from best_count - 1 and go down to 1
    for k in range(best_count - 1, 0, -1):
        if time.time() > deadline:
            break
        result = _try_k_parts(adj, k, deadline)
        if result is not None:
            best_parts = result
            best_count = k
            # If we found a 1-part solution, that's optimal
            if k == 1:
                break
    
    # Convert to proper Python types
    final_parts = []
    for part in best_parts:
        final_parts.append([int(v) for v in part])
    
    # Ensure all vertices are covered exactly once
    all_vertices = sorted(v for part in final_parts for v in part)
    assert all_vertices == list(range(n)), f"Invalid partition: {all_vertices}"
    
    # Verify each part is valid
    for part in final_parts:
        assert _is_valid_part(part, adj), f"Invalid part: {part}"
    
    return final_parts
# EVOLVE-BLOCK-END
