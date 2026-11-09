"""Rewritten program for cochromatic partitions of a pinned random graph."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    edges = P['edges']
    # adjacency as set of frozensets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    return n, adj, edge_set

def _is_clique(part, adj):
    """Check if all pairs in part are adjacent."""
    for i in range(len(part)):
        for j in range(i + 1, len(part)):
            if part[j] not in adj[part[i]]:
                return False
    return True

def _is_independent(part, adj):
    """Check if no pairs in part are adjacent."""
    for i in range(len(part)):
        for j in range(i + 1, len(part)):
            if part[j] in adj[part[i]]:
                return False
    return True

def _is_valid_part(part, adj):
    return _is_clique(part, adj) or _is_independent(part, adj)

def _try_add_vertex(part, v, adj):
    """Try adding v to part; return True if resulting part is clique or independent."""
    if not part:
        return True
    # Check if v is adjacent to all in part (clique case)
    all_adj = all(v in adj[u] for u in part)
    if all_adj:
        return True
    # Check if v is adjacent to none in part (independent case)
    none_adj = all(v not in adj[u] for u in part)
    if none_adj:
        return True
    return False

def _greedy_partition(adj, n, order):
    """Greedy: process vertices in given order, assign to first compatible part."""
    parts = []
    for v in order:
        placed = False
        for i, part in enumerate(parts):
            if _try_add_vertex(part, v, adj):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

def _random_order(n, rng):
    order = list(range(n))
    rng.shuffle(order)
    return order

def _backtracking_search(adj, n, max_parts, deadline, rng):
    """
    Backtracking search for a partition into <= max_parts parts.
    Each part must be a clique or independent set.
    Returns a list of parts (each a list of ints) or None.
    """
    import random
    
    parts = []  # list of (type, set_of_vertices) where type is 'clique' or 'independent'
    assigned = [False] * n
    vertex_order = list(range(n))
    # Heuristic: sort by degree descending for better pruning
    deg = [len(adj[v]) for v in range(n)]
    vertex_order.sort(key=lambda v: -deg[v])
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return [sorted(p[1]) for p in parts]
        
        v = vertex_order[idx]
        
        # Try adding v to each existing part
        for i, (ptype, pset) in enumerate(parts):
            if ptype == 'clique':
                # v must be adjacent to all in pset
                if all(v in adj[u] for u in pset):
                    parts[i] = ('clique', pset | {v})
                    assigned[v] = True
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    parts[i] = ('clique', pset)
                    assigned[v] = False
            else:  # independent
                # v must NOT be adjacent to any in pset
                if all(v not in adj[u] for u in pset):
                    parts[i] = ('independent', pset | {v})
                    assigned[v] = True
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    parts[i] = ('independent', pset)
                    assigned[v] = False
        
        # Try creating a new part (if we haven't exceeded max_parts)
        if len(parts) < max_parts:
            # New part as clique containing v
            parts.append(('clique', {v}))
            assigned[v] = True
            result = backtrack(idx + 1)
            if result is not None:
                return result
            parts.pop()
            assigned[v] = False
            
            # New part as independent containing v
            parts.append(('independent', {v}))
            assigned[v] = True
            result = backtrack(idx + 1)
            if result is not None:
                return result
            parts.pop()
            assigned[v] = False
        
        return None
    
    return backtrack(0)

def construct():
    n, adj, edge_set = _build_graph()
    
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    import random
    rng = random.Random(42)
    
    # First, get a greedy baseline
    baseline_parts = _greedy_partition(adj, n, list(range(n)))
    best_parts = baseline_parts
    best_count = len(best_parts)
    
    # Try multiple random orderings with greedy
    for _ in range(200):
        if time.time() > deadline - 10.0:
            break
        order = _random_order(n, rng)
        parts = _greedy_partition(adj, n, order)
        if len(parts) < best_count:
            best_count = len(parts)
            best_parts = parts
    
    # Try backtracking for smaller numbers of parts
    # Start from best_count - 1 and go down to 2
    for target in range(best_count - 1, 1, -1):
        if time.time() > deadline - 5.0:
            break
        result = _backtracking_search(adj, n, target, deadline, rng)
        if result is not None:
            best_parts = result
            best_count = target
            break
    
    # Final validation and type conversion
    # Ensure all parts are nonempty lists of Python ints
    final_parts = []
    seen = set()
    for part in best_parts:
        p = [int(v) for v in part]
        if not p:
            continue
        # Validate
        if not _is_valid_part(p, adj):
            # Fallback: use a simple valid partition (each vertex its own part)
            # But that's 14 parts. Let's re-run greedy to be safe.
            final_parts = None
            break
        for v in p:
            if v in seen:
                final_parts = None
                break
            seen.add(v)
        if final_parts is None:
            break
        final_parts.append(p)
    
    if final_parts is None or len(final_parts) == 0:
        # Absolute fallback: each vertex alone (always valid, 14 parts)
        final_parts = [[v] for v in range(n)]
    
    # Ensure all vertices are covered exactly once
    all_vertices = sorted(v for part in final_parts for v in part)
    if all_vertices != list(range(n)):
        final_parts = [[v] for v in range(n)]
    
    # Convert to list of lists with Python ints
    return [list(part) for part in final_parts]

# EVOLVE-BLOCK-END
