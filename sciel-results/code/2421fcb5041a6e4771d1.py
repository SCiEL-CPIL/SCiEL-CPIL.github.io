import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj

def _greedy_partition(n, adj):
    """Greedy: assign each vertex to first compatible part, else new part."""
    parts = []
    part_types = []  # 'clique' or 'indep'
    # Sort vertices by degree descending to help greedy
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    for v in order:
        placed = False
        for idx, part in enumerate(parts):
            ptype = part_types[idx]
            if ptype == 'clique':
                if all(v in adj[u] for u in part):
                    part.append(v)
                    placed = True
                    break
            else:
                if all(v not in adj[u] for u in part):
                    part.append(v)
                    placed = True
                    break
        if not placed:
            parts.append([v])
            # Start as clique? Or indep? 
            # A single vertex is both. 
            # Usually, it's safer to start as the type that allows more future additions?
            # Let's try starting as 'clique' first, as cliques are often more restrictive? 
            # Actually, if we start as 'clique', we can only add neighbors.
            # If we start as 'indep', we can only add non-neighbors.
            # Let's just pick 'clique' arbitrarily or based on degree?
            # For the greedy, let's stick to the seed's logic which didn't track type explicitly 
            # but checked if the *result* was clique or indep.
            # My explicit type tracking is better.
            part_types.append('clique')
    return parts, part_types

def _exact_search(n, adj, deadline):
    """
    Exact branch and bound to find minimum cochromatic number.
    Returns the partition (list of lists) or None if timeout.
    """
    # Initial upper bound from greedy
    init_parts, init_types = _greedy_partition(n, adj)
    best_k = len(init_parts)
    best_parts = [list(p) for p in init_parts]
    
    # Sort vertices by degree descending for better pruning
    vertex_order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    # State for backtracking
    # parts: list of lists of vertices
    # types: list of 'clique' or 'indep'
    # assigned: boolean array or set
    # We assign vertices in vertex_order
    
    parts = []
    types = []
    assigned = [False] * n
    
    def can_add(part, v, ptype):
        if ptype == 'clique':
            return all(v in adj[u] for u in part)
        else:
            return all(v not in adj[u] for u in part)

    def backtrack(idx):
        nonlocal best_k, best_parts
        if time.time() > deadline:
            return False
        
        if idx == n:
            if len(parts) < best_k:
                best_k = len(parts)
                best_parts = [list(p) for p in parts]
            return True
        
        v = vertex_order[idx]
        
        # Try adding v to existing parts
        for i in range(len(parts)):
            if can_add(parts[i], v, types[i]):
                parts[i].append(v)
                assigned[v] = True
                if backtrack(idx + 1):
                    return True
                parts[i].pop()
                assigned[v] = False
        
        # Try creating a new part
        # We can create it as 'clique' or 'indep'. 
        # However, a new part with a single vertex is both.
        # The type matters for future additions.
        # We should try both types? 
        # If we start as 'clique', we restrict future additions to neighbors.
        # If we start as 'indep', we restrict future additions to non-neighbors.
        # It's possible that starting as 'clique' leads to a dead end, while 'indep' works, or vice versa.
        # So we must try both.
        
        # Optimization: If we create a new part, the order of parts doesn't matter for the count, 
        # but it does for the search space symmetry. 
        # To avoid symmetric branches (e.g., part A=[v], part B=[w] vs part A=[w], part B=[v]),
        # we can enforce an ordering, but it's complex with types.
        # Given N=14, the symmetry might be manageable without complex canonicalization.
        
        if len(parts) + 1 < best_k:
            # Try new part as 'clique'
            parts.append([v])
            types.append('clique')
            assigned[v] = True
            if backtrack(idx + 1):
                return True
            parts.pop()
            types.pop()
            assigned[v] = False
            
            # Try new part as 'indep'
            parts.append([v])
            types.append('indep')
            assigned[v] = True
            if backtrack(idx + 1):
                return True
            parts.pop()
            types.pop()
            assigned[v] = False
        
        return False

    found = backtrack(0)
    return best_parts

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n, adj = _build_graph()
    
    # Try exact search
    result = _exact_search(n, adj, deadline)
    
    if result is None:
        # Fallback to greedy if exact search fails or times out
        parts, _ = _greedy_partition(n, adj)
        result = parts
        
    # Ensure types are correct (list of lists of ints)
    final_parts = []
    for part in result:
        final_parts.append([int(v) for v in part])
        
    return final_parts

# EVOLVE-BLOCK-END
