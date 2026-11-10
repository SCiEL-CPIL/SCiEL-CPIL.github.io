"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _is_clique(vertices, adj):
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            if vertices[j] not in adj[vertices[i]]:
                return False
    return True

def _is_independent(vertices, adj):
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            if vertices[j] in adj[vertices[i]]:
                return False
    return True

def _greedy_partition(n, adj):
    """Greedy: assign each vertex to first compatible part, else new part."""
    parts = []
    part_types = []  # 'clique' or 'indep'
    for v in range(n):
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
            part_types.append('clique')  # start as clique (single vertex is both)
    return parts

def _try_extend(part, v, ptype, adj):
    """Check if v can be added to part with given type."""
    if ptype == 'clique':
        return all(v in adj[u] for u in part)
    else:
        return all(v not in adj[u] for u in part)

def _branch_and_bound(n, adj, deadline):
    """Find minimum cochromatic number via branch and bound."""
    best = [n]  # worst case: n parts
    best_parts = None
    
    # Precompute: for each vertex, which other vertices can be in same clique/indep set
    # Use a recursive assignment approach
    
    unassigned = list(range(n))
    
    def can_add_to_part(part, v, ptype):
        if ptype == 'clique':
            return all(v in adj[u] for u in part)
        else:
            return all(v not in adj[u] for u in part)
    
    def solve(assigned_count, parts, part_types, current_depth):
        nonlocal best, best_parts
        if time.time() > deadline:
            return False
        
        if current_depth >= best[0]:
            return False
        
        if assigned_count == n:
            if current_depth < best[0]:
                best[0] = current_depth
                best_parts = [list(p) for p in parts]
            return True
        
        # Pick the next unassigned vertex (first one)
        v = None
        for i in range(n):
            if i not in assigned_set:
                v = i
                break
        if v is None:
            return False
        
        # Try placing v in each existing part
        for idx in range(len(parts)):
            if can_add_to_part(parts[idx], v, part_types[idx]):
                parts[idx].append(v)
                assigned_set.add(v)
                if solve(assigned_count + 1, parts, part_types, current_depth):
                    return True
                parts[idx].pop()
                assigned_set.remove(v)
        
        # Try creating a new part (as clique)
        if current_depth + 1 < best[0]:
            parts.append([v])
            part_types.append('clique')
            assigned_set.add(v)
            if solve(assigned_count + 1, parts, part_types, current_depth + 1):
                return True
            parts.pop()
            part_types.pop()
            assigned_set.remove(v)
        
        # Try creating a new part (as independent set)
        if current_depth + 1 < best[0]:
            parts.append([v])
            part_types.append('indep')
            assigned_set.add(v)
            if solve(assigned_count + 1, parts, part_types, current_depth + 1):
                return True
            parts.pop()
            part_types.pop()
            assigned_set.remove(v)
        
        return False
    
    assigned_set = set()
    found = solve(0, [], [], 0)
    return best_parts if best_parts else None

def _exact_search(n, adj, deadline):
    """Exact search: try k=1,2,3,... parts until we find a valid partition."""
    # Lower bound: clique number of the graph
    max_clique = 1
    for subset_size in range(2, n+1):
        if time.time() > deadline:
            break
        found = False
        for combo in itertools.combinations(range(n), subset_size):
            if _is_clique(combo, adj):
                max_clique = subset_size
                found = True
        if not found:
            break
    
    # Lower bound: independence number consideration
    # Also, alpha(G) gives lower bound on chromatic number of complement
    # For cochromatic: each part is clique or indep set
    
    # Try increasing number of parts
    for k in range(max_clique, n+1):
        if time.time() > deadline:
            break
        result = _try_k_parts(n, adj, k, deadline)
        if result is not None:
            return result
    return None

def _try_k_parts(n, adj, k, deadline):
    """Try to partition into exactly k parts, each a clique or independent set."""
    # Each part can be type 0 (clique) or 1 (independent set)
    # We need to assign each vertex to a part and choose the type of each part
    
    # Strategy: enumerate all type assignments for k parts (2^k possibilities)
    # For each type assignment, check if we can assign vertices
    
    best_result = None
    
    for type_mask in range(1 << k):
        if time.time() > deadline:
            return best_result
        
        part_types = []
        for i in range(k):
            part_types.append('clique' if (type_mask >> i) & 1 else 'indep')
        
        # Try to assign vertices to parts
        assignment = [-1] * n
        result = _assign_vertices(n, adj, k, part_types, assignment, 0, deadline)
        if result is not None:
            # Build the partition
            parts = [[] for _ in range(k)]
            for v in range(n):
                parts[assignment[v]].append(v)
            # Remove empty parts
            parts = [p for p in parts if p]
            return parts
    
    return best_result

def _assign_vertices(n, adj, k, part_types, assignment, idx, deadline):
    """Recursively assign vertex idx to one of k parts."""
    if time.time() > deadline:
        return None
    
    if idx == n:
        return True
    
    v = idx
    for p in range(k):
        part_type = part_types[p]
        # Check if v can be added to part p
        ok = True
        for u in range(idx):
            if assignment[u] == p:
                if part_type == 'clique':
                    if v not in adj[u]:
                        ok = False
                        break
                else:
                    if v in adj[u]:
                        ok = False
                        break
        if ok:
            assignment[idx] = p
            if _assign_vertices(n, adj, k, part_types, assignment, idx + 1, deadline):
                return True
            assignment[idx] = -1
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n, adj = _build_graph()
    
    # First get a greedy solution as upper bound
    greedy_parts = _greedy_partition(n, adj)
    greedy_count = len(greedy_parts)
    
    # Try exact search for better solution
    best_parts = greedy_parts
    
    # Try to improve: search for k < greedy_count
    for k in range(1, greedy_count):
        if time.time() > deadline - 5.0:
            break
        result = _try_k_parts(n, adj, k, deadline - 5.0)
        if result is not None:
            best_parts = result
            break
    
    # Ensure proper types
    final_parts = []
    for part in best_parts:
        final_parts.append([int(v) for v in part])
    
    return final_parts

# EVOLVE-BLOCK-END
