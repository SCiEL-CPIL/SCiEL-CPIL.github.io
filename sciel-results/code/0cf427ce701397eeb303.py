"""Rewritten program using backtracking with branch-and-bound for cochromatic partition."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

# Precompute adjacency matrix
N = P['n']
adj = [[False] * N for _ in range(N)]
for u, v in P['edges']:
    adj[u][v] = True
    adj[v][u] = True

def is_clique(vertices):
    """Check if all pairs in vertices are adjacent."""
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if not adj[vertices[i]][vertices[j]]:
                return False
    return True

def is_independent(vertices):
    """Check if no pairs in vertices are adjacent."""
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if adj[vertices[i]][vertices[j]]:
                return False
    return True

def can_add_to_clique(part, v):
    """Check if v can be added to part and still form a clique."""
    for u in part:
        if not adj[u][v]:
            return False
    return True

def can_add_to_independent(part, v):
    """Check if v can be added to part and still form an independent set."""
    for u in part:
        if adj[u][v]:
            return False
    return True

def backtrack(remaining, parts, best):
    """
    Backtracking search with branch-and-bound.
    remaining: tuple of unassigned vertices
    parts: list of (type, vertices) where type is 'clique' or 'independent'
    best: current best number of parts found
    Returns: best number of parts found
    """
    if not remaining:
        return len(parts)
    
    # Lower bound: at least len(remaining) parts if each vertex needs its own part
    # Better lower bound: we need at least 1 more part if remaining is non-empty
    if len(parts) >= best:
        return best
    
    v = remaining[0]
    rest = remaining[1:]
    
    # Try adding v to each existing part
    for i, (ptype, pverts) in enumerate(parts):
        if ptype == 'clique' and can_add_to_clique(pverts, v):
            new_parts = parts.copy()
            new_parts[i] = ('clique', pverts + [v])
            result = backtrack(rest, new_parts, best)
            if result < best:
                best = result
        elif ptype == 'independent' and can_add_to_independent(pverts, v):
            new_parts = parts.copy()
            new_parts[i] = ('independent', pverts + [v])
            result = backtrack(rest, new_parts, best)
            if result < best:
                best = result
    
    # Try putting v in a new clique part
    new_parts = parts + [('clique', [v])]
    result = backtrack(rest, new_parts, best)
    if result < best:
        best = result
    
    # Try putting v in a new independent part
    new_parts = parts + [('independent', [v])]
    result = backtrack(rest, new_parts, best)
    if result < best:
        best = result
    
    return best

def construct():
    # Try to find the minimum cochromatic partition using backtracking
    # Start with upper bound from a greedy heuristic
    # First, get a greedy solution to have an initial upper bound
    greedy_parts = []
    for v in range(N):
        placed = False
        for i, (ptype, pverts) in enumerate(greedy_parts):
            if ptype == 'clique' and can_add_to_clique(pverts, v):
                greedy_parts[i] = ('clique', pverts + [v])
                placed = True
                break
            elif ptype == 'independent' and can_add_to_independent(pverts, v):
                greedy_parts[i] = ('independent', pverts + [v])
                placed = True
                break
        if not placed:
            greedy_parts.append(('clique', [v]))
    
    initial_best = len(greedy_parts)
    
    # Now use backtracking to find the optimal solution
    # We'll try to improve upon the greedy solution
    best = initial_best
    # Start with empty parts and all vertices remaining
    result = backtrack(tuple(range(N)), [], best)
    
    # Reconstruct the partition from the best solution found
    # We need to run the search again to get the actual partition
    # Let's modify backtrack to also return the partition
    
    # Actually, let's just use the result and reconstruct
    # For now, let's do a simpler approach: run backtrack to find the min number,
    # then do another search to find a valid partition with that many parts
    
    min_parts = result
    
    # Now find a valid partition with min_parts parts
    def find_partition(remaining, parts, target):
        """Find a valid partition with exactly target parts."""
        if not remaining:
            if len(parts) == target:
                return parts
            return None
        
        if len(parts) >= target:
            return None
        
        v = remaining[0]
        rest = remaining[1:]
        
        # Try adding v to each existing part
        for i, (ptype, pverts) in enumerate(parts):
            if ptype == 'clique' and can_add_to_clique(pverts, v):
                new_parts = parts.copy()
                new_parts[i] = ('clique', pverts + [v])
                result = find_partition(rest, new_parts, target)
                if result is not None:
                    return result
            elif ptype == 'independent' and can_add_to_independent(pverts, v):
                new_parts = parts.copy()
                new_parts[i] = ('independent', pverts + [v])
                result = find_partition(rest, new_parts, target)
                if result is not None:
                    return result
        
        # Try putting v in a new part (if we still have capacity)
        if len(parts) < target:
            # Try new clique
            new_parts = parts + [('clique', [v])]
            result = find_partition(rest, new_parts, target)
            if result is not None:
                return result
            
            # Try new independent
            new_parts = parts + [('independent', [v])]
            result = find_partition(rest, new_parts, target)
            if result is not None:
                return result
        
        return None
    
    partition = find_partition(tuple(range(N)), [], min_parts)
    
    if partition is None:
        # Fallback to greedy if something went wrong
        partition = greedy_parts
    
    # Convert to list of lists (vertices only)
    result_partition = [list(pverts) for (_, pverts) in partition]
    
    # Ensure all vertices are present exactly once
    all_vertices = []
    for part in result_partition:
        all_vertices.extend(part)
    assert sorted(all_vertices) == list(range(N)), "Invalid partition"
    
    return result_partition

# EVOLVE-BLOCK-END
