"""Rewritten program: exact cochromatic partition via branch-and-bound."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_clique(vertices, adj):
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            u, v = vertices[i], vertices[j]
            if not (adj[u] & (1 << v)):
                return False
    return True

def _is_independent(vertices, adj):
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            u, v = vertices[i], vertices[j]
            if adj[u] & (1 << v):
                return False
    return True

def _find_maximal_cliques(adj, n):
    """Bron-Kerbosch with pivot to find all maximal cliques."""
    cliques = []
    
    def bronk(R, P_set, X):
        if not P_set and not X:
            if R:
                cliques.append(frozenset(R))
            return
        # Choose pivot
        pivot = None
        candidates = P_set | X
        for v in candidates:
            if not (adj[v] & (1 << pivot)) if pivot is not None else True:
                pivot = v
                break
        # Better pivot selection: pick vertex with max degree in P
        best_pivot = None
        best_count = -1
        for v in (P_set | X):
            count = bin(adj[v] & P_set).count('1')
            if count > best_count:
                best_count = count
                best_pivot = v
        pivot = best_pivot
        pivot_neighbors = adj[pivot] if pivot is not None else 0
        
        remaining = P_set & ~pivot_neighbors
        v_list = list(remaining)
        for v in v_list:
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set -= {v}
            X |= {v}
    
    bronk(set(), set(range(n)), set())
    return cliques

def _find_maximal_independent_sets(adj, n):
    """Find maximal independent sets by finding maximal cliques in complement."""
    comp_adj = [0] * n
    for u in range(n):
        for v in range(n):
            if u != v and not (adj[u] & (1 << v)):
                comp_adj[u] |= (1 << v)
    cliques = _find_maximal_cliques(comp_adj, n)
    return cliques

def _all_valid_parts(adj, n):
    """Find all cliques and independent sets (not just maximal) as bitsets."""
    valid = set()
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            vs = list(combo)
            if _is_clique(vs, adj) or _is_independent(vs, adj):
                mask = 0
                for v in vs:
                    mask |= (1 << v)
                valid.add(mask)
    return valid

def _can_cover(parts_masks, target_mask, adj, n):
    """Check if target_mask can be covered by up to len(parts_masks) parts."""
    # This is a set cover problem: can we pick subsets from parts_masks 
    # that together cover target_mask, where each part is a clique or independent set?
    # Use DP over subsets
    # parts_masks is a list of bitsets representing valid parts
    # We need to select a collection of parts whose union covers target_mask
    
    # For small n, use recursive search with memoization
    from functools import lru_cache
    
    # Sort parts by size descending for better pruning
    sorted_parts = sorted(parts_masks, key=lambda x: bin(x).count('1'), reverse=True)
    
    best = [None]
    
    def search(remaining, parts_list, current_parts):
        if not remaining:
            if best[0] is None or len(current_parts) < len(best[0]):
                best[0] = list(current_parts)
            return True
        if not parts_list:
            return False
        # Try using each part
        for i, part in enumerate(parts_list):
            # Part must be subset of remaining (or at least intersect)
            # Actually part can overlap, but we just need union to cover remaining
            # Let's think differently: we want to assign each vertex in remaining to a part
            # that contains it and is a valid clique/independent set
            
            # Alternative: for each vertex in remaining, it must be in some part
            # Let's use a different approach: try to cover remaining with parts
            # where each part is a subset of remaining (since if a part has vertices
            # outside remaining, those are already covered)
            
            # Actually, parts can have vertices outside 'remaining' - that's fine
            # We just need the union of selected parts to include 'remaining'
            
            # Let me reconsider: we want to select a set of parts (from valid parts)
            # such that their union covers target_mask
            # Each part is a clique or independent set
            
            # For the search: at each step, pick a vertex in remaining that's not yet
            # covered, and try to cover it with a part that contains it
            pass
        return False
    
    # Simpler approach: use DP over subsets
    # dp[mask] = minimum number of parts needed to cover 'mask'
    # But this might be too slow for n=14 with many valid parts
    
    # Let's use a different strategy: iterative deepening
    # Try k=1,2,3,... and for each k, check if k parts can cover all vertices
    
    return None

def construct():
    deadline = time.time() + 85  # Leave 5s safety margin
    n = P['n']
    adj = _build_graph()
    
    # Find all maximal cliques and independent sets
    max_cliques = _find_maximal_cliques(adj, n)
    max_indep = _find_maximal_independent_sets(adj, n)
    
    # All valid parts are subsets of maximal cliques or maximal independent sets
    # But we need all cliques and independent sets, not just maximal ones
    # However, for covering, we can restrict to maximal ones since any non-maximal
    # clique/independent set is a subset of a maximal one
    
    # Actually, we need to be careful: a part in the partition must be a clique or
    # independent set, but it doesn't have to be maximal. However, if we use a
    # non-maximal clique, we could always extend it to a maximal clique (if the
    # extension doesn't conflict with other parts). But in a partition, each vertex
    # is in exactly one part, so we can't extend a part.
    
    # Let's just enumerate all valid parts (cliques and independent sets)
    # For n=14, this is feasible
    
    valid_parts = []
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            vs = list(combo)
            if _is_clique(vs, adj) or _is_independent(vs, adj):
                mask = 0
                for v in vs:
                    mask |= (1 << v)
                valid_parts.append((mask, vs))
    
    # Sort by size descending
    valid_parts.sort(key=lambda x: bin(x[0]).count('1'), reverse=True)
    
    # Now search for minimum k such that k parts cover all n vertices
    # Use iterative deepening: try k=1,2,3,...
    
    all_mask = (1 << n) - 1
    
    def search_cover(k, remaining, used_parts, current_solution):
        """Try to cover 'remaining' with k parts from valid_parts."""
        if time.time() > deadline:
            return None
        if not remaining:
            if k <= 0:
                return list(current_solution)
            else:
                # We have extra parts but remaining is empty - that's fine
                # but we want exactly k parts, so this is valid only if we use exactly k
                # Actually, we want at most k parts, so if remaining is empty, we're done
                return list(current_solution)
        if k <= 0:
            return None
        
        # Pick a vertex in remaining to cover
        # Choose the vertex with fewest candidate parts (most constrained)
        remaining_vertices = [v for v in range(n) if remaining & (1 << v)]
        
        # For each vertex, find parts that contain it and are subset of remaining
        # (or at least intersect remaining)
        # Actually, a part can have vertices outside 'remaining' - those are already covered
        # So we just need the part to contain at least one vertex in remaining
        # And the part's intersection with remaining should help cover remaining
        
        # Better: pick the vertex in remaining with the fewest valid parts that contain it
        # and whose intersection with remaining is non-empty
        best_vertex = None
        best_count = float('inf')
        best_parts_for_vertex = []
        
        for v in remaining_vertices:
            count = 0
            parts_for_v = []
            for mask, vs in valid_parts:
                if mask & (1 << v):
                    # Part contains v
                    # The part can be used if it's a valid clique/independent set
                    # (which it is by construction)
                    # We just need it to help cover remaining
                    # Any part containing v is fine since v is in remaining
                    count += 1
                    parts_for_v.append((mask, vs))
                    if count >= best_count:
                        break
            if count < best_count:
                best_count = count
                best_vertex = v
                best_parts_for_vertex = parts_for_v[:best_count]
        
        if best_vertex is None:
            return None
        
        # Try each part that contains best_vertex
        for mask, vs in best_parts_for_vertex:
            # Use this part: it covers mask & remaining
            new_remaining = remaining & ~mask
            new_k = k - 1
            result = search_cover(new_k, new_remaining, used_parts, current_solution + [vs])
            if result is not None:
                return result
        
        return None
    
    # Iterative deepening
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        result = search_cover(k, all_mask, set(), [])
        if result is not None:
            # Convert to proper format
            parts = [list(vs) for vs in result]
            # Verify
            seen = set()
            for part in parts:
                for v in part:
                    if v in seen:
                        break
                    seen.add(v)
                else:
                    continue
                break
            else:
                if seen == set(range(n)):
                    # Verify each part is clique or independent
                    valid = True
                    for part in parts:
                        if not (_is_clique(part, adj) or _is_independent(part, adj)):
                            valid = False
                            break
                    if valid:
                        return parts
    
    # Fallback: greedy first-fit
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in set(map(tuple, P['edges'])) for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts
# EVOLVE-BLOCK-END
