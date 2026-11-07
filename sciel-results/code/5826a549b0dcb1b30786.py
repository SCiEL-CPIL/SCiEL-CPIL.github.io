"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

# Precompute adjacency
_n = P['n']
_adj = [set() for _ in range(_n)]
for a, b in P['edges']:
    _adj[a].add(b)
    _adj[b].add(a)

def _find_clique_subdivision(branches):
    """
    Given a set of branch vertices, try to find vertex-disjoint paths between
    every pair. Returns a dict with 'branches' and 'paths' or None.
    """
    if len(branches) < 2:
        return None
    
    branches_set = set(branches)
    n = _n
    branch_list = list(branches)
    pairs = list(itertools.combinations(branch_list, 2))
    
    # For each pair, we need a path. Internal vertices must be disjoint across all paths
    # and cannot be branch vertices.
    
    # We'll use a greedy approach: for each pair, find a shortest path avoiding
    # already-used internal vertices.
    # Order pairs by "difficulty" - pairs with fewer available paths first.
    
    # Actually, let's try all permutations of pair orderings? That's too expensive.
    # Instead, let's use a smarter approach: try to find paths in order of
    # increasing distance (shortest first), which tends to work better.
    
    # Precompute distances between all pairs of branches avoiding other branches
    # as internal nodes.
    
    def bfs_shortest_path(src, dst, blocked):
        """Find shortest path from src to dst avoiding blocked internal vertices."""
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for nb in _adj[node]:
                if nb == dst:
                    return path + [nb]
                if nb in blocked or nb in branches_set:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None
    
    # Try different orderings of pairs
    # Sort pairs by BFS distance (shortest first)
    def pair_distance(p):
        blocked = branches_set - {p[0], p[1]}
        path = bfs_shortest_path(p[0], p[1], blocked)
        return len(path) if path else float('inf')
    
    # We'll try a few strategies for ordering
    best_result = None
    
    # Strategy 1: shortest first
    sorted_pairs = sorted(pairs, key=pair_distance)
    
    for pair_order in [sorted_pairs, list(reversed(sorted_pairs)), pairs]:
        used_internal = set()
        paths = {}
        success = True
        
        for a, b in pair_order:
            blocked = used_internal | (branches_set - {a, b})
            path = bfs_shortest_path(a, b, blocked)
            if path is None:
                success = False
                break
            # Mark internal vertices as used
            for v in path[1:-1]:
                used_internal.add(v)
            paths[(a, b)] = path
        
        if success:
            # Verify all constraints
            all_internal = set()
            for (a, b), path in paths.items():
                for v in path[1:-1]:
                    if v in all_internal:
                        success = False
                        break
                    all_internal.add(v)
                if v in branches_set:
                    success = False
                    break
            if not success:
                continue
            
            # Build result
            result_paths = []
            for a, b in pairs:
                result_paths.append(paths[(a, b)])
            
            result = {
                'branches': list(branches),
                'paths': result_paths
            }
            
            if best_result is None or len(result['branches']) > len(best_result['branches']):
                best_result = result
            
            if best_result is not None and len(best_result['branches']) >= len(branches):
                # Found a valid solution for this branch set
                return best_result
    
    return None


def _bron_kerbosch(adj, n):
    """Find all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            cliques.append(list(R))
            return
        
        # Choose pivot with maximum degree in P
        if P:
            pivot = max(P, key=lambda v: len(adj[v] & P))
            candidates = P - adj[pivot]
        else:
            candidates = P
        
        for v in list(candidates):
            bk(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}
    
    bk(set(), set(range(n)), set())
    return cliques


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    n = _n
    adj_sets = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj_sets[a].add(b)
        adj_sets[b].add(a)
    
    best = None
    
    # Find all maximal cliques
    cliques = _bron_kerbosch(adj_sets, n)
    
    # Sort cliques by size descending
    cliques.sort(key=len, reverse=True)
    
    for clique in cliques:
        if time.time() > deadline:
            break
        
        if len(clique) < 2:
            continue
        
        # Check if this clique size is potentially better
        if best is not None and len(clique) <= len(best['branches']):
            continue
        
        result = _find_clique_subdivision(clique)
        if result is not None:
            best = result
    
    # Also try to extend: for each found clique, try adding vertices
    if best is not None:
        best_branches = set(best['branches'])
        # Try to add one more vertex to the clique
        for v in range(n):
            if time.time() > deadline:
                break
            if v in best_branches:
                continue
            # Check if v is adjacent to all current branches
            if all(v in adj_sets[b] for b in best_branches):
                new_clique = best_branches | {v}
                result = _find_clique_subdivision(new_clique)
                if result is not None:
                    best = result
                    best_branches = set(best['branches'])
    
    if best is None:
        # Fallback: just use an edge
        u, v = P['edges'][0]
        best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Ensure types are correct
    best['branches'] = [int(x) for x in best['branches']]
    best['paths'] = [[int(x) for x in path] for path in best['paths']]
    
    return best
# EVOLVE-BLOCK-END
