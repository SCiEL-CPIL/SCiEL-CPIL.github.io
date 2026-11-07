"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
EDGE_SET = set()
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    EDGE_SET.add((u, v))
    EDGE_SET.add((v, u))
    ADJ[u].add(v)
    ADJ[v].add(u)


def bron_kerbosch_max_cliques():
    """Find all maximal cliques."""
    cliques = []
    
    def bk(R, P_set, X):
        if not P_set and not X:
            cliques.append(set(R))
            return
        # Pivot selection: choose vertex in P_set ∪ X with max |P_set ∩ N(v)|
        candidates = P_set | X
        if candidates:
            pivot = max(candidates, key=lambda v: len(P_set & ADJ[v]))
            candidates = P_set - ADJ[pivot]
        else:
            candidates = set()
        for v in list(candidates):
            bk(R | {v}, P_set & ADJ[v], X & ADJ[v])
            P_set.discard(v)
            X.add(v)
    
    bk(set(), set(range(N)), set())
    return cliques


def find_disjoint_paths(branches, pair_order, forbidden, deadline):
    """
    Try to find vertex-disjoint paths for each pair in pair_order.
    branches: list of branch vertices
    pair_order: list of (u,v) pairs to connect
    forbidden: set of vertices that cannot be used as path interiors
    Returns dict mapping (u,v) -> path (list), or None on failure.
    """
    used_interiors = set(forbidden)
    paths = {}
    
    for (u, v) in pair_order:
        if time.time() > deadline:
            return None
        # BFS for shortest path from u to v avoiding used_interiors (except u,v themselves)
        # We need path where interiors avoid used_interiors and all branches
        all_branches_set = set(branches)
        
        # BFS
        from collections import deque
        parent = {}
        visited = set()
        queue = deque()
        queue.append(u)
        visited.add(u)
        parent[u] = None
        
        found = False
        while queue:
            cur = queue.popleft()
            if cur == v:
                found = True
                break
            for nb in sorted(ADJ[cur]):
                if nb in visited:
                    continue
                # nb can be v (endpoint) or must not be in used_interiors or branches (except u,v)
                if nb == v:
                    pass
                elif nb in all_branches_set:
                    continue
                elif nb in used_interiors:
                    continue
                else:
                    pass
                visited.add(nb)
                parent[nb] = cur
                queue.append(nb)
        
        if not found:
            return None
        
        # Reconstruct path
        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        
        # Verify: consecutive vertices must be adjacent
        for i in range(len(path) - 1):
            if path[i+1] not in ADJ[path[i]]:
                return None
        
        # Check interiors don't conflict
        for mid in path[1:-1]:
            if mid in used_interiors:
                return None
            if mid in all_branches_set:
                return None
        
        # Add interiors to used
        for mid in path[1:-1]:
            used_interiors.add(mid)
        
        paths[(u, v)] = path
    
    return paths


def try_clique_subdivision(branches, deadline, rng):
    """
    Try to find disjoint paths for all pairs in the branch set.
    Uses randomized pair ordering with multiple restarts.
    Returns (paths_dict, branch_list) or None.
    """
    bset = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    
    if not pairs:
        return None
    
    # Forbidden: all branch vertices (interiors can't be branches)
    forbidden = set(branches)
    
    best = None
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        # Randomize pair order
        shuffled_pairs = pairs[:]
        rng.shuffle(shuffled_pairs)
        
        result = find_disjoint_paths(branches, shuffled_pairs, forbidden, deadline)
        if result is not None:
            best = result
            break
    
    if best is None:
        return None
    
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 75  # safety margin
    
    # Build adjacency
    # Already done at module level
    
    # Find maximal cliques
    cliques = bron_kerbosch_max_cliques()
    
    # Sort by size descending
    cliques.sort(key=len, reverse=True)
    
    rng = random.Random(42)
    
    best_result = None
    best_size = 0
    
    # Try each maximal clique
    for clique in cliques:
        if time.time() > deadline:
            break
        
        csize = len(clique)
        if csize < 2:
            continue
        
        # If we already have a result of this size or larger, skip smaller
        if csize <= best_size:
            continue
        
        branch_list = sorted(clique)
        result = try_clique_subdivision(branch_list, deadline, rng)
        
        if result is not None:
            # Build the output
            paths_out = []
            branches_out = list(branch_list)
            # Need paths for every unordered pair
            for (u, v) in combinations(branch_list, 2):
                paths_out.append(list(result[(u, v)]))
            
            best_result = {
                'branches': branches_out,
                'paths': paths_out
            }
            best_size = csize
    
    # Also try non-clique branch sets? For now, cliques should suffice for subdivision
    # A subdivision of K_k requires k branch vertices with disjoint paths between all pairs.
    # The branch vertices don't need to form a clique in the original graph,
    # but finding them as a clique is a good heuristic.
    
    if best_result is None:
        # Fallback: just an edge
        u, v = P['edges'][0]
        best_result = {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }
    
    # Ensure types are correct
    best_result['branches'] = [int(x) for x in best_result['branches']]
    best_result['paths'] = [[int(x) for x in p] for p in best_result['paths']]
    
    return best_result
# EVOLVE-BLOCK-END
