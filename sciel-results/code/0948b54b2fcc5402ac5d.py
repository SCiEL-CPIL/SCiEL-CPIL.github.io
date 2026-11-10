"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                a, b = vertices[i], vertices[j]
                if (min(a, b), max(a, b)) not in edge_set:
                    return False
        return True
    
    # Find maximum clique using Bron-Kerbosch with pivot
    best_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        # Choose pivot to minimize branching
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
        else:
            pivot = -1
        candidates = P_set - adj[pivot] if pivot >= 0 else set(P_set)
        
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    
    # Try to find larger branch sets using disjoint paths
    # For a set of k branches, we need k*(k-1)/2 pairwise disjoint paths
    # with disjoint interiors and avoiding all branches
    
    def find_disjoint_paths(branches, forbidden):
        """Find pairwise disjoint paths between all pairs of branches."""
        pairs = list(combinations(branches, 2))
        paths = {}
        used_interiors = set(forbidden)
        
        for a, b in pairs:
            # BFS to find shortest path avoiding used_interiors
            if a == b:
                continue
            if a in used_interiors or b in used_interiors:
                return None
            # BFS from a to b
            parent = {a: None}
            queue = deque([a])
            found = False
            while queue:
                cur = queue.popleft()
                if cur == b:
                    found = True
                    break
                for nb in adj[cur]:
                    if nb not in parent and nb not in used_interiors:
                        parent[nb] = cur
                        queue.append(nb)
            if not found:
                return None
            # Reconstruct path
            path = []
            cur = b
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            # Check that path is valid (no repeats except endpoints)
            interiors = path[1:-1]
            if len(set(interiors)) != len(interiors):
                return None
            # Add interiors to used
            for v in interiors:
                if v in used_interiors:
                    return None
                used_interiors.add(v)
            paths[(a, b)] = path
        
        return paths
    
    # Try clique first
    if len(best_clique) >= 2:
        branches = list(best_clique)
        paths = {}
        for a, b in combinations(branches, 2):
            paths[(a, b)] = [a, b]
        # This is valid since all pairs are adjacent
        result = {
            'branches': [int(v) for v in branches],
            'paths': [[int(v) for v in paths[(a, b)]] for a, b in combinations(branches, 2)]
        }
        best_result = result
        best_count = len(branches)
    else:
        best_result = None
        best_count = 0
    
    # Try to find larger branch sets by searching over subsets
    # Use the clique as a starting point and try adding vertices
    # Or try random subsets
    
    import random
    random.seed(42)
    
    # Try all subsets of size up to some limit
    # Start with the clique and try to add vertices
    current_branches = set(best_clique) if best_clique else set()
    
    # Try to add vertices one at a time
    for attempt in range(50):
        if time.time() > deadline:
            break
        # Try adding a random vertex to the current set
        candidates = [v for v in range(n) if v not in current_branches]
        if not candidates:
            break
        v = random.choice(candidates)
        new_branches = current_branches | {v}
        branches_list = sorted(new_branches)
        paths = find_disjoint_paths(branches_list, set(branches_list))
        if paths is not None:
            if len(branches_list) > best_count:
                best_count = len(branches_list)
                result = {
                    'branches': [int(v) for v in branches_list],
                    'paths': [[int(v) for v in paths[(a, b)]] for a, b in combinations(branches_list, 2)]
                }
                best_result = result
                current_branches = new_branches
    
    # Try greedy: start with clique, try to add each remaining vertex
    if best_clique:
        current = set(best_clique)
        for v in range(n):
            if time.time() > deadline:
                break
            if v in current:
                continue
            new_set = current | {v}
            branches_list = sorted(new_set)
            paths = find_disjoint_paths(branches_list, set(branches_list))
            if paths is not None:
                current = new_set
                if len(current) > best_count:
                    best_count = len(current)
                    result = {
                        'branches': [int(v) for v in sorted(current)],
                        'paths': [[int(v) for v in paths[(a, b)]] for a, b in combinations(sorted(current), 2)]
                    }
                    best_result = result
    
    if best_result is None:
        # Fallback: just use an edge
        u, v = edges[0]
        best_result = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    return best_result
# EVOLVE-BLOCK-END
