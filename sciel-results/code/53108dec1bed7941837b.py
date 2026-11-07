"""Packing trees T1..T20 into K20 - greedy randomized backtracking search."""
import time
from random import Random, shuffle

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    DEADLINE = time.time() + 140.0  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    N = 20
    all_edges = set()
    for u in range(N):
        for v in range(u + 1, N):
            all_edges.add(frozenset((u, v)))
    
    # Build adjacency for K20
    adj = [set() for _ in range(N)]
    for u in range(N):
        for v in range(N):
            if u != v:
                adj[u].add(v)
    
    # We'll embed trees in order: largest first (T20, T19, ..., T1)
    # indices: 19, 18, ..., 0
    tree_order = list(range(19, -1, -1))
    
    # Tree adjacencies
    tree_adj = []
    for k in range(N):
        t_edges = trees[k]
        t_adj = [set() for _ in range(k)]
        for u, v in t_edges:
            t_adj[u].add(v)
            t_adj[v].add(u)
        tree_adj.append(t_adj)
    
    def try_embed(tree_idx, used_edges, rng):
        """Try to embed tree tree_idx into remaining graph. Returns mapping or None."""
        k = tree_idx + 1
        if k == 1:
            # Single vertex, no edges
            # Find any available vertex
            for v in range(N):
                return [v]
            return None
        if k == 2:
            # Two vertices, one edge
            for u in range(N):
                for v in range(u + 1, N):
                    e = frozenset((u, v))
                    if e not in used_edges:
                        return [u, v]
            return None
        
        t_adj_local = tree_adj[tree_idx]
        
        # Root at vertex 0 of the tree
        # BFS order for the tree
        parent = [-1] * k
        order = [0]
        parent[0] = -2
        for node in order:
            for nb in t_adj_local[node]:
                if parent[nb] == -1:
                    parent[nb] = node
                    order.append(nb)
        
        # Recursive backtracking
        mapping = [-1] * k
        mapping[0] = -1  # will be assigned
        
        def backtrack(pos):
            if time.time() > DEADLINE:
                return False
            if pos == len(order):
                return True
            tv = order[pos]
            if pos == 0:
                # Root: try all vertices
                candidates = list(range(N))
                rng.shuffle(candidates)
            else:
                pv = parent[tv]
                ph = mapping[pv]
                candidates = list(adj[ph])
                # Filter: host must not already be used by another tree vertex
                used_hosts = set()
                for i in range(k):
                    if mapping[i] != -1 and i != tv:
                        used_hosts.add(mapping[i])
                candidates = [c for c in candidates if c not in used_hosts]
                # Also check edge availability
                candidates = [c for c in candidates if frozenset((ph, c)) not in used_edges]
                rng.shuffle(candidates)
            
            for hv in candidates:
                if pos > 0:
                    e = frozenset((mapping[parent[tv]], hv))
                    if e in used_edges:
                        continue
                    # Add edge temporarily
                    used_edges.add(e)
                    mapping[tv] = hv
                    if backtrack(pos + 1):
                        return True
                    used_edges.discard(e)
                else:
                    mapping[tv] = hv
                    if backtrack(pos + 1):
                        return True
                mapping[tv] = -1
            
            return False
        
        if backtrack(0):
            return mapping
        return None
    
    # Try multiple random restarts
    for attempt in range(5000):
        if time.time() > DEADLINE - 5:
            break
        rng = Random(42 + attempt)
        used_edges = set()
        result = [None] * N
        success = True
        
        for tree_idx in tree_order:
            if time.time() > DEADLINE - 5:
                success = False
                break
            mapping = try_embed(tree_idx, used_edges, rng)
            if mapping is None:
                success = False
                break
            result[tree_idx] = mapping
            # Mark edges as used
            for u, v in trees[tree_idx]:
                used_edges.add(frozenset((mapping[u], mapping[v])))
        
        if success:
            # Verify
            for k in range(N):
                if result[k] is None:
                    continue
                if len(result[k]) != k:
                    success = False
                    break
                if len(set(result[k])) != k:
                    success = False
                    break
                for v in result[k]:
                    if not (0 <= v < N) or not isinstance(v, int):
                        success = False
                        break
            if success:
                # Count edges
                total = 0
                for k in range(N):
                    total += len(trees[k])
                # Ensure no edge conflicts
                edge_set = set()
                for k in range(N):
                    if result[k] is None:
                        continue
                    for u, v in trees[k]:
                        e = frozenset((result[k][u], result[k][v]))
                        if e in edge_set:
                            success = False
                            break
                        edge_set.add(e)
                    if not success:
                        break
            
            if success:
                # Convert to proper int lists
                final = []
                for k in range(N):
                    if result[k] is None:
                        final.append(None)
                    else:
                        final.append([int(x) for x in result[k]])
                return final
    
    # Fallback: return just T20
    return [None] * 19 + [list(range(20))]
