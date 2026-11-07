import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    trees = instance_trees(20, 611)
    N = 20
    
    # Pre-calculate tree structures
    tree_structures = []
    for idx, edges in enumerate(trees):
        k = idx + 1
        if k == 1:
            tree_structures.append({
                'k': k,
                'edges': [],
                'adj': [[]],
                'deg': [0],
                'var_order': [0]
            })
            continue
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort vertices by degree descending for MRV
        var_order = sorted(range(k), key=lambda v: -deg[v])
        
        tree_structures.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'var_order': var_order
        })

    def calculate_host_remaining_deg(used_edges):
        """
        Calculate the number of available edges for each host vertex.
        A host vertex h has N-1 potential edges. 
        remaining_deg[h] = (N - 1) - (number of used edges incident to h)
        """
        host_deg = [0] * N
        for u, v in used_edges:
            host_deg[u] += 1
            host_deg[v] += 1
        return [N - 1 - d for d in host_deg]

    def embed_tree(tree_idx, used_edges, rng, node_limit=200000):
        """
        Try to embed tree_idx into the graph defined by N=20 and used_edges.
        Returns a list of length k with host vertices, or None if failed.
        """
        struct = tree_structures[tree_idx]
        k = struct['k']
        adj = struct['adj']
        var_order = struct['var_order']
        
        if k == 1:
            # Any vertex works, pick 0 for simplicity (or random)
            # Since vertices can be shared, this doesn't consume edges.
            return [0]
            
        mapping = [-1] * k
        used_hosts = set()
        nodes_explored = [0]
        
        # Pre-calculate remaining degrees of hosts to guide candidate ordering
        host_rem_deg = calculate_host_remaining_deg(used_edges)
        
        def backtrack(idx):
            if nodes_explored[0] > node_limit:
                return False
            if idx == k:
                return True
            
            v = var_order[idx]
            cands = []
            for h in range(N):
                if h in used_hosts:
                    continue
                # Check if edges to already mapped neighbors are available
                ok = True
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        edge = (min(h, mw), max(h, mw))
                        if edge in used_edges:
                            ok = False
                            break
                if ok:
                    cands.append(h)
            
            if not cands:
                return False
            
            # Heuristic: Sort candidates by remaining degree in host graph (descending)
            # This tends to pack "dense" parts of the tree into "dense" parts of the host
            # leaving "sparse" parts for later, or vice versa?
            # Actually, if we map high-degree tree nodes to high-degree host nodes, we use up the most connected hosts early.
            # This is generally good for packing.
            
            # Add a bit of randomness to break ties or escape local minima if stuck
            # We sort by (-host_rem_deg[h], rng.random()) to add jitter
            cands.sort(key=lambda h: (-host_rem_deg[h], rng.random()))
            
            for h in cands:
                nodes_explored[0] += 1
                if nodes_explored[0] > node_limit:
                    return False
                
                mapping[v] = h
                used_hosts.add(h)
                
                # Record added edges to undo later
                added_edges = []
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        e = (min(h, mw), max(h, mw))
                        used_edges.add(e)
                        added_edges.append(e)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                for e in added_edges:
                    used_edges.discard(e)
                used_hosts.discard(h)
                mapping[v] = -1
            
            return False
        
        if backtrack(0):
            return [mapping[i] for i in range(k)]
        return None

    best_result = None
    best_edges = -1
    
    # Process trees in order of decreasing size (most constrained first)
    # T20 (19 edges) down to T1 (0 edges)
    order = list(range(19, -1, -1))
    
    # We will try a few random restarts with different random seeds for the internal heuristics
    # If the first deterministic-like run fails, randomness might help.
    
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
            
        rng = Random(611 + attempt)
        used_edges = set()
        embedding = [None] * N
        total_edges = 0
        success = True
        
        for idx in order:
            if time.time() > deadline - 5:
                success = False
                break
                
            k = idx + 1
            if k == 1:
                embedding[idx] = [0]
                continue
                
            # For T2 (1 edge), it's very easy, but we use the same solver for consistency
            # The solver handles k=2 fine.
            
            m = embed_tree(idx, used_edges, rng)
            if m is None:
                success = False
                break
            
            embedding[idx] = m
            # Add edges to used_edges
            tree_edges = trees[idx]
            for u, v in tree_edges:
                host_u = m[u]
                host_v = m[v]
                e = (min(host_u, host_v), max(host_u, host_v))
                used_edges.add(e)
            total_edges += len(tree_edges)
            
        if success and total_edges > best_edges:
            best_edges = total_edges
            best_result = embedding
            if best_edges == 190:
                break

    if best_result is not None:
        # Ensure types are correct: list of ints or None
        return [list(e) if e is not None else None for e in best_result]
    
    # Fallback if nothing worked (should not happen for T20 alone)
    return [None] * 19 + [list(range(20))]

# Test harness / main entry point
if __name__ == "__main__":
    result = construct()
    print(result)
