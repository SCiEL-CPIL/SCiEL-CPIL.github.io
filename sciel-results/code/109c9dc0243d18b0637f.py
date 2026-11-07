import time
import random
import sys
from collections import deque

sys.setrecursionlimit(100000)

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = start_time + 145
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_v = 20
    total_edges = sum(len(t) for t in trees)  # should be 190
    
    # Precompute tree data
    tree_adj = []
    tree_edges = []
    tree_deg = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_edges.append(edges)
        tree_deg.append(deg)
    
    # Pack largest trees first
    pack_order = sorted(range(n_trees), key=lambda i: -(i + 1))
    
    # Edge bitmask: 190 edges in K20. Edge (i,j) with i<j has index = i*19 + j - i*(i+1)//2
    # Actually let's use a simpler mapping: edge index for (i,j) where i<j
    edge_index = {}
    idx = 0
    for i in range(n_v):
        for j in range(i + 1, n_v):
            edge_index[(i, j)] = idx
            idx += 1
    total_e = idx  # 190
    
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return edge_index[(u, v)]
    
    best_score = 0
    best_mappings = [None] * n_trees
    
    def try_full_packing(seed):
        nonlocal best_score, best_mappings
        rng = random.Random(seed)
        
        # used_edges as a set of edge indices
        used_edges = set()
        mappings = [None] * n_trees
        # For each tree, track which host vertices are used
        host_used = [set() for _ in range(n_trees)]  # not quite right; we need global host usage per tree
        
        # Actually, each tree uses distinct host vertices, but different trees can share host vertices
        # The constraint is only on edges: no two trees can use the same edge of K20
        # And within a tree, all host vertices must be distinct
        
        # Let's use: for each tree, a mapping from tree_vertex -> host_vertex
        # Constraint: for each tree, all host vertices distinct
        # For each edge in tree, the corresponding K20 edge not in used_edges
        
        def pack_tree(t_idx, mapping, used_hosts):
            """Try to embed tree t_idx given partial mapping and used_hosts set"""
            k = t_idx + 1
            adj = tree_adj[t_idx]
            edges = tree_edges[t_idx]
            
            if k == 1:
                # Just pick any unused host
                for hv in range(n_v):
                    if hv not in used_hosts:
                        mapping[0] = hv
                        return True
                return False
            
            # Find unassigned tree vertex with fewest valid candidates
            best_tv = -1
            best_cands = None
            min_cnt = float('inf')
            
            for tv in range(k):
                if mapping[tv] != -1:
                    continue
                cands = []
                for hv in range(n_v):
                    if hv in used_hosts:
                        continue
                    # Check edges to already-mapped neighbors
                    valid = True
                    for nb in adj[tv]:
                        if mapping[nb] != -1:
                            hn = mapping[nb]
                            ek = edge_key(hv, hn)
                            if ek in used_edges:
                                valid = False
                                break
                    if valid:
                        cands.append(hv)
                if not cands:
                    return False
                if len(cands) < min_cnt:
                    min_cnt = len(cands)
                    best_tv = tv
                    best_cands = cands
                    if min_cnt == 1:
                        break
            
            if best_tv == -1:
                # All assigned; verify
                for u, v in edges:
                    ek = edge_key(mapping[u], mapping[v])
                    if ek in used_edges:
                        return False
                return True
            
            rng.shuffle(best_cands)
            
            for hv in best_cands:
                mapping[best_tv] = hv
                used_hosts.add(hv)
                
                # Check if this creates a conflict with already-mapped neighbors
                ok = True
                for nb in adj[best_tv]:
                    if mapping[nb] != -1:
                        hn = mapping[nb]
                        ek = edge_key(hv, hn)
                        if ek in used_edges:
                            ok = False
                            break
                if ok:
                    if pack_tree(t_idx, mapping, used_hosts):
                        return True
                
                used_hosts.remove(hv)
                mapping[best_tv] = -1
            
            return False
        
        # Pack trees in order
        for t_idx in pack_order:
            if time.time() > deadline:
                break
            
            k = t_idx + 1
            mapping = [-1] * k
            used_hosts = set()
            
            # Try multiple random shuffles for this tree
            success = False
            for attempt in range(50):
                mapping = [-1] * k
                used_hosts = set()
                if pack_tree(t_idx, mapping, used_hosts):
                    success = True
                    break
            
            if not success:
                # Try with different global edge set? No, we can't change used_edges easily
                # Just skip this tree
                continue
            
            mappings[t_idx] = mapping[:]
            for u, v in tree_edges[t_idx]:
                ek = edge_key(mapping[u], mapping[v])
                used_edges.add(ek)
        
        # Score
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(tree_edges[i])
        
        if score > best_score:
            best_score = score
            best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]
        
        return best_score == total_e
    
    # Try many seeds
    seed = 0
    while time.time() < deadline:
        if try_full_packing(seed):
            break
        seed += 1
    
    result = []
    for i in range(n_trees):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])
    
    return result
