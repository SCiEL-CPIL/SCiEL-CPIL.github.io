"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def instance_trees(n=20, seed=611):
    from random import Random
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
    deadline = start_time + 140.0  # Safety margin
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency for each tree
    tree_adjs = []
    for k in range(1, n + 1):
        adj = defaultdict(list)
        for u, v in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adjs.append(adj)
        
    result = [None] * n
    used_edges = set()
    
    # Process trees in reverse order to prioritize larger trees
    # T20 has 19 edges, T1 has 0.
    for k in range(n, 0, -1):
        if time.time() > deadline:
            break
            
        tree_edges = trees[k-1]
        if not tree_edges:
            result[k-1] = [0]
            continue
            
        # Try to embed tree T_k
        embedding = _embed_tree_csp(k, tree_adjs[k-1], used_edges, n, deadline)
        
        if embedding is not None:
            result[k-1] = embedding
            for u, v in tree_edges:
                hu, hv = embedding[u], embedding[v]
                edge = (min(hu, hv), max(hu, hv))
                used_edges.add(edge)
        else:
            result[k-1] = None
            
    return result

def _embed_tree_csp(k, tree_adj, used_edges, n, deadline):
    """
    Embed tree of size k into K_n avoiding used_edges.
    Returns list of length k, where embedding[i] is host vertex for tree vertex i, or None.
    """
    if time.time() > deadline:
        return None
        
    # If k == 1, just pick any vertex
    if k == 1:
        for h in range(n):
            return [h]
            
    # Fix tree vertex 0 to host 0 to break symmetry
    assigned = {0: 0}
    unassigned = set(range(1, k))
    
    def get_available_hosts(tv):
        """
        Get available host vertices for tree vertex tv, given current assignments.
        """
        avail = set(range(n))
        # Remove hosts already used by assigned tree vertices
        for other_tv, h in assigned.items():
            if other_tv != tv:
                avail.discard(h)
                
        # Check edge constraints with assigned neighbors
        for neighbor in tree_adj[tv]:
            if neighbor in assigned:
                h_n = assigned[neighbor]
                # Edge (h, h_n) must not be in used_edges
                for h in list(avail):
                    edge = (min(h, h_n), max(h, h_n))
                    if edge in used_edges:
                        avail.discard(h)
                        
        return avail

    def backtrack():
        if time.time() > deadline:
            return False
            
        if not unassigned:
            return True
            
        # MRV: Pick unassigned vertex with smallest domain
        min_tv = None
        min_domain = None
        min_size = float('inf')
        
        for tv in unassigned:
            domain = get_available_hosts(tv)
            if len(domain) < min_size:
                min_size = len(domain)
                min_tv = tv
                min_domain = domain
            if min_size == 0:
                return False
                
        if min_size == 0:
            return False
            
        # Try each host in min_domain
        for h in min_domain:
            # Assign
            assigned[min_tv] = h
            unassigned.remove(min_tv)
            
            if backtrack():
                return True
                
            # Unassign
            del assigned[min_tv]
            unassigned.add(min_tv)
            
        return False

    if backtrack():
        return [assigned[i] for i in range(k)]
    else:
        return None
# EVOLVE-BLOCK-END
