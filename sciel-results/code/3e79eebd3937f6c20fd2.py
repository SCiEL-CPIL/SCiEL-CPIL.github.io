import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def get_subsets(n, k):
        return list(itertools.combinations(range(n), k))

    def check_regular(adj, subset):
        # adj is n x n int array
        # subset is tuple of 6 indices
        # Returns True if regular
        sub_adj = adj[np.ix_(subset, subset)]
        degs = sub_adj.sum(axis=1)
        return np.all(degs == degs[0])

    def repair(adj, n, max_iter=2000):
        subsets = get_subsets(n, 6)
        num_subsets = len(subsets)
        
        # Precompute which subsets contain which edges?
        # Or just iterate subsets containing the flipped edge.
        
        # Initial state
        is_reg = np.array([check_regular(adj, s) for s in subsets])
        num_reg = np.sum(is_reg)
        
        # Map edge to list of subset indices containing it
        # Edge (u, v) with u < v
        # Subset indices
        # This map might be large but n=20 is small.
        # Edges: 190. Subsets: 38760.
        # Each subset has 15 edges. 38760 * 15 = 581400 entries.
        # This is fine.
        
        edge_to_subsets = {}
        for i, s in enumerate(subsets):
            for u in range(6):
                for v in range(u+1, 6):
                    e = (s[u], s[v])
                    if e not in edge_to_subsets:
                        edge_to_subsets[e] = []
                    edge_to_subsets[e].append(i)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return False, adj
            if num_reg == 0:
                return True, adj
            
            # Pick a random regular subset
            reg_indices = np.where(is_reg)[0]
            idx = random.choice(reg_indices)
            subset = subsets[idx]
            
            # Pick a random edge in this subset to flip
            # We can just pick random u, v from subset
            u, v = random.sample(subset, 2)
            if u > v: u, v = v, u
            e = (u, v)
            
            # Flip edge
            adj[u, v] = 1 - adj[u, v]
            adj[v, u] = adj[u, v]
            
            # Update is_reg for all subsets containing e
            affected = edge_to_subsets.get(e, [])
            for idx2 in affected:
                old_reg = is_reg[idx2]
                new_reg = check_regular(adj, subsets[idx2])
                if old_reg and not new_reg:
                    num_reg -= 1
                    is_reg[idx2] = False
                elif not old_reg and new_reg:
                    num_reg += 1
                    is_reg[idx2] = True
                # If old_reg and new_reg: impossible? 
                # Flipping an edge in a regular graph makes it non-regular.
                # So old_reg=True => new_reg=False always.
                # If old_reg=False, new_reg can be True or False.
                
        return False, adj

    def to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    # Try n=20
    n_target = 20
    adj = np.zeros((n_target, n_target), dtype=int)
    # Random init
    mask = np.random.rand(n_target, n_target) < 0.5
    np.fill_diagonal(mask, False)
    adj = mask.astype(int)
    adj = (adj + adj.T) // 2
    
    success, adj = repair(adj, n_target, max_iter=5000)
    
    if success:
        return {'n': n_target, 'edges': to_edges(adj, n_target)}
    
    # If failed, try again with different seed/init?
    # Or try n=19
    n_target = 19
    adj = np.zeros((n_target, n_target), dtype=int)
    mask = np.random.rand(n_target, n_target) < 0.5
    np.fill_diagonal(mask, False)
    adj = mask.astype(int)
    adj = (adj + adj.T) // 2
    
    success, adj = repair(adj, n_target, max_iter=5000)
    if success:
        return {'n': n_target, 'edges': to_edges(adj, n_target)}
        
    return {'n': 5, 'edges': []}
