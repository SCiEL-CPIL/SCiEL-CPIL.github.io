import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135
    n = 20
    vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)
    
    pair_to_idx = {}
    idx_to_pair = [None] * num_pairs
    for idx, (i, j) in enumerate(all_pairs):
        pair_to_idx[(i, j)] = idx
        idx_to_pair[idx] = (i, j)

    triples = list(combinations(vertices, 3))
    
    # Precompute which triples contain each vertex
    triples_with_vertex = [[] for _ in range(n)]
    for idx, (u, v, w) in enumerate(triples):
        triples_with_vertex[u].append(idx)
        triples_with_vertex[v].append(idx)
        triples_with_vertex[w].append(idx)

    def popcount(x):
        return x.bit_count()

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Only need to check triples containing u or v.
        """
        bit_v = 1 << v
        bit_u = 1 << u
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        # Check triples containing u
        for idx in triples_with_vertex[u]:
            a, b, c = triples[idx]
            ma = adj_mask[a] if a != u else new_u
            mb = adj_mask[b] if b != u else new_u
            mc = adj_mask[c] if c != u else new_u
            if popcount(ma & mb & mc) > 2:
                return False
        
        # Check triples containing v but not u (to avoid double-checking)
        for idx in triples_with_vertex[v]:
            a, b, c = triples[idx]
            if a == u or b == u or c == u:
                continue  # Already checked
            ma = adj_mask[a] if a != v else new_v
            mb = adj_mask[b] if b != v else new_v
            mc = adj_mask[c] if c != v else new_v
            if popcount(ma & mb & mc) > 2:
                return False
        
        return True

    def greedy_add(adj_mask):
        """Add all possible edges greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & (1 << j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
                            changed = True
        return adj_mask

    def local_search(adj_mask, rng, max_iter=2000):
        """
        Local search with edge removal and re-addition.
        """
        best_adj = list(adj_mask)
        best_cnt = edge_count(best_adj)
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Pick a random edge to remove
            edges_in_graph = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        edges_in_graph.append((i, j))
            
            if not edges_in_graph:
                break
            
            u, v = rng.choice(edges_in_graph)
            new_adj = list(best_adj)
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            # Greedy add on the reduced graph
            new_adj = greedy_add(new_adj)
            cnt = edge_count(new_adj)
            
            if cnt > best_cnt:
                best_cnt = cnt
                best_adj = new_adj
        return best_adj, best_cnt

    def smart_repair(adj_mask):
        """
        Deterministically remove edges to remove all K3,3 subgraphs.
        """
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            
            # Find violation
            viol = None
            for t in triples:
                u, v, w = t
                common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                if popcount(common) > 2:
                    c_list = [x for x in range(n) if common & (1 << x)]
                    if c_list:
                        viol = (u, c_list[0])
                        break
            
            if viol is None:
                break
            
            u, c = viol
            adj_mask[u] &= ~(1 << c)
            adj_mask[c] &= ~(1 << u)
        
        return adj_mask

    def structured_start(rng):
        """
        Create a dense structured graph.
        """
        adj = [0] * n
        # Start with a fairly dense random graph
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.7:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    best_adj = None
    best_edges = -1
    
    # Run multiple restarts with different strategies
    for seed in range(80):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(seed * 12345 + 67890)
        adj = structured_start(rng)
        
        # Phase 1: Smart Repair to make it K3,3-free
        adj = smart_repair(adj)
        
        # Phase 2: Greedy Add to maximize edges
        adj = greedy_add(adj)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
        
        # Phase 3: Local Search
        if time.time() < deadline - 10:
            adj, cnt = local_search(adj, rng)
            if cnt > best_edges:
                best_edges = cnt
                best_adj = list(adj)
    
    # Final polish: try a few more local search iterations from the best
    if best_adj is not None and time.time() < deadline - 5:
        rng = random.Random(99999)
        best_adj, best_edges = local_search(best_adj, rng, max_iter=5000)

    if best_adj is None:
        # Fallback: Empty graph or simple bipartite
        best_adj = [0] * n
        best_edges = 0
        # Construct a bipartite graph K10,10
        for i in range(10):
            for j in range(10, 20):
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
        best_edges = 100

    edges = mask_to_edges(best_adj)
    return edges
