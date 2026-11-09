import time
import random
import numpy as np

DEADLINE = 130.0  # safety margin before 150s kill

def _score_fast(adj, deg, n):
    """Fast score computation using adjacency matrix and degree array."""
    best = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
    return best

def _build_adj_deg(edges, n):
    adj = [[False]*n for _ in range(n)]
    deg = [0]*n
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _rand_base(n, m):
    verts = list(range(n))
    all_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.append((i, j))
    random.shuffle(all_pairs)
    return [list(p) for p in all_pairs[:m]]

def construct():
    t0 = time.time()
    n, m = 16, 65
    best_edges = None
    best_score = None
    rng = random.Random(12345)
    
    def base_solution():
        a = 8
        A = list(range(a))
        B = list(range(a, n))
        edges = []
        for u in A:
            for v in B:
                edges.append([u, v])
        edges.append([A[0], A[1]])
        return edges
    
    e0 = base_solution()
    adj0, deg0 = _build_adj_deg(e0, n)
    s0 = _score_fast(adj0, deg0, n)
    best_edges = e0
    best_score = s0
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_pairs_set = set(all_pairs)
    
    def edges_from_set(es):
        return [[i, j] for (i, j) in es]
    
    # Phase 1: Local search with incremental updates
    cur_set = set((min(u, v), max(u, v)) for u, v in e0)
    cur_adj, cur_deg = _build_adj_deg(e0, n)
    cur_score = s0
    no_improve = 0
    max_iters = 50000
    
    for it in range(max_iters):
        if time.time() - t0 > DEADLINE:
            break
        if no_improve > 3000:
            # Random restart with small perturbation
            if cur_set:
                rem = rng.choice(list(cur_set))
                cur_set.discard(rem)
                non_edges = [p for p in all_pairs if p not in cur_set]
                if non_edges:
                    cur_set.add(rng.choice(non_edges))
                cur_adj, cur_deg = _build_adj_deg(edges_from_set(cur_set), n)
                cur_score = _score_fast(cur_adj, cur_deg, n)
            continue
        
        if not cur_set:
            break
        rem = rng.choice(list(cur_set))
        non_edges = [p for p in all_pairs if p not in cur_set]
        if not non_edges:
            break
        add = rng.choice(non_edges)
        
        # Incremental update: try the swap
        new_set = set(cur_set)
        new_set.discard(rem)
        new_set.add(add)
        
        # Only compute full score if the swap looks promising
        # Quick heuristic: check if we're reducing max degree
        new_adj, new_deg = _build_adj_deg(edges_from_set(new_set), n)
        new_score = _score_fast(new_adj, new_deg, n)
        
        if new_score < cur_score:
            cur_set = new_set
            cur_adj = new_adj
            cur_deg = new_deg
            cur_score = new_score
            no_improve = 0
            if cur_score < best_score:
                best_score = cur_score
                best_edges = edges_from_set(cur_set)
        else:
            no_improve += 1
            # Simulated annealing accept
            if rng.random() < 0.05 and new_score <= cur_score + 1:
                cur_set = new_set
                cur_adj = new_adj
                cur_deg = new_deg
                cur_score = new_score
    
    # Phase 2: Systematic 2-swap local search for final polish
    if time.time() - t0 < DEADLINE - 10:
        # Try all pairs of edge removals and edge additions
        edge_list = list(cur_set)
        non_edge_list = [p for p in all_pairs if p not in cur_set]
        
        found_improvement = True
        while found_improvement and time.time() - t0 < DEADLINE - 5:
            found_improvement = False
            # Sample a subset of swaps to keep within time budget
            sample_size = min(len(edge_list) * len(non_edge_list), 5000)
            pairs = []
            for rem in edge_list:
                for add in non_edge_list:
                    pairs.append((rem, add))
            if len(pairs) > sample_size:
                pairs = rng.sample(pairs, sample_size)
            
            for rem, add in pairs:
                if time.time() - t0 > DEADLINE - 5:
                    break
                if rem == add:
                    continue
                new_set = set(cur_set)
                new_set.discard(rem)
                new_set.add(add)
                new_adj, new_deg = _build_adj_deg(edges_from_set(new_set), n)
                new_score = _score_fast(new_adj, new_deg, n)
                if new_score < cur_score:
                    cur_set = new_set
                    cur_adj = new_adj
                    cur_deg = new_deg
                    cur_score = new_score
                    found_improvement = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = edges_from_set(cur_set)
                    break
    
    return {'n': n, 'edges': best_edges}
