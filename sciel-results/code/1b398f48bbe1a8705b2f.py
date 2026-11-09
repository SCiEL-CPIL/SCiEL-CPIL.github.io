"""Rewritten program: spectral init + penalty-based objective + SA + high-precision polish."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _build_adj_bits(edges, n):
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    return adj_bits

def _find_triangles(adj_bits, n):
    """Find all triangles using bitmask adjacency. Returns list of (u,v,w) with u<v<w."""
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_bits[u] >> v) & 1:
                continue
            common = adj_bits[u] & adj_bits[v]
            # Only consider w > v to avoid duplicates
            for w in range(v + 1, n):
                if (common >> w) & 1:
                    triangles.append((u, v, w))
    return triangles

def _score_graph(adj_bits, n):
    """Return (max_triangle_deg_sum, triangles, degrees)."""
    deg = [0] * n
    for i in range(n):
        deg[i] = bin(adj_bits[i]).count('1')
    triangles = _find_triangles(adj_bits, n)
    if not triangles:
        return 0, triangles, deg
    max_sum = max(deg[u] + deg[v] + deg[w] for u, v, w in triangles)
    return max_sum, triangles, deg

def _penalty_objective(triangles, deg, t_target=23.0, gamma=2.0):
    """
    Penalty-based smooth surrogate: for each triangle, compute max(0, sum - t_target)^gamma,
    sum them, and add a small variance term to encourage uniformity.
    This is a smooth-ish objective that aggressively penalizes triangles above t_target.
    """
    if not triangles:
        return 0.0
    sums = [deg[u] + deg[v] + deg[w] for u, v, w in triangles]
    # Penalty for exceeding target
    penalty = sum(max(0, s - t_target) ** gamma for s in sums)
    # Variance to flatten distribution
    if len(sums) > 1:
        mean_s = sum(sums) / len(sums)
        variance = sum((s - mean_s) ** 2 for s in sums) / len(sums)
    else:
        variance = 0.0
    return penalty + 0.05 * variance

def _spectral_init(n, num_edges, seed=42):
    """
    Spectral initialization: use eigenvector of random symmetric matrix to rank edges.
    """
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0)
    A += 0.01 * np.eye(n)
    
    eigvals, eigvecs = np.linalg.eigh(A)
    x = eigvecs[:, -1]
    
    edge_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            score = abs(x[i] * x[j])
            edge_scores.append((score, i, j))
    
    edge_scores.sort(key=lambda t: -t[0])
    selected = [(i, j) for _, i, j in edge_scores[:num_edges]]
    return selected

def _k88_plus_one(n, num_edges):
    """K_{8,8} plus one intra-partition edge."""
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((u, v))
    edges.add((0, 1))
    return edges

def _local_search_polish(n, edges, deadline, max_iter=200000):
    """
    High-precision local search: try all single-edge swaps and accept
    any improvement in the hard max score, then in the penalty objective.
    """
    best_edges = set(edges)
    adj_bits = _build_adj_bits(best_edges, n)
    best_score, best_tris, best_deg = _score_graph(adj_bits, n)
    best_obj = _penalty_objective(best_tris, best_deg)
    
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    non_edges = list(all_pairs - best_edges)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if not non_edges:
            break
        
        improved = False
        deg = best_deg
        
        # Try removing edges from worst triangles first
        if best_tris:
            worst = max(best_tris, key=lambda t: deg[t[0]] + deg[t[1]] + deg[t[2]])
            worst_edges = [(min(worst[0], worst[1]), max(worst[0], worst[1])),
                          (min(worst[0], worst[2]), max(worst[0], worst[2])),
                          (min(worst[1], worst[2]), max(worst[1], worst[2]))]
            candidates_rem = worst_edges + [random.choice(list(best_edges))]
        else:
            candidates_rem = [random.choice(list(best_edges))]
        
        for rem in candidates_rem:
            if time.time() > deadline:
                break
            if rem not in best_edges:
                continue
            
            for add in random.sample(non_edges, min(len(non_edges), 50)):
                new_edges = best_edges - {rem} | {add}
                new_adj = [b for b in adj_bits]
                ru, rv = rem
                new_adj[ru] &= ~(1 << rv)
                new_adj[rv] &= ~(1 << ru)
                au, av = add
                new_adj[au] |= (1 << av)
                new_adj[av] |= (1 << au)
                
                new_score, new_tris, new_deg = _score_graph(new_adj, n)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = new_edges
                    adj_bits = new_adj
                    best_tris = new_tris
                    best_deg = new_deg
                    best_obj = _penalty_objective(new_tris, new_deg)
                    non_edges = list(all_pairs - best_edges)
                    improved = True
                    break
                elif new_score == best_score:
                    new_obj = _penalty_objective(new_tris, new_deg)
                    if new_obj < best_obj - 1e-6:
                        best_edges = new_edges
                        adj_bits = new_adj
                        best_tris = new_tris
                        best_deg = new_deg
                        best_obj = new_obj
                        non_edges = list(all_pairs - best_edges)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try random swaps
            rem = random.choice(list(best_edges))
            add = random.choice(non_edges)
            new_edges = best_edges - {rem} | {add}
            new_adj = [b for b in adj_bits]
            ru, rv = rem
            new_adj[ru] &= ~(1 << rv)
            new_adj[rv] &= ~(1 << ru)
            au, av = add
            new_adj[au] |= (1 << av)
            new_adj[av] |= (1 << au)
            
            new_score, new_tris, new_deg = _score_graph(new_adj, n)
            new_obj = _penalty_objective(new_tris, new_deg)
            
            if new_score < best_score or (new_score == best_score and new_obj < best_obj - 1e-6):
                best_score = new_score
                best_edges = new_edges
                adj_bits = new_adj
                best_tris = new_tris
                best_deg = new_deg
                best_obj = new_obj
                non_edges = list(all_pairs - best_edges)
    
    return best_edges, best_score

def construct():
    n = 16
    num_edges = 65
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    random.seed(42)
    
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    # Multiple initializations
    inits = []
    
    # 1. K_{8,8} + 1
    k88 = _k88_plus_one(n, num_edges)
    inits.append(k88)
    
    # 2. Spectral with different seeds
    for seed in [42, 123, 777, 2024, 555, 999, 111, 333]:
        spec = set(_spectral_init(n, num_edges, seed=seed))
        if len(spec) == num_edges:
            inits.append(spec)
    
    # Evaluate all inits
    best_edges = None
    best_score = float('inf')
    best_obj = float('inf')
    best_adj = None
    best_tris = None
    best_deg = None
    
    for edge_set in inits:
        if len(edge_set) != num_edges:
            continue
        adj = _build_adj_bits(edge_set, n)
        sc, tris, deg = _score_graph(adj, n)
        obj = _penalty_objective(tris, deg)
        if sc < best_score or (sc == best_score and obj < best_obj):
            best_score = sc
            best_obj = obj
            best_edges = set(edge_set)
            best_adj = adj
            best_tris = tris
            best_deg = deg
    
    if best_edges is None:
        best_edges = k88
        best_adj = _build_adj_bits(best_edges, n)
        best_score, best_tris, best_deg = _score_graph(best_adj, n)
        best_obj = _penalty_objective(best_tris, best_deg)
    
    # Simulated annealing with penalty objective
    T0 = 8.0
    T_end = 0.01
    cooling_rate = 0.99998
    T = T0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 25:
        cur_score, cur_tris, cur_deg = _score_graph(best_adj, n)
        cur_obj = _penalty_objective(cur_tris, cur_deg)
        
        # Generate move: remove one edge, add one edge
        if cur_tris:
            worst = max(cur_tris, key=lambda t: cur_deg[t[0]] + cur_deg[t[1]] + cur_deg[t[2]])
            worst_edges = [(min(worst[0], worst[1]), max(worst[0], worst[1])),
                          (min(worst[0], worst[2]), max(worst[0], worst[2])),
                          (min(worst[1], worst[2]), max(worst[1], worst[2]))]
            if random.random() < 0.6:
                rem = random.choice(worst_edges)
                if rem not in best_edges:
                    rem = random.choice(list(best_edges))
            else:
                rem = random.choice(list(best_edges))
        else:
            rem = random.choice(list(best_edges))
        
        non_edges_list = [p for p in all_pairs if p not in best_edges]
        if not non_edges_list:
            continue
        add = random.choice(non_edges_list)
        
        new_adj = [b for b in best_adj]
        ru, rv = rem
        new_adj[ru] &= ~(1 << rv)
        new_adj[rv] &= ~(1 << ru)
        au, av = add
        new_adj[au] |= (1 << av)
        new_adj[av] |= (1 << au)
        
        new_score, new_tris, new_deg = _score_graph(new_adj, n)
        new_obj = _penalty_objective(new_tris, new_deg)
        
        delta_obj = new_obj - cur_obj
        
        accept = False
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score and new_obj < cur_obj:
            accept = True
        elif T > 0 and random.random() < math.exp(-max(0, delta_obj) / T):
            accept = True
        
        if accept:
            best_edges = set(best_edges) - {rem} | {add}
            best_adj = new_adj
            best_tris = new_tris
            best_deg = new_deg
            best_score = new_score
            best_obj = new_obj
            no_improve = 0
        else:
            no_improve += 1
        
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        if no_improve > max_no_improve:
            T = T0
            no_improve = 0
    
    # Final high-precision polish
    polish_deadline = deadline - 3
    best_edges, best_score = _local_search_polish(n, best_edges, polish_deadline, max_iter=300000)
    
    # Verify
    adj_final = _build_adj_bits(best_edges, n)
    final_score, final_tris, final_deg = _score_graph(adj_final, n)
    
    edge_list = [[u, v] for u, v in sorted(best_edges)]
    return {'n': n, 'edges': edge_list}
# EVOLVE-BLOCK-END
