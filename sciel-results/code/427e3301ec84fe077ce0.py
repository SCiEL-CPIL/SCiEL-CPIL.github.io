"""
Structural approach: treat the problem as a degree-sequence optimization problem.

Key insight: The score is max over triangles of (deg[u] + deg[v] + deg[w]).
To minimize this, we want:
1. A degree sequence that is as balanced as possible (to reduce max degree)
2. The graph structure such that high-degree vertices don't form triangles together

The K_{8,8} + 1 edge structure gives all vertices degree 8 or 9.
The one internal edge creates exactly 8 triangles (the endpoints + their 8 common neighbors).
Each such triangle has degree sum = 9 + 9 + 8 = 26.

But we can do better by:
- Starting from a different base structure
- Using a smarter local search that focuses on the "bottleneck" triangles
- Trying multiple restarts with different initial structures

The structural change: instead of just K_{8,8}+1, we try:
1. K_{8,8}+1 (baseline)
2. Near-bipartite structures with small internal clusters
3. Random balanced graphs with 65 edges
4. Use a multi-start approach with different seeds

For the local search, we use a more targeted approach:
- Identify the worst triangle (max degree sum)
- Try moves that specifically reduce that triangle's degree sum
- Use simulated annealing with a better cooling schedule
"""
import time
import random
import math
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0  # safety margin before 150s kill


def _compute_score(adj, deg):
    """Compute max triangle degree sum."""
    best = 0
    for i in range(N):
        for j in range(i + 1, N):
            if not adj[i][j]:
                continue
            for k in range(j + 1, N):
                if adj[i][k] and adj[j][k]:
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best


def _compute_score_fast(adj_np, deg_np):
    """Vectorized score computation."""
    # For 16 vertices, we can enumerate all C(16,3) = 560 triples
    # and check if they form a triangle
    best = 0
    for i in range(N):
        for j in range(i + 1, N):
            if adj_np[i, j] == 0:
                continue
            for k in range(j + 1, N):
                if adj_np[i, k] and adj_np[j, k]:
                    s = deg_np[i] + deg_np[j] + deg_np[k]
                    if s > best:
                        best = s
    return best


def _gen_k88_plus1(seed_offset=0):
    """Generate K_{8,8} + 1 edge structure."""
    rng = random.Random(seed_offset)
    adj = [[False] * N for _ in range(N)]
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = adj[j][i] = True
    # Add one internal edge
    a, b = rng.sample(range(8), 2)
    adj[a][b] = adj[b][a] = True
    deg = [0] * N
    for i in range(N):
        for j in range(N):
            if adj[i][j]:
                deg[i] += 1
    return adj, deg


def _gen_balanced_random(seed_offset=0):
    """Generate a random balanced graph with 65 edges."""
    rng = random.Random(seed_offset)
    # Target degree ~ 65*2/16 = 8.125
    # Start with a random graph and adjust
    adj = [[False] * N for _ in range(N)]
    # Use configuration model-like approach
    # Assign target degrees
    target_deg = [8] * 16
    # We need sum = 130, so adjust
    # 8*16 = 128, need 2 more
    target_deg[0] = 9
    target_deg[1] = 9
    
    # Build graph using random edge selection
    edges_set = set()
    deg = [0] * N
    attempts = 0
    while len(edges_set) < TARGET_EDGES and attempts < 10000:
        attempts += 1
        u, v = rng.sample(range(N), 2)
        if u > v:
            u, v = v, u
        if (u, v) not in edges_set:
            # Check if adding this edge would exceed degree bounds
            if deg[u] < target_deg[u] and deg[v] < target_deg[v]:
                edges_set.add((u, v))
                adj[u][v] = adj[v][u] = True
                deg[u] += 1
                deg[v] += 1
    
    # If we didn't get enough edges, add more randomly
    while len(edges_set) < TARGET_EDGES:
        u, v = rng.sample(range(N), 2)
        if u > v:
            u, v = v, u
        if (u, v) not in edges_set:
            edges_set.add((u, v))
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
    
    return adj, deg


def _gen_near_bipartite(seed_offset=0):
    """Generate a near-bipartite graph: K_{8,8} with some internal edges
    and some missing cross edges to maintain 65 total."""
    rng = random.Random(seed_offset)
    adj = [[False] * N for _ in range(N)]
    
    # Start with K_{8,8}
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = adj[j][i] = True
    
    # Remove k random cross edges and add k random internal edges
    cross_edges = [(i, j) for i in range(8) for j in range(8, 16)]
    internal_edges = [(i, j) for i in range(8) for j in range(i+1, 8)] + \
                     [(i, j) for i in range(8, 16) for j in range(i+1, 16)]
    
    k = rng.randint(0, 4)  # 0 to 4 swaps
    removed = set(rng.sample(cross_edges, k))
    added = set(rng.sample(internal_edges, k))
    
    for u, v in removed:
        adj[u][v] = adj[v][u] = False
    for u, v in added:
        if not adj[u][v]:
            adj[u][v] = adj[v][u] = True
    
    deg = [0] * N
    for i in range(N):
        for j in range(N):
            if adj[i][j]:
                deg[i] += 1
    
    return adj, deg


def _local_search(adj, deg, cur_score, t0, rng, iters_budget, target_worst=None):
    """Targeted local search that focuses on reducing the worst triangle."""
    adj_np = np.array(adj, dtype=np.int8)
    deg_np = np.array(deg, dtype=np.int8)
    
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i][j]]
    nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i][j]]

    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]

    # Adaptive temperature: start high, cool down
    start_temp = 5.0
    end_temp = 0.1
    n = max(1, iters_budget)
    cool = (end_temp / start_temp) ** (1.0 / n)
    temp = start_temp

    iters = 0
    stagnant = 0
    MAX_STAGNANT = 2000

    while iters < iters_budget:
        if iters % 1000 == 0:
            if time.time() - t0 > DEADLINE:
                break
        
        if not edges or not nonedges:
            break

        # Strategy: 70% random swaps, 30% targeted swaps
        if rng.random() < 0.7:
            er = edges[rng.randrange(len(edges))]
            ea = nonedges[rng.randrange(len(nonedges))]
        else:
            # Targeted: try to remove an edge from a high-degree triangle
            # and add an edge that balances degrees
            # Find vertices with high degree
            max_deg = max(deg)
            high_deg_verts = [i for i in range(N) if deg[i] >= max_deg - 1]
            if len(high_deg_verts) >= 2:
                u, v = rng.sample(high_deg_verts, 2)
                # Try to find an edge (u,v) or (u,w) or (v,w) that's in a triangle
                found_er = None
                for a in high_deg_verts:
                    for b in high_deg_verts:
                        if a < b and adj[a][b]:
                            found_er = (a, b)
                            break
                    if found_er:
                        break
                if found_er:
                    er = found_er
                    # Add edge between lower degree vertices
                    low_deg_verts = [i for i in range(N) if deg[i] <= min(deg) + 1]
                    if len(low_deg_verts) >= 2:
                        x, y = rng.sample(low_deg_verts, 2)
                        ea = (x, y) if x < y else (y, x)
                        if adj[x][y]:
                            ea = nonedges[rng.randrange(len(nonedges))]
                    else:
                        ea = nonedges[rng.randrange(len(nonedges))]
                else:
                    er = edges[rng.randrange(len(edges))]
                    ea = nonedges[rng.randrange(len(nonedges))]
            else:
                er = edges[rng.randrange(len(edges))]
                ea = nonedges[rng.randrange(len(nonedges))]

        if er == ea:
            iters += 1
            continue
        u, v = er
        x, y = ea

        # Apply swap
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1; deg[v] -= 1
        adj[x][y] = adj[y][x] = True
        deg[x] += 1; deg[y] += 1
        
        adj_np[u, v] = 0; adj_np[v, u] = 0
        adj_np[x, y] = 1; adj_np[y, x] = 1
        deg_np[u] -= 1; deg_np[v] -= 1
        deg_np[x] += 1; deg_np[y] += 1

        new_score = _compute_score_fast(adj_np, deg_np)

        # Acceptance criterion
        accept = False
        if new_score < cur_score:
            accept = True
            stagnant = 0
        elif new_score == cur_score:
            accept = rng.random() < 0.3
        else:
            d = new_score - cur_score
            if temp > 1e-9:
                accept = math.exp(-d / temp) > rng.random()
            else:
                accept = False
        
        if accept:
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
            # Update edge lists
            edges.remove(er)
            nonedges.remove(ea)
            nonedges.append(er)
            edges.append(ea)
        else:
            # Undo
            adj[x][y] = adj[y][x] = False
            deg[x] -= 1; deg[y] -= 1
            adj[u][v] = adj[v][u] = True
            deg[u] += 1; deg[v] += 1
            
            adj_np[x, y] = 0; adj_np[y, x] = 0
            adj_np[u, v] = 1; adj_np[v, u] = 1
            deg_np[x] -= 1; deg_np[y] -= 1
            deg_np[u] += 1; deg_np[v] += 1
            stagnant += 1
        
        iters += 1
        temp *= cool

    return best_adj, best_deg, best_score


def construct():
    t0 = time.time()
    
    best_score = float('inf')
    best_adj = None
    best_deg = None
    
    # Multiple initial structures with different seeds
    generators = [
        _gen_k88_plus1,
        _gen_balanced_random,
        _gen_near_bipartite,
    ]
    
    # Try each generator with a few seeds
    seeds_per_gen = 3
    total_attempts = len(generators) * seeds_per_gen
    
    # Time budget per attempt
    time_per_attempt = (DEADLINE - (time.time() - t0) - 5.0) / max(1, total_attempts)
    
    for gen_idx, gen_func in enumerate(generators):
        for seed_idx in range(seeds_per_gen):
            if time.time() - t0 > DEADLINE - 10.0:
                break
            
            seed_offset = gen_idx * 1000 + seed_idx * 7 + 42
            adj, deg = gen_func(seed_offset)
            
            # Verify edge count
            edge_count = sum(sum(row) for row in adj) // 2
            if edge_count != TARGET_EDGES:
                continue
            
            score = _compute_score(adj, deg)
            
            if score < best_score:
                best_score = score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
            
            # Local search with time budget
            remaining = DEADLINE - (time.time() - t0) - 5.0
            if remaining > 3.0:
                # Estimate iterations: ~2000/s for this scoring
                iters_budget = int(min(remaining * 2000, 50000))
                iters_budget = max(100, iters_budget)
                
                rng = random.Random(seed_offset + 999)
                new_adj, new_deg, new_score = _local_search(
                    adj, deg, score, t0, rng, iters_budget
                )
                
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in new_adj]
                    best_deg = new_deg[:]
    
    # Final verification
    if best_adj is None:
        # Fallback: generate K_{8,8}+1
        best_adj, best_deg = _gen_k88_plus1(0)
    
    edges = [[i, j] for i in range(N) for j in range(i + 1, N) if best_adj[i][j]]
    return {'n': N, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res["edges"]))
    # Compute and print score for debugging
    adj = [[False]*N for _ in range(N)]
    for u, v in res["edges"]:
        adj[u][v] = adj[v][u] = True
    deg = [0]*N
    for i in range(N):
        for j in range(N):
            if adj[i][j]:
                deg[i] += 1
    score = _compute_score(adj, deg)
    print("score:", score)
