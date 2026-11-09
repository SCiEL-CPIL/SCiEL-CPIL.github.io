"""
Cyclic-group (Z_31) difference-set construction for minimizing 4-cycles
in graphs with 31 vertices and 91 edges.

Strategy:
- Model vertices as elements of Z_31.
- Build a symmetric connection set S (S = -S mod 31) of size 91 (since 91 is odd,
  we actually need |S| such that the number of undirected edges is 91).
  In a circulant graph on Z_n, the number of edges is n * |S| / 2 if S is symmetric
  and 0 not in S. But 31 * |S| / 2 must equal 91, so |S| = 182/31 which is not integer.
  
  So we cannot use a pure circulant graph. Instead, we use a "near-circulant" approach:
  Start with a circulant-like base and then adjust.

  Actually, let's think differently. The number of edges in a circulant graph with
  connection set S (symmetric, 0 not in S) is n * |S| / 2. For n=31, we need 91 edges.
  91 * 2 = 182. 182 / 31 = 5.87... not integer. So a pure circulant won't work.

  Alternative: Use a difference set approach where we define the edge set directly.
  For each vertex i, connect to i + s mod 31 for s in some set. But we need exactly 91 edges.

  Let's try a different structural approach:
  - Use the fact that 31 = 5^2 + 5 + 1 is a projective plane order.
  - The polarity graph of PG(2,5) has 31 vertices and is strongly regular.
  - But the polarity graph has a specific number of edges. Let's compute:
    In PG(2,q), the polarity graph has q^2+q+1 vertices. Each vertex is connected
    to q^2 - 1 vertices (the points on the polar line, minus itself). Wait, actually
    for a null polarity, each vertex is connected to q^2 - 1 others? Let me check.
    
    Actually, for a symplectic polarity in PG(2,q) with q odd, the polarity graph
    has each vertex of degree q^2 - 1. For q=5, that's 24. Total edges = 31*24/2 = 372.
    That's way too many.

  Let's go back to the difference set idea but more carefully.

  We want 91 edges on 31 vertices. Average degree = 182/31 ≈ 5.87.
  
  Let's try to construct a graph that is "almost regular" with degrees 5 and 6.
  27 vertices of degree 6 and 4 vertices of degree 5 gives 27*6 + 4*5 = 162 + 20 = 182. Good.

  Approach:
  1. Start with a random graph with 91 edges.
  2. Use simulated annealing with fast incremental C4 counting to minimize.

  The key insight from inspiration 3 is to use incremental delta calculation for C4,
  which is much faster than recomputing from scratch.

  When we add an edge (u,v), the new C4s created are:
  For each pair of common neighbors (a,b) of u and v, where a and b are not adjacent to each other
  in a way that would form a shorter cycle... Actually, adding edge (u,v) creates a C4
  for each pair {a,b} such that a and b are both neighbors of u and v, and a != b.
  But we also need to ensure that the C4 u-a-v-b-u is a valid 4-cycle, meaning a != b, 
  a != v, b != u, and the edges u-a, a-v, v-b, b-u all exist. Since a and b are common neighbors,
  u-a, a-v, v-b, b-u all exist. So the number of new C4s is C(common_neighbors(u,v), 2).

  When we remove an edge (u,v), the C4s destroyed are exactly C(common_neighbors(u,v), 2),
  but we need to be careful: some of those C4s might still exist via other paths? No,
  if we remove (u,v), then any C4 that used (u,v) is destroyed. The C4s that use (u,v)
  are exactly those where u and v are opposite vertices, which is C(common_neighbors(u,v), 2).
  But wait, u and v could also be adjacent vertices in a C4. If u and v are adjacent,
  then a C4 using edge (u,v) would have u and v as consecutive vertices. In that case,
  removing (u,v) destroys C4s of the form u-v-b-a-u where a and b are common neighbors
  of u and v? No, that's not right.

  Let me reconsider. A C4 is a cycle of length 4. If we have edge (u,v), it can be part of
  a C4 in two ways:
  1. u and v are opposite vertices in the C4: then the C4 is u-a-v-b-u where a,b are common neighbors.
  2. u and v are adjacent vertices in the C4: then the C4 is u-v-b-a-u where a is a neighbor of u 
     (a != v) and b is a neighbor of v (b != u), and a-b is an edge, and a != b.

  This is getting complicated. Let's use a simpler approach:

  The total number of C4s can be computed as:
  (1/2) * sum over all pairs {u,v} of C(common_neighbors(u,v), 2) - (number of triangles * something)
  
  Actually, the standard formula is:
  Number of C4s = (1/2) * [sum_v C(d(v), 2) - sum_{u<v} A[u,v] * (common_neighbors(u,v))]
  
  Hmm, that's not quite right either. Let me use the correct formula:

  The number of 4-cycles in a graph is:
  (1/8) * [sum_{u,v} (A^2)_{u,v}^2 - sum_v d(v)^2 + sum_{u<v} A[u,v] * (A^2)_{u,v}]
  
  This is getting too complex. Let's just use the straightforward approach:
  For each pair of non-adjacent vertices (u,v), the number of C4s where u and v are opposite
  is C(common_neighbors(u,v), 2). Each C4 has two pairs of opposite vertices, so:
  Total C4s = (1/2) * sum_{u<v, not edge} C(common_neighbors(u,v), 2)

  This is what the original code uses. For incremental updates, when we add edge (u,v):
  - Before adding, (u,v) was a non-edge, so it contributed C(common_neighbors(u,v), 2) to the sum.
  - After adding, (u,v) is an edge, so it no longer contributes.
  - But adding (u,v) changes common_neighbors for other pairs. Specifically, for any w, 
    common_neighbors(u,w) increases by 1 if v is a neighbor of w, and similarly for v.
  
  This incremental update is complex. Let's use a different approach:
  
  Use simulated annealing with periodic full recomputation, but make the full recomputation
  very fast using numpy. The original code's count_c4 is O(n^2) with a constant factor.
  For n=31, that's 31*30/2 = 465 pairs, which is fast.

  Let's just use a well-tuned simulated annealing with full recomputation, but with
  a better initial solution.

  For the initial solution, use a structured construction:
  - Start with a random graph.
  - Or, use a greedy construction that minimizes C4s at each step.

  Let's combine:
  1. Generate several random initial graphs.
  2. For each, run simulated annealing with a fast C4 counter.
  3. Keep the best result.

  To speed up, we can use numpy for the C4 count.
"""

import time
import random
import numpy as np

DEADLINE = 150
SAFETY_MARGIN = 5


def count_c4_fast(adj):
    """Count C4s using numpy. adj is a 31x31 int array (0/1)."""
    n = adj.shape[0]
    adj_sq = adj @ adj
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 0:
                common = int(adj_sq[u, v])
                if common >= 2:
                    total += common * (common - 1) // 2
    return total // 2


def random_graph(n, m, seed=None):
    rng = random.Random(seed)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges = rng.sample(all_pairs, m)
    return edges


def edges_to_adj(edges, n):
    adj = np.zeros((n, n), dtype=np.int32)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


def simulated_annealing(edges, n, m, time_limit, rng):
    """Simulated annealing to minimize C4s."""
    adj = edges_to_adj(edges, n)
    current_c4 = count_c4_fast(adj)
    best_c4 = current_c4
    best_edges = [list(e) for e in edges]
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Temperature schedule
    t_start = time.time()
    t_end = t_start + time_limit
    initial_temp = 10.0
    cooling_rate = 0.999
    
    temp = initial_temp
    accepted = 0
    rejected = 0
    
    while time.time() < t_end:
        # Pick a random edge to remove
        e1 = rng.choice(list(edge_set))
        
        # Pick a random non-edge to add
        while True:
            u2 = rng.randint(0, n - 1)
            v2 = rng.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) not in edge_set:
                break
        
        e2 = (u2, v2)
        
        if e1 == e2:
            continue
        
        # Perform swap
        u1, v1 = e1
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        edge_set.discard(e1)
        
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        edge_set.add(e2)
        
        new_c4 = count_c4_fast(adj)
        delta = new_c4 - current_c4
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_c4 = new_c4
            accepted += 1
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_edges = [list(e) for e in edge_set]
        else:
            # Undo
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            edge_set.add(e1)
            
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            edge_set.discard(e2)
            rejected += 1
        
        temp *= cooling_rate
    
    return best_c4, best_edges


def greedy_construct(n, m, time_limit=None):
    """Greedy: add edges one by one, choosing the edge that adds fewest new C4s."""
    if time_limit is None:
        time_limit = DEADLINE - SAFETY_MARGIN
    
    adj = np.zeros((n, n), dtype=np.int32)
    edge_set = set()
    edges = []
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    deadline = time.time() + time_limit
    
    while len(edges) < m:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_count = float('inf')
        
        # Consider a random subset of candidates for speed
        candidates = all_edges[:300]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            
            # Count common neighbors of u and v
            common = 0
            for w in range(n):
                if w != u and w != v and adj[u, w] and adj[v, w]:
                    common += 1
            
            new_c4s = common * (common - 1) // 2 if common >= 2 else 0
            
            if new_c4s < best_count:
                best_count = new_c4s
                best_edge = (u, v)
                if best_count == 0:
                    break
        
        if best_edge is None:
            for e in all_edges:
                if e not in edge_set:
                    best_edge = e
                    break
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
        edge_set.add((u, v))
        edges.append([u, v])
    
    return edges


def main():
    start_time = time.time()
    n = 31
    m = 91
    
    best_c4 = float('inf')
    best_edges = None
    
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Phase 1: Greedy construction
    greedy_edges = greedy_construct(n, m, time_limit=10)
    if greedy_edges and len(greedy_edges) == m:
        adj = edges_to_adj(greedy_edges, n)
        c4 = count_c4_fast(adj)
        if c4 < best_c4:
            best_c4 = c4
            best_edges = greedy_edges
    
    # Phase 2: Simulated annealing from greedy result and random starts
    if best_edges is None:
        best_edges = random_graph(n, m, seed=0)
    
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 5:
            c4, improved = simulated_annealing(
                best_edges, n, m,
                time_limit=remaining - 2,
                rng=random.Random(42)
            )
            if c4 < best_c4:
                best_c4 = c4
                best_edges = improved
    
    # Phase 3: More SA trials from different random starts
    for trial in range(20):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        seed = trial * 104729 + 1
        edges = random_graph(n, m, seed=seed)
        
        time_per_trial = min(remaining - 1, 5.0)
        
        c4, improved = simulated_annealing(
            edges, n, m,
            time_limit=time_per_trial,
            rng=random.Random(seed + 1)
        )
        
        if c4 < best_c4:
            best_c4 = c4
            best_edges = improved
    
    # Final polish
    if best_edges is None:
        best_edges = random_graph(n, m, seed=0)
    
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 2:
            c4, polished = simulated_annealing(
                best_edges, n, m,
                time_limit=remaining - 1,
                rng=random.Random(9999)
            )
            if c4 < best_c4:
                best_c4 = c4
                best_edges = polished
    
    # Verify
    if len(best_edges) != m:
        edge_set = set(tuple(sorted(e)) for e in best_edges)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for p in all_pairs:
            if len(edge_set) >= m:
                break
            if p not in edge_set:
                edge_set.add(p)
                best_edges.append(list(p))
        best_edges = best_edges[:m]
    
    return best_edges


def construct():
    return main()


if __name__ == "__main__":
    result = construct()
    assert len(result) == 91
    adj = edges_to_adj(result, 31)
    c4 = count_c4_fast(adj)
    print(f"C4 count: {c4}")
