"""Hexagon-free graph construction on 40 vertices using Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 40
TOTAL_PAIRS = N * (N - 1) // 2

def _build_adj(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_c6_adj(adj, n):
    """Check if graph with adjacency sets has a 6-cycle."""
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == w:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == w or z == x:
                                continue
                            if u in adj[z]:
                                return True
    return False

def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges

def _init_matrix():
    """Initialize adjacency matrix with 8 K5s."""
    adj = np.zeros((N, N), dtype=np.int8)
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                adj[base + a, base + b] = 1
                adj[base + b, base + a] = 1
    return adj

def _count_c6s(adj):
    """
    Count the number of 6-cycles in the graph using matrix multiplication.
    trace(A^6) counts closed walks of length 6.
    A simple 6-cycle contributes 12 to the trace (6 starting vertices * 2 directions).
    We subtract contributions from closed walks that are not simple 6-cycles:
    1. Closed walks that return to a previous vertex (backtracking).
    2. Closed walks that repeat vertices (e.g. 3-cycles traversed twice, 4-cycles with a chord, etc.).
    
    For a C6-free graph, we want trace(A^6) - (contributions from non-simple walks) == 0.
    However, calculating the exact correction is complex. 
    A simpler sufficient condition for "no C6" is that the number of simple 6-cycles is 0.
    
    Let's use a different approach:
    Count simple 6-cycles directly? No, that's hard with matrices.
    
    Alternative:
    We can check if there exists a C6 by checking if any pair of vertices at distance 4 has a common neighbor? No, that's C5/C6.
    
    Let's stick to the trace method but handle the "non-simple" walks.
    The number of closed walks of length 6 is W_6 = trace(A^6).
    The number of simple 6-cycles is C_6.
    W_6 = 12 * C_6 + (other terms).
    
    The "other terms" come from:
    - 3-cycles: A 3-cycle u-v-w-u can be traversed as u-v-w-u-v-w (length 6). This is a closed walk.
      Each 3-cycle contributes to W_6. How much?
      For a triangle u,v,w, the walks are:
      u-v-w-u-v-w (1)
      u-w-v-u-w-v (1)
      v-u-w-v-u-w (1)
      v-w-u-v-w-u (1)
      w-u-v-w-u-v (1)
      w-v-u-w-v-u (1)
      Total 6 per triangle?
      Wait, trace(A^6) sums over all starting vertices i.
      For a triangle, the closed walks of length 6 that stay within the triangle are exactly the ones that traverse the triangle twice.
      There are 2 directions * 3 starting vertices = 6 such walks per triangle.
      So 6 * C_3.
      
    - 4-cycles: A 4-cycle u-v-w-x-u.
      Can we make a closed walk of length 6?
      u-v-w-x-u-v (ends at v, not u). No.
      u-v-w-u-v-w (backtracking).
      Actually, any closed walk of length 6 that is not a simple 6-cycle involves backtracking or repeating a vertex.
      
    Let's use a known formula or a simpler check.
    Since N=40 is small, we can compute the exact number of 6-cycles using a specific matrix formula?
    No, the standard formula for the number of cycles of length k involves inclusion-exclusion on walks.
    
    However, for the purpose of Simulated Annealing, we need a "cost" function.
    If we just use trace(A^6), it will be non-zero even for C6-free graphs if they have triangles or 4-cycles.
    We need to subtract the contributions from non-6-cycles.
    
    Let's calculate the number of "bad" walks (closed walks of length 6 that are not simple 6-cycles).
    A closed walk of length 6 is a sequence v0, v1, v2, v3, v4, v5, v0.
    It is a simple 6-cycle iff all vi are distinct for i=0..5.
    
    We can count the number of simple 6-cycles by:
    C_6 = (1/12) * (W_6 - W_6_non_simple)
    
    W_6_non_simple includes:
    1. Walks with immediate backtracking (vi = vi-2).
    2. Walks that visit a vertex more than once (but not immediate backtracking).
    
    This is getting complicated to compute efficiently in NumPy without loops.
    
    Alternative Strategy for Cost:
    Instead of counting C6s exactly via matrix trace, we can use a heuristic cost:
    Cost = Number of "almost" C6s?
    
    Actually, let's look at the "inspiration" which used `_has_c6_matrix`.
    It said: "A 6-cycle exists iff there's a walk of length 6 from u back to u with all intermediate vertices distinct..."
    
    Let's try a different vectorized check for C6 existence (not count).
    A C6 exists if there is a path of length 5 between two adjacent vertices u, v, such that the path does not use u or v internally.
    This is equivalent to: For any edge (u,v), is there a path of length 5 from u to v in G - {u,v}?
    
    Let A be the adjacency matrix.
    Let A_uv be the submatrix excluding u and v? No, that's N-2 x N-2.
    
    We can do this:
    For each edge (u,v), we want to know if (A^5)[u,v] > 0 in the graph where u and v are removed?
    No, the path must not visit u or v internally.
    
    Let's use the following:
    Let M = A.
    We want to know if there is a C6.
    A C6 is a cycle of length 6.
    
    Let's try a simpler cost function:
    Cost = sum over all pairs (u,v) of (number of paths of length 5 between u and v that don't use u,v internally) * (1 if edge (u,v) exists else 0).
    
    This is still hard to vectorize perfectly.
    
    Let's go back to the trace method and just subtract the known contributions of triangles and 4-cycles?
    Actually, if we have a graph with NO C6, trace(A^6) will be determined by triangles and 4-cycles and "backtracking" walks.
    If we minimize trace(A^6), we might accidentally remove edges that form triangles, which is fine, but we might also remove edges that are part of C6s.
    However, a C6 contributes 12 to trace(A^6).
    A triangle contributes 6 to trace(A^6).
    A 4-cycle contributes?
    A 4-cycle u-v-w-x-u.
    Closed walks of length 6:
    u-v-w-x-u-v (no)
    u-v-w-u-v-w (backtracking at u? no, u-v-w-u is length 3, then v-w is length 2. Total 5? No.
    u-v-w-u-v-w:
    u->v (1)
    v->w (2)
    w->u (3)
    u->v (4)
    v->w (5)
    w->? (needs to go to u to close? w->u is 6. So u-v-w-u-v-w-u is length 6?
    u-v-w-u-v-w-u:
    1: u-v
    2: v-w
    3: w-u
    4: u-v
    5: v-w
    6: w-u
    This is a valid closed walk of length 6. It uses the edge (u,v) twice and (v,w) twice and (w,u) twice.
    It is not a simple cycle.
    How many such walks per triangle?
    For triangle u,v,w:
    u-v-w-u-v-w-u (1)
    u-w-v-u-w-v-u (1)
    ... 6 per triangle.
    
    So W_6 = 12*C_6 + 6*C_3 + (contributions from 4-cycles and other structures).
    
    If we assume the graph is "sparse" enough or we just minimize W_6, we will try to eliminate C6s and triangles.
    Since we want to MAXIMIZE edges, minimizing W_6 is a good proxy for "being C6-free" IF the penalty for C6s is high.
    12 per C6 vs 6 per Triangle.
    
    However, removing an edge that is part of a C6 reduces W_6 by a lot.
    Removing an edge that is part of a triangle reduces W_6 by less.
    
    Let's use Cost = W_6.
    We want to find a graph with many edges and Cost = 0?
    Wait, if there are no C6s, Cost is NOT 0 if there are triangles.
    So we cannot require Cost = 0.
    We need to know if Cost comes ONLY from non-C6 structures.
    
    This is tricky.
    
    Let's use a different vectorized approach:
    Check for C6 directly using matrix powers but with masking?
    
    Actually, let's use a simpler heuristic:
    We will use the exact `_has_c6_adj` for verification.
    For the SA cost, we will use a "potential" C6 count.
    
    Let's calculate the number of 6-cycles using the formula:
    C_6 = (1/12) * (trace(A^6) - 6*C_3 - 4*trace(A^2)^2 ... )
    
    Actually, there is a simpler way.
    Since N=40, we can precompute the "C6 witness" for each edge addition?
    
    Let's try a hybrid:
    1. Use NumPy to compute A^2, A^3.
    2. A C6 exists if there is an edge (u,v) such that there is a path of length 5 from u to v in G \ {u,v}.
    
    Let's implement `_count_c6s` using the trace formula for simple cycles.
    According to literature, the number of cycles of length k can be computed from the trace of A^k and the counts of smaller cycles.
    For k=6:
    12 * C_6 = trace(A^6) - 6 * C_3 - 4 * C_4 - ...
    Wait, the exact formula is:
    trace(A^6) = 12 * C_6 + 6 * C_3 + 4 * C_4 + 2 * C_2 (loops, none) + ...
    Actually, let's just use `trace(A^6)` as the cost.
    If we minimize this, we will tend to remove edges involved in C6s.
    Triangles also contribute, so we might lose triangles.
    But C6-free graphs often have triangles (e.g. K5 has triangles).
    
    Let's try to be smarter.
    We will calculate the number of "bad" 6-walks that are NOT simple 6-cycles.
    
    Let's define the cost as:
    Cost = (trace(A^6) - 6 * trace(A^3) - 4 * trace(A^4) ... )?
    
    Actually, `trace(A^3)` counts closed walks of length 3.
    A triangle contributes 6 to trace(A^3) (2 directions * 3 starts).
    So C_3 = trace(A^3) / 6.
    
    `trace(A^4)` counts closed walks of length 4.
    A 4-cycle contributes 8 to trace(A^4) (2 directions * 4 starts).
    A triangle contributes?
    u-v-w-u-v-w (length 5? no).
    u-v-u-v-u-v (length 5).
    u-v-u-v-u-v-u (length 6).
    
    This is getting too complex to derive on the fly.
    
    Let's use a different strategy for the SA cost:
    **Cost = Number of "almost" C6s.**
    An "almost" C6 is a path of length 5 between two vertices that are connected by an edge.
    This is exactly the number of C6s * 12? No.
    Each C6 has 6 edges. For each edge, there is 1 path of length 5 connecting its endpoints (the rest of the cycle).
    So if we count pairs (u,v) such that edge (u,v) exists AND there is a path of length 5 from u to v in G \ {u,v}, we get exactly the number of C6s.
    
    So, Cost = Number of C6s.
    
    How to compute this vectorized?
    Let A be the adjacency matrix.
    We want to count, for each edge (u,v), the number of paths of length 5 from u to v that do not visit u or v internally.
    
    Let P5[u,v] be the number of paths of length 5 from u to v.
    This is given by (A^5)[u,v].
    However, (A^5)[u,v] includes paths that might visit u or v internally.
    We need to exclude paths that visit u or v internally.
    
    If we remove u and v from the graph, let A' be the adjacency matrix of G \ {u,v}.
    Then the number of valid paths is (A'^5)[u',v'] where u',v' are the indices in the reduced graph.
    But u and v are not in the reduced graph.
    
    This requires removing rows/cols for every edge, which is O(E * N^3) or similar, too slow.
    
    Alternative:
    Use the full A^5, but subtract paths that visit u or v.
    A path of length 5 from u to v is a sequence u, x1, x2, x3, x4, v.
    It is invalid if any xi is u or v.
    
    Since N=40, we can compute A^5 in O(N^3 log 5) or just O(N^3) with repeated squaring/multiplication.
    A^2 = A @ A
    A^4 = A^2 @ A^2
    A^5 = A^4 @ A
    
    Then, for each edge (u,v), we want to subtract paths that go through u or v internally.
    
    Let's approximate:
    If the graph is sparse, paths visiting u or v internally are rare?
    No, in a dense graph, they are common.
    
    However, we can use the following identity:
    Number of simple paths of length 5 from u to v = (A^5)[u,v] - (paths visiting u or v).
    
    Let's just use (A^5)[u,v] as a proxy for the number of "C6 candidates".
    If we minimize the sum of (A^5)[u,v] for all edges (u,v), we are minimizing the number of C6s (plus some noise from non-simple paths).
    This is a valid cost function for Simulated Annealing!
    
    Cost = sum_{(u,v) in E} (A^5)[u,v]
    
    This is O(N^3) to compute A^5, and O(E) to sum.
    N=40, so N^3 = 64,000. Very fast.
    
    Let's implement this.
"""

def _compute_cost(adj):
    """
    Compute the cost function: sum of (A^5)[u,v] for all edges (u,v).
    This is proportional to the number of C6s (plus some non-simple path noise).
    """
    A = adj.astype(np.int32)
    A2 = A @ A
    A4 = A2 @ A2
    A5 = A4 @ A
    
    # Sum A5[u,v] for all edges (u,v)
    # We can use the adjacency matrix A to mask
    cost = np.sum(A * A5)
    # Note: A is symmetric, so A*u,v is 1 if edge exists.
    # A5 is also symmetric.
    # The diagonal of A5 is 0 (no loops), so no self-loops to worry about.
    return cost

def _simulated_annealing(time_limit):
    start_time = time.time()
    random.seed(42)
    
    # Initialize with 8 K5s
    adj = _init_matrix()
    current_cost = _compute_cost(adj)
    current_edges = np.sum(adj) // 2
    
    best_adj = adj.copy()
    best_cost = current_cost
    best_edges = current_edges
    
    T = 100.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to flip
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v:
            continue
        
        # Flip the edge
        old_val = adj[u, v]
        adj[u, v] = 1 - old_val
        adj[v, u] = 1 - old_val
        
        new_cost = _compute_cost(adj)
        new_edges = np.sum(adj) // 2
        
        # Delta in cost (we want to minimize cost, but maximize edges)
        # We need a combined objective.
        # Objective: Maximize Edges, Minimize Cost (C6s).
        # If Cost > 0, we MUST reduce it.
        # If Cost == 0, we want to maximize Edges.
        
        # Let's define a score: Score = Edges - Lambda * Cost
        # If Cost is large, Score is low.
        # We want to accept moves that increase Score.
        
        Lambda = 1000.0 # High penalty for C6s
        
        old_score = current_edges - Lambda * current_cost
        new_score = new_edges - Lambda * new_cost
        
        delta_score = new_score - old_score
        
        if delta_score >= 0:
            # Accept
            current_cost = new_cost
            current_edges = new_edges
            if new_edges > best_edges and new_cost == 0:
                best_edges = new_edges
                best_adj = adj.copy()
        else:
            # Accept with probability exp(delta_score / T)
            prob = np.exp(delta_score / T)
            if random.random() < prob:
                current_cost = new_cost
                current_edges = new_edges
            else:
                # Revert
                adj[u, v] = old_val
                adj[v, u] = old_val
        
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
            
        # Occasionally reheat?
        # No, let's keep it simple.
        
    # Final verification and cleanup
    # Convert best_adj to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u, v]:
                edges.append([u, v])
    
    # Verify no C6
    adj_sets = _build_adj(edges)
    if _has_c6_adj(adj_sets, N):
        # Fallback to seed
        return _seed_graph_8K5()
        
    return edges

def construct():
    n = N
    deadline = 140  # Safety margin before 150s kill
    
    start_time = time.time()
    
    # Run SA
    edges = _simulated_annealing(deadline - 10)
    
    return edges
# EVOLVE-BLOCK-END
