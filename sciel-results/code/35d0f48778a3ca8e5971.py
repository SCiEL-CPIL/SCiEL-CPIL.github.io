import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _is_k33_free(adj):
    """Check if graph with adjacency matrix adj (n x n bool) is K3,3-free.
    K3,3 exists iff there are two disjoint triples with all 9 cross-edges.
    Equivalent: there exist three vertices a,b,c and three vertices d,e,f
    (all six distinct) such that each of a,b,c is adjacent to each of d,e,f.
    """
    n = adj.shape[0]
    # For each pair (u,v), count common neighbors. If any pair has >=3 common
    # neighbors, we need to check if those common neighbors can form a triple
    # disjoint from {u,v} that are all adjacent to u,v. Actually simpler:
    # K3,3 exists iff there exist 3 vertices A,B,C and 3 vertices D,E,F with
    # all edges between {A,B,C} and {D,E,F}.
    # 
    # Efficient check: For each 3-subset S, let N(S) = intersection of neighborhoods.
    # If |N(S) \ S| >= 3, then pick any 3 from N(S)\S, but we also need those 3
    # to be pairwise non-adjacent? No! K3,3 doesn't require non-adjacency within parts.
    # Wait, K3,3 as a subgraph: we need 6 vertices partitioned into two sets of 3,
    # with all 9 cross edges present. The vertices within each part don't need to be
    # non-adjacent. So we just need: exist S = {a,b,c} and T = {d,e,f}, disjoint,
    # such that every vertex in S is adjacent to every vertex in T.
    #
    # So: for each 3-subset S, compute common neighbors CN(S) = N(a) ∩ N(b) ∩ N(c).
    # If |CN(S) \ S| >= 3, then K3,3 exists (pick any 3 from CN(S)\S as T).
    
    from itertools import combinations
    
    for S in combinations(range(n), 3):
        a, b, c = S
        # Common neighbors of a, b, c
        common = adj[a] & adj[b] & adj[c]
        # Remove a, b, c themselves (they shouldn't be in their own neighborhood in simple graph)
        common[a] = False
        common[b] = False
        common[c] = False
        if common.sum() >= 3:
            return False
    return True

def _edge_count(adj):
    return int(adj.sum() // 2)

def _can_add_edge(adj, u, v):
    """Check if adding edge (u,v) keeps graph K3,3-free."""
    n = adj.shape[0]
    from itertools import combinations
    
    # Adding edge (u,v) creates K3,3 iff there exist 3-subsets S and T such that
    # the new edge is the "last" missing edge. Specifically, the new K3,3 must use
    # the edge (u,v). So u and v must be in different parts (or same part? No, 
    # in K3,3, edges only go between parts, so u and v are in different parts).
    # 
    # Case 1: u is in part A (size 3), v is in part B (size 3).
    #   We need 2 more vertices a1,a2 adjacent to v (for A part) and 2 more b1,b2
    #   adjacent to u (for B part), with all cross edges present.
    #   Actually: A = {u, a1, a2}, B = {v, b1, b2}.
    #   Need: a1,a2 adjacent to v; b1,b2 adjacent to u; and all existing cross edges
    #   between {a1,a2} and {b1,b2}, plus a1,a2 adjacent to v (already said), 
    #   u adjacent to b1,b2 (already said), u adjacent to v (the new edge).
    #   So: a1,a2 in N(v) \ {u}, b1,b2 in N(u) \ {v}, and all 4 edges between 
    #   {a1,a2} and {b1,b2} exist, and a1,a2 not in {v,b1,b2}, b1,b2 not in {u,a1,a2}.
    #
    # This is getting complex. Let's just try adding the edge and checking.
    
    adj_new = adj.copy()
    adj_new[u, v] = True
    adj_new[v, u] = True
    
    # Check K3,3-free
    for S in combinations(range(n), 3):
        a, b, c = S
        common = adj_new[a] & adj_new[b] & adj_new[c]
        common[a] = False
        common[b] = False
        common[c] = False
        if common.sum() >= 3:
            return False
    return True

def _random_dense_graph(n, p=0.5, seed=None):
    """Generate a random graph with edge probability p."""
    rng = random.Random(seed)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p:
                adj[i, j] = True
                adj[j, i] = True
    return adj

def _greedy_add(adj, n):
    """Try to add all possible edges greedily."""
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u, v]:
                    if _can_add_edge(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
                        improved = True
    return adj

def _edge_swap_search(adj, n, max_iters=5000, time_limit=100):
    """Local search via edge swaps: remove one edge, add another."""
    start_time = time.time()
    best_adj = adj.copy()
    best_count = _edge_count(best_adj)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random edge to remove
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
        if not edges:
            break
        u_rm, v_rm = random.choice(edges)
        
        # Try to add a random non-edge
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
        if not non_edges:
            break
        u_add, v_add = random.choice(non_edges)
        
        # Perform swap
        adj[u_rm, v_rm] = False
        adj[v_rm, u_rm] = False
        adj[u_add, v_add] = True
        adj[v_add, u_add] = True
        
        # Check K3,3-free
        if _is_k33_free(adj):
            count = _edge_count(adj)
            if count > best_count:
                best_count = count
                best_adj = adj.copy()
        else:
            # Revert
            adj[u_rm, v_rm] = True
            adj[v_rm, u_rm] = True
            adj[u_add, v_add] = False
            adj[v_add, u_add] = False
    
    return best_adj, best_count

def _spectral_init(n, seed=None):
    """Initialize using spectral method: top eigenvector of random symmetric matrix."""
    rng = np.random.RandomState(seed)
    # Random symmetric matrix
    A = rng.randn(n, n)
    A = (A + A.T) / 2
    # Get top eigenvector
    eigvals, eigvecs = np.linalg.eigh(A)
    top_vec = eigvecs[:, -1]
    # Threshold at mean
    threshold = np.mean(top_vec)
    # Create adjacency: edge (i,j) if top_vec[i]*top_vec[j] > 0 and i != j
    # Or simpler: threshold the vector itself
    # Let's use: edge (i,j) if top_vec[i] + top_vec[j] > 2*threshold
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if top_vec[i] + top_vec[j] > 2 * threshold:
                adj[i, j] = True
                adj[j, i] = True
    return adj

def construct():
    """Construct a K3,3-free graph on 20 vertices maximizing edges."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 20
    best_adj = None
    best_count = 0
    
    # Strategy 1: Spectral initialization + greedy + swap search
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize
        if restart == 0:
            # Start with a random dense graph
            adj = _random_dense_graph(n, p=0.5, seed=restart)
        elif restart % 2 == 0:
            # Spectral init
            adj = _spectral_init(n, seed=restart)
        else:
            # Random init with varying density
            p = 0.3 + 0.4 * (restart % 5) / 4.0
            adj = _random_dense_graph(n, p=p, seed=restart * 7 + 1)
        
        # Check if K3,3-free; if not, try to fix by removing edges
        while not _is_k33_free(adj):
            # Find a K3,3 and remove an edge from it
            # Simple approach: randomly remove edges until K3,3-free
            n_edges = _edge_count(adj)
            if n_edges == 0:
                break
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
            random.shuffle(edges)
            removed = False
            for u, v in edges:
                adj[u, v] = False
                adj[v, u] = False
                if _is_k33_free(adj):
                    removed = True
                    break
            if not removed:
                break
        
        # Greedy add
        adj = _greedy_add(adj, n)
        count = _edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = adj.copy()
        
        # Edge swap local search
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            adj_swapped, count_swapped = _edge_swap_search(
                adj, n, max_iters=2000, time_limit=min(remaining_time - 2, 30)
            )
            if count_swapped > best_count:
                best_count = count_swapped
                best_adj = adj_swapped.copy()
    
    # Final greedy pass on best
    if best_adj is not None and time.time() < deadline - 5:
        best_adj = _greedy_add(best_adj, n)
        best_count = _edge_count(best_adj)
    
    # Convert to edge list
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
