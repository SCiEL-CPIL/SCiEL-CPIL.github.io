import time
import random
import numpy as np

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 145
    
    # Vector-space construction over GF(5)
    # Map 31 vertices to 5-dimensional vectors with structure
    # Use a different algebraic structure than the polarity graph
    
    # Generate 31 distinct vectors in GF(5)^5 with specific properties
    # We'll use a structured approach: take all 1-dimensional subspaces of GF(5)^5
    # That gives (5^5 - 1)/(5-1) = 3120/4 = 780, too many
    
    # Alternative: use a carefully chosen set of 31 vectors
    # Let's use vectors from a 5-dimensional space with specific patterns
    
    # Create 31 vectors by taking a systematic set
    # Use the fact that we need exactly 91 edges
    
    # Strategy: Start with a structured graph based on modular arithmetic
    # Vertices 0..30, connect i to j if (j-i) mod 31 is in a specific set
    
    # First, let's try a circulant-like construction
    # For n=31 (prime), we can use difference sets
    
    # A (31, 15, 7) difference set would give a strongly regular graph
    # But we need 91 edges, so degree ~ 2*91/31 ≈ 5.87
    
    # Let's try a different approach: use the finite field GF(5) structure
    # Represent vertices as (a,b) where a in GF(5), b in GF(5), plus special vertices
    
    # Actually, let's use a direct algebraic construction
    # Vertices: elements of GF(5)^2 = 25 elements, plus 6 more
    
    # Better: use a random initial graph and optimize with fast numpy-based C4 counting
    
    # Initialize with a structured graph
    # Use a bipartite-like structure to minimize C4s
    
    # Create a 5-partite structure: partition 31 vertices into 5 groups
    # Sizes: 7, 6, 6, 6, 6
    groups = []
    remaining = 31
    for i in range(5):
        if i < 4:
            groups.append(6)
        else:
            groups.append(remaining - 24)  # 7
    
    # Assign vertices to groups
    vertex_group = []
    idx = 0
    for g in groups:
        for _ in range(g):
            vertex_group.append(idx)
            idx += 1
    
    # Build initial edge set: connect within a pattern that minimizes C4s
    # Use a structured random initialization
    random.seed(42)
    
    # Start with a random graph with 91 edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    
    # But let's be smarter: use a structured initialization
    # Connect vertices based on a modular pattern
    
    # Try: connect i to j if (j - i) mod 31 is in S
    # We need |S| such that 31*|S|/2 = 91, so |S| = 5.87... not integer
    
    # Let's use a hybrid: some modular edges + some random
    # First, pick 5 differences that give 31*5/2 = 77.5 edges... not clean
    
    # Alternative: just use a well-structured random graph
    # Initialize with edges that have low common-neighbor density
    
    edges = set()
    adj = [set() for _ in range(n)]
    
    # Greedy initialization: add edges that minimize C4 creation
    # Sort all pairs by some heuristic and add greedily
    
    # Compute a score for each potential edge: how many C4s it would create
    # An edge (u,v) creates C4s equal to the number of common neighbors of u and v
    # (each common neighbor w forms a C4 with any other common neighbor)
    # Actually, adding edge (u,v) creates C4s = C(common_neighbors, 2)
    
    # But we also need to account for existing edges
    
    # Let's do a greedy construction
    pair_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            # Initial common neighbors is 0 for empty graph
            # We'll update as we add edges
            pair_scores.append((0, i, j))
    
    # Sort by initial score (all 0), then by some tiebreaker
    # Use a deterministic order
    pair_scores.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Add edges greedily
    for score, i, j in pair_scores:
        if len(edges) >= target_edges:
            break
        common = len(adj[i] & adj[j])
        if common == 0:
            # This edge creates no new C4s
            edges.add((i, j))
            adj[i].add(j)
            adj[j].add(i)
        elif len(edges) < target_edges - 10:
            # Allow some edges with common neighbors if we need to fill up
            if common <= 1:
                edges.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    # If we don't have enough edges, add more
    if len(edges) < target_edges:
        remaining_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edges]
        random.shuffle(remaining_pairs)
        for i, j in remaining_pairs:
            if len(edges) >= target_edges:
                break
            edges.add((i, j))
            adj[i].add(j)
            adj[j].add(i)
    
    assert len(edges) == target_edges, f"Expected {target_edges} edges, got {len(edges)}"
    
    def binom2(x):
        return x * (x - 1) // 2 if x >= 2 else 0
    
    # Fast C4 counting using numpy
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int32)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    
    # C4 count = 1/4 * sum over all edges (u,v) of C(|N(u) ∩ N(v)|, 2)
    # Using matrix: for each pair (u,v), common neighbors = (A @ A)[u,v] - A[u,v]
    # But we only care about edges (u,v) where A[u,v]=1
    
    def count_c4(A, edges_set):
        A2 = A @ A
        total = 0
        for u, v in edges_set:
            common = A2[u, v] - A[u, v]
            total += binom2(int(common))
        return total // 4
    
    current_c4 = count_c4(A, edges)
    best_c4 = current_c4
    best_edges = set(edges)
    best_adj = [set(s) for s in adj]
    
    # Local search with fast delta computation
    # For swapping edge (u1,v1) with (u2,v2):
    # The C4 change can be computed efficiently
    
    random.seed(42)
    all_pairs_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    iterations = 0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and iterations < 200000 and no_improve < max_no_improve:
        iterations += 1
        
        if not edges:
            break
        
        # Pick random edge to remove
        u1, v1 = random.choice(list(edges))
        
        # Try several candidate replacements
        best_delta = float('inf')
        best_swap = None
        
        for _ in range(30):
            u2, v2 = random.choice(all_pairs_list)
            if (u2, v2) in edges or (v2, u2) in edges:
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            
            # Compute delta C4 efficiently
            # Nodes whose neighborhoods change: u1, v1, u2, v2
            # Edges affected: all edges incident to any of these 4 nodes
            
            # Old contributions of affected edges
            sum_old = 0
            affected = set()
            for node in [u1, v1, u2, v2]:
                for nb in adj[node]:
                    if node < nb:
                        if (node, nb) in edges:
                            affected.add((node, nb))
                    else:
                        if (nb, node) in edges:
                            affected.add((nb, node))
            
            affected.discard((u1, v1))
            
            for u, v in affected:
                common = len(adj[u] & adj[v])
                sum_old += binom2(common)
            
            # New contributions
            sum_new = 0
            
            # Helper to get new neighborhood
            def get_n_prime(x):
                n_x = set(adj[x])
                if x == u1:
                    n_x.discard(v1)
                elif x == v1:
                    n_x.discard(u1)
                if x == u2:
                    n_x.add(v2)
                elif x == v2:
                    n_x.add(u2)
                return n_x
            
            new_affected = set(affected)
            new_affected.add((u2, v2))
            
            for u, v in new_affected:
                n_u = get_n_prime(u)
                n_v = get_n_prime(v)
                common = len(n_u & n_v)
                sum_new += binom2(common)
            
            delta = sum_new - sum_old
            
            if delta < best_delta:
                best_delta = delta
                best_swap = (u2, v2)
        
        if best_swap is not None and best_delta < 0:
            u2, v2 = best_swap
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            adj[u2].add(v2)
            adj[v2].add(u2)
            
            edges.discard((u1, v1))
            edges.add((u2, v2))
            
            # Update adjacency matrix
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            current_c4 += best_delta // 4
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = set(edges)
                best_adj = [set(s) for s in adj]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        if time.time() > deadline - 5:
            break
    
    return list(best_edges)
