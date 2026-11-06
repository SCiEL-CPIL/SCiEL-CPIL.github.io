import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    # Partition into A (0-9) and B (10-19)
    A = list(range(10))
    B = list(range(10, 20))
    
    # 1. Add ALL intra-part edges
    # These cannot form K3,3 with each other, and K3,3 requires 3 from A and 3 from B.
    # Intra edges don't contribute to the bipartite cross-edges of a K3,3 subgraph.
    adj = [0] * n
    for i in range(10):
        for j in range(i+1, 10):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    for i in range(10, 20):
        for j in range(i+1, 20):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    # 2. Construct Bipartite Edges (A-B)
    # We want to maximize edges between A and B without creating K3,3.
    # A K3,3 in the bipartite subgraph consists of 3 vertices in A and 3 in B with all 9 edges present.
    
    # Spectral Initialization for Bipartite Part:
    # Generate a random 10x10 matrix
    # Use its principal eigenvector to weight vertices
    # This creates a "soft" structure where we prioritize edges between vertices with similar "spectral signatures"
    
    def generate_bipartite_edges_spectral(adj, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # Random matrix for spectral features
        M = np.random.rand(10, 10)
        # Symmetrize to get real eigenvalues
        M = (M + M.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(M)
        
        # Use the top eigenvector (last column of eigvecs for eigh)
        v = eigvecs[:, -1]
        
        # Normalize v to have mean 0? Or just use as is.
        # We want to create a bipartite graph.
        # Let's try to create a "block" structure or "cluster" structure.
        # If v_i is positive for A and negative for B, we might connect same signs?
        # Actually, let's just use v to order the edges.
        
        # Candidate edges: all (a, b)
        candidates = []
        for a in A:
            for b in B:
                # Score based on spectral similarity
                # If we want to avoid K3,3, we want to avoid dense subgraphs.
                # A rank-1 structure (v_a * v_b) creates a complete bipartite graph if threshold is low.
                # Let's use the score to *order* the greedy addition.
                score = v[a] * v[b - 10] # v is indexed 0-9 for A, but B is 10-19 in adj.
                # Wait, v is 10x10. v[a] is for A. What for B?
                # We generated M from scratch. Let's generate two vectors?
                # Or just use a random ordering if spectral doesn't help directly.
                # Let's try: Score = random() for baseline, or use v to create "cliques" in A and B?
                # If we connect A_clique to B_clique, we get K_k,k.
                # We want to AVOID that.
                # So maybe connect A_high to B_low?
                # Let's try: score = v[a] - v[b-10] (if we map B to 0-9)
                
                # Let's just use a random shuffle but biased by a spectral "potential"
                # Potential: u_a = v[a], w_b = v[b-10]
                # Connect if u_a * w_b > threshold?
                # This creates a bipartite graph that is a "threshold graph".
                # Threshold graphs are K_3,3-free?
                # A bipartite threshold graph is a chain graph.
                # Chain graphs are K_2,2-free, hence K_3,3-free.
                # Max edges in chain graph on 10+10:
                # Degrees must be non-decreasing.
                # Max edges is sum of min(d_i, 10) where d_i is degree.
                # To maximize, make it as dense as possible while remaining a chain graph.
                # A complete bipartite graph K_10,10 is a chain graph? No, K_10,10 has K_2,2.
                # Chain graphs are K_2,2-free.
                # So max edges in chain graph is 10*1 + 10*2 ... no.
                # Chain graph degrees: 1, 1, ..., 1, 2, 2, ..., 10.
                # Sum = 10*10 = 100? No.
                # If degrees are 1..10, sum is 55.
                # If we can have higher degrees?
                # In a bipartite graph, if it is K_2,2-free, it is a chain graph.
                # Max edges in K_2,2-free bipartite graph with parts 10,10 is 10*10? No.
                # It's the sum of the degrees.
                # The maximum number of edges in a K_2,2-free bipartite graph with n vertices in each part is n^2? No.
                # It is at most 2n - 1? No.
                # It is at most n * sqrt(n)? No.
                # Actually, a complete bipartite graph K_{n, 1} is K_2,2-free. Edges = n.
                # K_{n, k} is K_2,2-free if k=1.
                # If we take a "staircase" graph:
                # Row 1: 1 edge
                # Row 2: 2 edges
                # ...
                # Row 10: 10 edges
                # Total = 55.
                # This is low.
                
                # We want K_3,3-free, not K_2,2-free.
                # So we can be denser.
                # Let's just use Greedy with Spectral Ordering.
                
                # For now, let's just randomize and let Local Search do the work,
                # but start with a "dense" bipartite graph and prune.
                # Or start with empty and add.
                
                # Let's try a specific spectral construction:
                # Use the eigenvector to define a "rank" for each vertex.
                # Sort A by v[a]. Sort B by v[b-10].
                # Connect A_i to B_j if i + j < 10? (Triangle)
                # This creates a K_2,2-free graph (chain graph).
                # Edges = 55.
                # This is safe but maybe suboptimal.
                
                # Let's try a "Quadratic Residue" like construction if available?
                # No, stick to spectral.
                
                # Let's try: Connect A[i] to B[j] if (i * j) % 10 < 5?
                # This is a heuristic.
                
                # Let's just use a random initialization for bipartite, 
                # but ensure we don't start with K3,3.
                # Start with empty bipartite, add edges greedily.
                
                pass
        return adj

    # Since spectral bipartite construction is tricky to guarantee high density without K3,3,
    # and we already have 90 edges from intra-part,
    # we will use a fast Greedy Bipartite Add.
    
    def can_add_bipartite(adj, u, v):
        # u in A, v in B
        if adj[u] & (1 << v):
            return False
        
        # Check if adding u-v creates K3,3
        # K3,3 requires 3 common neighbors in the *other* part?
        # No, K3,3 in bipartite graph:
        # 3 vertices in A, 3 in B.
        # If we add u (in A) to v (in B):
        # We need to check if there exist 2 other vertices a1, a2 in A (neighbors of v)
        # and 2 other vertices b1, b2 in B (neighbors of u)
        # such that a1, a2 are connected to b1, b2.
        
        # Neighbors of v in A: N_A(v)
        # Neighbors of u in B: N_B(u)
        
        N_A_v = [a for a in A if (adj[a] & (1 << v)) and a != u] # u is not in A? u is in A.
        # Wait, u is in A. v is in B.
        # N_A(v) are vertices in A connected to v.
        N_A_v = [a for a in A if (adj[a] & (1 << v))]
        
        N_B_u = [b for b in B if (adj[b] & (1 << u))]
        
        # If we add u-v, u becomes a neighbor of v.
        # So the new set of A-neighbors of v is N_A_v U {u}.
        # The new set of B-neighbors of u is N_B_u U {v}.
        
        # We need to check if there is a K3,3 involving u and v.
        # Case 1: u and v are in the K3,3.
        # We need 2 other A-vertices (a1, a2) that are connected to v.
        # And 2 other B-vertices (b1, b2) that are connected to u.
        # And a1, a2 must be connected to b1, b2.
        
        # Let's just check all pairs of (a1, a2) from N_A_v and (b1, b2) from N_B_u.
        # If a1, a2 are both connected to b1, b2, then adding u-v completes the K3,3 
        # (u, a1, a2) x (v, b1, b2).
        
        # Optimization:
        # For each pair a1, a2 in N_A_v:
        #   Common neighbors of a1, a2 in B: C12
        #   If |C12 & N_B_u| >= 2, then we have a K3,3.
        
        if len(N_A_v) < 2 or len(N_B_u) < 2:
            return True
            
        # Precompute common neighbors in B for pairs in A?
        # Or just loop.
        # N_A_v size <= 10. N_B_u size <= 10.
        
        for i in range(len(N_A_v)):
            a1 = N_A_v[i]
            for j in range(i+1, len(N_A_v)):
                a2 = N_A_v[j]
                
                # Common neighbors of a1, a2 in B
                common_b = [b for b in B if (adj[a1] & (1 << b)) and (adj[a2] & (1 << b))]
                
                # Check if at least 2 of these are in N_B_u
                count = 0
                for b in common_b:
                    if b in N_B_u:
                        count += 1
                        if count >= 2:
                            return False
                            
        # Case 2: K3,3 does NOT involve u or v as part of the 3x3 core?
        # No, adding an edge only creates new K3,3s that include that edge.
        # So checking K3,3s containing (u, v) is sufficient.
        
        return True

    # Greedy Bipartite Construction
    # Use spectral ordering to prioritize edges
    # Generate spectral scores
    np.random.seed(42)
    M = np.random.rand(10, 10)
    M = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M)
    v_vec = eigvecs[:, -1]
    
    # Score for edge (a, b):
    # We want to avoid K3,3.
    # Let's try to maximize "diversity" of connections.
    # Score = v_vec[a] * v_vec[b-10]
    
    bipartite_edges = []
    for a in A:
        for b in B:
            score = v_vec[a] * v_vec[b-10]
            bipartite_edges.append((score, a, b))
            
    bipartite_edges.sort(reverse=True)
    
    # Add edges greedily
    for score, a, b in bipartite_edges:
        if time.time() > deadline - 10:
            break
        if can_add_bipartite(adj, a, b):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            
    # 3. Local Search
    # Standard local search on full graph
    
    def can_add_full(adj, u, v):
        if adj[u] & (1 << v):
            return False
        for a in range(n):
            if a == u or a == v: continue
            for b in range(a + 1, n):
                if b == u or b == v: continue
                if (adj[a] & (1 << v)) and (adj[b] & (1 << v)):
                    if (adj[u] & adj[a] & adj[b]).bit_count() > 1:
                        return False
                if (adj[a] & (1 << u)) and (adj[b] & (1 << u)):
                    if (adj[v] & adj[a] & adj[b]).bit_count() > 1:
                        return False
        return True

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def local_search(adj):
        current_edges = sum(x.bit_count() for x in adj) // 2
        best_adj = adj[:]
        best_edges = current_edges
        
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        steps = 0
        max_steps = 5000
        while steps < max_steps and time.time() < deadline - 2:
            steps += 1
            if not missing:
                break
            idx = random.randint(0, len(missing) - 1)
            u, v = missing[idx]
            
            if can_add_full(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                missing.pop(idx)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Kick: remove random edge, try to add
                existing = get_edges(adj)
                if not existing: break
                ra, rb = random.choice(existing)
                adj[ra] &= ~(1 << rb)
                adj[rb] &= ~(1 << ra)
                current_edges -= 1
                missing.append((ra, rb))
                
                # Greedy add
                random.shuffle(missing)
                for me in missing[:]:
                    mu, mv = me
                    if can_add_full(adj, mu, mv):
                        adj[mu] |= (1 << mv)
                        adj[mv] |= (1 << mu)
                        current_edges += 1
                        missing.remove(me)
                        
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                
                if current_edges >= best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                    
        return best_adj, best_edges

    # Run Local Search
    adj, edges = local_search(adj)
    
    # Multiple Restarts for Bipartite Part?
    # The intra-part edges are fixed and optimal.
    # Only the bipartite part needs optimization.
    # We can restart the bipartite construction with different seeds.
    
    best_adj = adj[:]
    best_edges = edges
    
    for seed in range(1, 100):
        if time.time() > deadline - 5:
            break
            
        # Reset to base (Intra-part only)
        base_adj = [0] * n
        for i in range(10):
            for j in range(i+1, 10):
                base_adj[i] |= (1 << j)
                base_adj[j] |= (1 << i)
        for i in range(10, 20):
            for j in range(i+1, 20):
                base_adj[i] |= (1 << j)
                base_adj[j] |= (1 << i)
                
        # New Spectral Bipartite
        np.random.seed(seed)
        M = np.random.rand(10, 10)
        M = (M + M.T) / 2
        eigvals, eigvecs = np.linalg.eigh(M)
        v_vec = eigvecs[:, -1]
        
        bipartite_edges = []
        for a in A:
            for b in B:
                score = v_vec[a] * v_vec[b-10]
                bipartite_edges.append((score, a, b))
        bipartite_edges.sort(reverse=True)
        
        adj = base_adj[:]
        for score, a, b in bipartite_edges:
            if can_add_bipartite(adj, a, b):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
        # Local Search
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
