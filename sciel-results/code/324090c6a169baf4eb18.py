"""K4-free graph on 18 vertices minimizing max triangle-free induced subgraph."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n = 18
    
    # Precompute all 4-subsets for K4 check
    all_quads = list(combinations(range(n), 4))
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_k4_free(adj_mask):
        """Check K4-free using bitmasks. adj_mask[v] is bitmask of neighbors of v."""
        for quad in all_quads:
            a, b, c, d = quad
            if ((adj_mask[a] >> b) & 1) and ((adj_mask[a] >> c) & 1) and \
               ((adj_mask[a] >> d) & 1) and ((adj_mask[b] >> c) & 1) and \
               ((adj_mask[b] >> d) & 1) and ((adj_mask[c] >> d) & 1):
                return False
        return True
    
    def max_triangle_free_induced(adj_mask):
        """Find largest triangle-free induced subgraph using branch and bound with bitmasks."""
        best = [0]
        verts = list(range(n))
        
        # Precompute for each vertex: which other vertices form a triangle with it
        # tri[v] = set of w such that (v,w) is an edge and they share a common neighbor
        # Actually, for triangle-free check: a set S is triangle-free iff no triangle is fully in S
        # We need: for any three vertices in S, not all three pairs are edges
        
        # Better approach: build up triangle-free set
        # A vertex v can be added to set S iff v's neighbors in S form an independent set
        # i.e., no two neighbors of v in S are adjacent
        
        # Precompute: for each pair (u,v) that is an edge, the set of common neighbors
        # This helps check: adding v to S creates triangle iff there exist u,w in S 
        # with (u,v) in E, (w,v) in E, (u,w) in E
        
        # Fast check: for vertex v, its neighbors in S must be an independent set
        # Precompute adjacency as bitmask
        adj = adj_mask  # list of bitmasks
        
        def can_add(S_mask, v):
            """Check if v can be added to triangle-free set S (represented as bitmask)."""
            # v's neighbors in S
            v_neigh_in_S = adj[v] & S_mask
            # Check if v_neigh_in_S is independent: no edge between any two vertices in v_neigh_in_S
            # For each u in v_neigh_in_S, check if adj[u] & v_neigh_in_S has any bit other than u
            temp = v_neigh_in_S
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                if adj[u] & v_neigh_in_S & ~lsb:
                    return False
                temp ^= lsb
            return True
        
        def search(S_mask, candidates_mask, depth):
            nonlocal best
            if time.time() > deadline - 2:
                return
            # Upper bound: current size + number of candidates
            if S_mask.bit_count() + candidates_mask.bit_count() <= best[0]:
                return
            if candidates_mask == 0:
                if S_mask.bit_count() > best[0]:
                    best[0] = S_mask.bit_count()
                return
            
            # Pick a candidate vertex
            lsb = candidates_mask & (-candidates_mask)
            v = lsb.bit_length() - 1
            remaining = candidates_mask ^ lsb
            
            # Option 1: include v
            if can_add(S_mask, v):
                new_S = S_mask | lsb
                # Filter candidates: only those compatible with new_S
                new_cands = 0
                temp = remaining
                while temp:
                    lsb2 = temp & (-temp)
                    w = lsb2.bit_length() - 1
                    if can_add(new_S, w):
                        new_cands |= lsb2
                    temp ^= lsb2
                search(new_S, new_cands, depth + 1)
            
            # Option 2: exclude v
            search(S_mask, remaining, depth + 1)
        
        all_mask = (1 << n) - 1
        search(0, all_mask, 0)
        return best[0]
    
    def random_k4_free_graph(rng):
        """Generate a random K4-free graph by adding edges one at a time if K4-free."""
        adj = [0] * n
        # Start with random permutation of edges
        edges = list(all_pairs)
        rng.shuffle(edges)
        for u, v in edges:
            # Try adding edge (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not is_k4_free(adj):
                # Remove it
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj
    
    def local_search(adj, rng):
        """Local search: try edge additions and removals to improve score."""
        current_score = max_triangle_free_induced(adj)
        best_adj = adj[:]
        best_score = current_score
        
        while time.time() < deadline - 5:
            improved = False
            # Try adding random missing edges
            missing_edges = []
            for u, v in all_pairs:
                if not (adj[u] >> v) & 1:
                    missing_edges.append((u, v))
            
            if missing_edges:
                for _ in range(min(20, len(missing_edges))):
                    if time.time() > deadline - 5:
                        break
                    u, v = rng.choice(missing_edges)
                    new_adj = adj[:]
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    if is_k4_free(new_adj):
                        score = max_triangle_free_induced(new_adj)
                        if score < best_score:
                            best_score = score
                            best_adj = new_adj
                            improved = True
            
            # Try removing random existing edges
            existing_edges = []
            for u, v in all_pairs:
                if (adj[u] >> v) & 1:
                    existing_edges.append((u, v))
            
            if existing_edges:
                for _ in range(min(20, len(existing_edges))):
                    if time.time() > deadline - 5:
                        break
                    u, v = rng.choice(existing_edges)
                    new_adj = adj[:]
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                    score = max_triangle_free_induced(new_adj)
                    if score < best_score:
                        best_score = score
                        best_adj = new_adj
                        improved = True
            
            if not improved:
                # Perturbation: flip several random edges
                for _ in range(rng.randint(3, 8)):
                    u, v = rng.choice(all_pairs)
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        new_adj = adj[:]
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
                        if is_k4_free(new_adj):
                            adj = new_adj
                        # else keep as is
            else:
                adj = best_adj[:]
        
        return best_adj, best_score
    
    # Main: try multiple initializations
    rng = random.Random(42)
    
    # Start with tripartite K_{6,6,6}
    adj_triple = [0] * n
    parts = [set(range(0, 6)), set(range(6, 12)), set(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj_triple[a] |= (1 << b)
                    adj_triple[b] |= (1 << a)
    
    best_adj = adj_triple[:]
    best_score = max_triangle_free_induced(adj_triple)
    
    # Try random K4-free graphs
    for restart in range(50):
        if time.time() > deadline - 20:
            break
        adj_init = random_k4_free_graph(rng)
        score = max_triangle_free_induced(adj_init)
        if score < best_score:
            best_score = score
            best_adj = adj_init[:]
        
        # Local search from this start
        if time.time() < deadline - 10:
            adj_ls, score_ls = local_search(adj_init[:], rng)
            if score_ls < best_score:
                best_score = score_ls
                best_adj = adj_ls[:]
    
    # Final local search from best found
    if time.time() < deadline - 5:
        adj_final, score_final = local_search(best_adj[:], rng)
        if score_final < best_score:
            best_score = score_final
            best_adj = adj_final[:]
    
    # Convert back to edge list
    result = []
    for u, v in all_pairs:
        if (best_adj[u] >> v) & 1:
            result.append([u, v])
    return result
