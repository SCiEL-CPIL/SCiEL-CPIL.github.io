import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    n = 20
    vertices = list(range(n))
    
    # Precompute triples for validation
    triples = list(combinations(vertices, 3))
    
    # --- Algebraic Construction: Polarity Graph on F_19 + 1 ---
    # Vertices 0..18 correspond to elements of F_19.
    # Vertex 19 is an extra vertex.
    # Edge (u, v) for u,v in 0..18 if (v-u)^2 is a quadratic residue mod 19 (and u!=v).
    # This graph is triangle-free (hence K3,3-free) and regular.
    # Q = quadratic residues mod 19.
    # 19 is prime. Q = {x^2 mod 19 | x in 1..18}
    # Q = {1, 4, 9, 16, 6, 17, 11, 7, 5, 5, 7, 11, 17, 6, 16, 9, 4, 1}
    # Unique: {1, 4, 5, 6, 7, 9, 11, 16, 17}
    # Size 9. So degree is 9. Total edges 20*9/2 = 90? 
    # Wait, for vertex 19, we need to define edges.
    # Let's just connect 19 to all vertices that are "safe".
    # Actually, simpler: Start with the F_19 polarity graph on 0..18.
    # It has 19*9/2 = 85.5? No, 19 vertices, degree 9. 19*9/2 is not integer.
    # In F_p with p=4k+3, -1 is non-residue. x^2=y^2 => x=+-y.
    # Number of residues is (p-1)/2 = 9.
    # For each u, v is connected if v-u in Q or u-v in Q.
    # Since -1 is non-residue in F_19 (19=4*4+3), Q and -Q are disjoint.
    # So (v-u) in Q implies (u-v) not in Q.
    # Thus the graph is directed? No, we want undirected.
    # Standard Polarity Graph: u~v if (v-u)^2 in Q OR (u-v)^2 in Q.
    # Since Q is a subgroup, if d in Q, -d not in Q.
    # So for any pair u!=v, exactly one of (v-u) or (u-v) is in Q?
    # No. d and -d. One is in Q, one is not.
    # So yes, for any u!=v, exactly one direction is "residue".
    # So the undirected graph where u~v if (v-u) in Q is well-defined and regular of degree 9.
    # Total edges: 19 * 9 / 2 = 85.5 -> Wait.
    # Sum of degrees = 19*9 = 171. 171/2 = 85.5. Impossible.
    # Ah, 19 is odd. Sum of degrees must be even.
    # Let's re-evaluate.
    # In F_p, p=4k+3. Q has size (p-1)/2.
    # For u, neighbors are u+Q. Size (p-1)/2 = 9.
    # Total edges = 19*9/2. 171/2. 
    # Wait, is the graph symmetric?
    # u~v iff v-u in Q.
    # v~u iff u-v in Q.
    # If v-u in Q, then u-v = -(v-u). Since -1 notin Q, u-v notin Q.
    # So u~v is NOT equivalent to v~u.
    # This defines a tournament-like structure or a directed graph?
    # No, a simple graph must be undirected.
    # We can define u~v if (v-u) in Q OR (u-v) in Q.
    # Since exactly one is in Q, this condition is always true for u!=v.
    # That would be a complete graph K_19. K_19 has K3,3. Bad.
    
    # Correct Polarity Graph Construction for K3,3-free:
    # Use a quadratic form.
    # Let V = F_p x {0,1}? No, n=20.
    # Let's use the known construction for ex(n, K_{3,3}).
    # A common high-density K3,3-free graph is derived from a finite projective plane or similar.
    # However, for n=20, we can use a simpler heuristic:
    # Start with a random regular graph of degree ~7-8 and optimize.
    # Or better: Use the "Polarity Graph" of the finite field F_19 properly.
    # Actually, the graph defined by u~v iff (v-u)^2 is a square in F_p is NOT undirected.
    # To make it undirected, we can take the union of u~v iff (v-u) in Q and u~v iff (u-v) in Q?
    # That's K_p.
    
    # Alternative Algebraic Construction:
    # Vertices are elements of F_19.
    # Edge (u,v) if (v-u) is a quadratic residue mod 19.
    # As established, this is not symmetric.
    # But we can define an undirected graph by: u~v if (v-u) in Q OR (u-v) in Q.
    # Wait, if I define the graph G where u~v iff (v-u) in Q, it's a directed graph.
    # If I symmetrize it: u~v in G_sym if u~v in G or v~u in G.
    # Since for any u!=v, exactly one of (v-u) or (u-v) is in Q, G_sym is K_19.
    # So that doesn't work.
    
    # Let's use a different algebraic structure.
    # The Clebsch graph? No, n=16.
    # The Schläfli graph? n=27.
    # The Hoffman-Singleton? n=50.
    
    # Let's stick to a robust Random + Fast Local Search, but improve the initialization.
    # Use a "greedy construction" from scratch.
    # 1. Start with empty graph.
    # 2. Add edges in random order if they don't create K3,3.
    # This guarantees validity at every step.
    # 3. Then perform local search (swaps and additions).
    
    # The previous program did random init + repair.
    # Repair is slow because it checks all triples.
    # Greedy add is fast because it only checks affected triples.
    
    def popcount(x):
        return x.bit_count()
        
    # Precompute for fast can_add
    # For each vertex v, list of triples containing v
    vertex_to_triples = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(triples):
        vertex_to_triples[a].append(idx)
        vertex_to_triples[b].append(idx)
        vertex_to_triples[c].append(idx)
        
    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Only checks triples containing u or v.
        """
        bit_v = 1 << v
        bit_u = 1 << u
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        # Check triples containing u
        for t_idx in vertex_to_triples[u]:
            a, b, c = triples[t_idx]
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == u: ma = new_u
            elif a == v: ma = new_v
            if b == u: mb = new_u
            elif b == v: mb = new_v
            if c == u: mc = new_u
            elif c == v: mc = new_v
            
            if popcount(ma & mb & mc) > 2:
                return False
        
        # Check triples containing v (that do not contain u)
        for t_idx in vertex_to_triples[v]:
            a, b, c = triples[t_idx]
            if u in (a, b, c):
                continue
            
            ma = adj_mask[a]
            mb = adj_mask[b]
            mc = adj_mask[c]
            
            if a == v: ma = new_v
            if b == v: mb = new_v
            if c == v: mc = new_v
            
            if popcount(ma & mb & mc) > 2:
                return False
        
        return True

    def is_k33_free(adj_mask):
        for t in triples:
            u, v, w = t
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return False
        return True

    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

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

    def local_search(adj_mask, max_iters=1000):
        # Phase 1: Greedy Add
        adj_mask = greedy_add(adj_mask)
        
        # Phase 2: Local Search (Swaps)
        # We try to swap an existing edge for a missing edge, then greedily add more.
        # This allows escaping local optima where no single addition works.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Collect missing and existing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj_mask[i] & (1 << j))]
            existing = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_mask[i] & (1 << j)]
            
            if not missing:
                break
            
            # Try to add directly first
            improved = False
            # Shuffle for randomness
            random.shuffle(missing)
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no direct addition, try swaps
            if not existing:
                break
                
            random.shuffle(existing)
            random.shuffle(missing)
            
            # Try a few random swaps
            swapped = False
            for k in range(min(50, len(existing), len(missing))):
                if time.time() > deadline:
                    break
                r_u, r_v = existing[k]
                a_u, a_v = missing[k]
                
                # Remove r_u-r_v
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                # Add a_u-a_v
                adj_mask[a_u] |= (1 << a_v)
                adj_mask[a_v] |= (1 << a_u)
                
                # Check validity
                if is_k33_free(adj_mask):
                    # If valid, try to add more edges
                    adj_mask = greedy_add(adj_mask)
                    swapped = True
                    break
                else:
                    # Revert
                    adj_mask[r_u] |= (1 << r_v)
                    adj_mask[r_v] |= (1 << r_u)
                    adj_mask[a_u] &= ~(1 << a_v)
                    adj_mask[a_v] &= ~(1 << a_u)
            
            if not swapped:
                # Stuck? Try removing a random edge and re-adding greedily (simulated annealing style)
                # This is risky but can escape.
                if existing:
                    r_u, r_v = existing[0]
                    adj_mask[r_u] &= ~(1 << r_v)
                    adj_mask[r_v] &= ~(1 << r_u)
                    adj_mask = greedy_add(adj_mask)
                    
        return adj_mask

    best_adj = [0] * n
    best_edges = 0
    rng = random.Random(42)
    
    # Strategy: Multiple restarts with different densities
    # But since we start empty and greedy add, density doesn't matter for init.
    # We can vary the *order* of greedy addition or use random shuffles in local search.
    
    # To diversify, we can start with a random subset of edges, then greedy add.
    # Or just run local search multiple times with different random seeds/shuffles.
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        
        # Optional: Random initial seed
        # density = 0.2
        # for i in range(n):
        #     for j in range(i+1, n):
        #         if rng.random() < density:
        #             adj[i] |= (1 << j)
        #             adj[j] |= (1 << i)
        #             # Ensure validity? No, greedy_add will handle it?
        #             # No, greedy_add only ADDS. If init is invalid, it stays invalid.
        #             # So we must start valid. Empty is valid.
        
        # Just start empty and let greedy_add do the work.
        # The randomness comes from local_search shuffles.
        # To get different results, we can shuffle the vertex order or edge order in greedy_add?
        # Greedy_add is deterministic.
        # So we need random init or random swaps.
        
        # Let's do a random valid init:
        # Add random edges one by one, checking validity.
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        for u, v in all_pairs:
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Now optimize
        adj = local_search(adj, max_iters=500)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
            
    if best_adj is None:
        best_adj = [0] * n
        
    edges = mask_to_edges(best_adj)
    return edges
