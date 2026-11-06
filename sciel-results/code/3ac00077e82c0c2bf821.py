import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s kill

    n = 40

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6.
        A new C6 must use the new edge (u, v).
        So we need a path of length 5 from u to v in the current graph:
        u - a - b - c - d - v, with a, b, c, d distinct and not u, v.
        """
        Nu = adj[u]
        Nv = adj[v]

        # a in N(u) \ {v}
        # d in N(v) \ {u}
        # a != d

        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a

            # d in N(v) \ {u, a}
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d

                Na = adj[a]
                Nd = adj[d]

                # b in N(a) \ {u, v, d}
                # c in N(d) \ {u, v, a}
                # such that b != c and (b, c) is an edge.
                
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # We need b in b_mask such that adj[b] & c_mask != 0
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def repair_graph(adj, edges_set):
        """
        Remove edges that create C6s until the graph is C6-free.
        Returns the repaired adjacency list and edge set.
        """
        # We need to check if the current graph has C6s.
        # Efficient way: iterate through all edges, if removing an edge breaks a C6, remove it?
        # No, we need to remove edges such that NO C6 exists.
        # A greedy repair: pick a random edge, check if it's part of a C6. If yes, remove it.
        # To check if edge (u,v) is part of a C6:
        # There exists u-a-b-c-d-v where a,b,c,d distinct, not u,v.
        # This is similar to would_create_c6 but checking existing edges.
        
        def edge_in_c6(adj, u, v):
            # Check if there is a path of length 5 from u to v not using edge (u,v)
            # This is exactly the logic of would_create_c6 but assuming (u,v) is present.
            # So we can reuse the logic by temporarily removing (u,v) from adj?
            # Or just run the same loops.
            
            Nu = adj[u]
            Nv = adj[v]
            
            a_mask = Nu & ~(1 << v)
            while a_mask:
                lsb_a = a_mask & -a_mask
                a = lsb_a.bit_length() - 1
                a_mask ^= lsb_a
                
                d_mask = Nv & ~(1 << u) & ~(1 << a)
                while d_mask:
                    lsb_d = d_mask & -d_mask
                    d = lsb_d.bit_length() - 1
                    d_mask ^= lsb_d
                    
                    Na = adj[a]
                    Nd = adj[d]
                    
                    b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                    c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                    
                    b_temp = b_mask
                    while b_temp:
                        lsb_b = b_temp & -b_temp
                        b = lsb_b.bit_length() - 1
                        b_temp ^= lsb_b
                        
                        if adj[b] & c_mask:
                            return True
            return False

        # Greedy repair loop
        # We iterate through a shuffled list of edges.
        # If an edge is in a C6, we remove it.
        # We repeat until no edges are removed in a pass or time runs out.
        
        # Optimization: Instead of checking all edges every time, we can just shuffle once
        # and process. If we remove an edge, it might break other C6s, so previously "bad" edges
        # might become "good". But a simple pass is a good start.
        
        # To be more robust, we loop multiple passes.
        
        # Initial shuffle
        edges_list = list(edges_set)
        random.shuffle(edges_list)
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Re-shuffle to avoid order bias
            random.shuffle(edges_list)
            
            for u, v in edges_list:
                if (u, v) not in edges_set and (v, u) not in edges_set:
                    continue
                
                if edge_in_c6(adj, u, v):
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edges_set.discard((u, v))
                    edges_set.discard((v, u))
                    changed = True
                    
        return adj, edges_set

    def dense_random_init(seed_val, p=0.6):
        """
        Initialize a dense random graph and repair it.
        """
        random.seed(seed_val)
        edges = set()
        adj = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    edges.add((i, j))
        
        adj = build_adj(edges)
        adj, edges = repair_graph(adj, edges)
        return adj, edges

    def k5_init():
        """
        Initialize with 8 K5s.
        """
        edges = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    edges.add((u, v))
        adj = build_adj(edges)
        return adj, edges

    def run_greedy(adj, edges_set, seed_val):
        """
        Add edges greedily to maximize count while maintaining C6-free property.
        """
        random.seed(seed_val)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 5:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_set.add((u, v))
        
        return adj, edges_set

    def local_search(adj, edges_set, seed_val):
        """
        Improve the edge set by adding edges and swapping.
        """
        random.seed(seed_val)
        
        current_edges = set(edges_set)
        
        # Pass 1: Add any possible non-edge
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 2:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
        
        # Pass 2: Try swaps (remove 1, add 2)
        removable = list(current_edges)
        random.shuffle(removable)
        
        # Re-generate non_edges list as it changed
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        consecutive_failures = 0
        max_failures = 100 # Allow more failures to explore more
        
        while time.time() < deadline - 2:
            if consecutive_failures > max_failures:
                break
                
            if not removable:
                break
            
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            # Save state
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            
            # Try to add 2 non-edges
            added = 0
            # We prioritize non-edges that are incident to u_rem or v_rem?
            # Let's just pick from the shuffled non_edges list, but maybe prioritize
            # those closer to the removed edge?
            # For now, random from non_edges.
            
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 2:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added >= 2:
                # Success: Edge count increased by (added - 1)
                consecutive_failures = 0
                # Update removable and non_edges for next iteration
                removable = list(current_edges)
                random.shuffle(removable)
                
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] & (1 << j)):
                            non_edges.append((i, j))
                random.shuffle(non_edges)
                
                # Continue searching for more improvements
            else:
                # Revert
                adj = old_adj
                current_edges = old_edges
                consecutive_failures += 1

        return adj, current_edges

    best_edges = set()
    best_count = 0
    
    # Strategy:
    # 1. Generate diverse initial candidates (Dense Random + Repair, K5)
    # 2. Run greedy on them
    # 3. Run local search on the best ones
    
    candidates = []
    
    # Phase 1: Diverse Initializations
    # We will try a mix of dense random (various p) and K5.
    
    # K5 is fast and reliable
    adj_k5, edges_k5 = k5_init()
    adj_k5, edges_k5 = run_greedy(adj_k5, edges_k5, 1)
    candidates.append((len(edges_k5), adj_k5, edges_k5))
    
    # Dense Random Repairs
    # p=0.6 is a good starting point for n=40
    # We try a few seeds to get diversity
    
    random_seeds = [10, 20, 30, 40, 50]
    for seed in random_seeds:
        if time.time() > deadline - 30: # Reserve time for LS
            break
            
        adj_rand, edges_rand = dense_random_init(seed, p=0.6)
        # Greedy refine
        adj_rand, edges_rand = run_greedy(adj_rand, edges_rand, seed + 100)
        candidates.append((len(edges_rand), adj_rand, edges_rand))
        
    # Sort candidates by size
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Local Search on top candidates
    # We only have time for a few local searches
    max_ls = 3
    for i in range(min(max_ls, len(candidates))):
        if time.time() > deadline - 10:
            break
            
        count, adj_c, edges_c = candidates[i]
        # Deep copy adj and edges for LS
        adj_ls = adj_c[:]
        edges_ls = set(edges_c)
        
        adj_ls, edges_ls = local_search(adj_ls, edges_ls, 1000 + i)
        
        if len(edges_ls) > len(best_edges):
            best_edges = edges_ls
            best_count = len(edges_ls)
            
    # Also check if any greedy was better than LS results (unlikely but possible if LS failed)
    for count, adj_c, edges_c in candidates:
        if len(edges_c) > len(best_edges):
            best_edges = edges_c
            best_count = len(edges_c)
            
    # Final Greedy Pass on the best candidate found
    if best_edges:
        adj_best = build_adj(best_edges)
        # One last quick greedy pass
        adj_best, best_edges = run_greedy(adj_best, best_edges, 9999)
        
    # Convert set of tuples to list of lists
    return [list(e) for e in best_edges]
