"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    n = 40
    all_mask = (1 << n) - 1

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6.
        """
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
                
                # Vectorized check: for each b in b_mask, check if adj[b] & c_mask
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

    def find_one_c6(adj):
        """
        Find one C6 in the graph. Returns the list of 6 vertices or None.
        Path: u - a - b - c - d - v where u, v are non-adjacent.
        """
        # Iterate over non-adjacent pairs (u, v)
        for u in range(n):
            nu = adj[u]
            # v candidates: not u, not in N(u)
            v_mask = ~nu & ~(1 << u) & all_mask
            
            while v_mask:
                lsb_v = v_mask & -v_mask
                v = lsb_v.bit_length() - 1
                v_mask ^= lsb_v
                
                nv = adj[v]
                
                # a in N(u), d in N(v), a != d
                a_mask = nu
                while a_mask:
                    lsb_a = a_mask & -a_mask
                    a = lsb_a.bit_length() - 1
                    a_mask ^= lsb_a
                    
                    d_mask = nv & ~(1 << a)
                    while d_mask:
                        lsb_d = d_mask & -d_mask
                        d = lsb_d.bit_length() - 1
                        d_mask ^= lsb_d
                        
                        # b in N(a) \ {u, v, d, a}
                        # c in N(d) \ {u, v, a, d}
                        # b != c
                        # b ~ c
                        
                        b_cand = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << d) & ~(1 << a)
                        c_cand = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                        
                        # Check if there is an edge between b and c
                        b_temp = b_cand
                        while b_temp:
                            lsb_b = b_temp & -b_temp
                            b = lsb_b.bit_length() - 1
                            b_temp ^= lsb_b
                            
                            if adj[b] & c_cand:
                                # Found C6: u-a-b-c-d-v
                                # Wait, we need to find c.
                                # c is in c_cand and adjacent to b.
                                c_mask = adj[b] & c_cand
                                if c_mask:
                                    lsb_c = c_mask & -c_mask
                                    c = lsb_c.bit_length() - 1
                                    return [u, a, b, c, d, v]
        return None

    def prune_graph(adj, edges_set, max_iters=500):
        """
        Remove edges from C6s until the graph is C6-free.
        Returns the pruned edge set.
        """
        edges = set(edges_set)
        changed = True
        iters = 0
        while changed and iters < max_iters:
            changed = False
            iters += 1
            c6 = find_one_c6(adj)
            if c6:
                # Remove a random edge from the C6
                # The C6 is [u, a, b, c, d, v]
                # Edges are (u,a), (a,b), (b,c), (c,d), (d,v), (v,u)
                # Note: (v,u) is not in the graph because u,v are non-adjacent in the search?
                # Wait, find_one_c6 searches for u,v non-adjacent.
                # So the cycle is u-a-b-c-d-v-u? No, v-u is not an edge.
                # The cycle is u-a-b-c-d-v ... wait.
                # A C6 is a cycle of length 6.
                # If u and v are non-adjacent, the path u-a-b-c-d-v is length 5.
                # To form a cycle, we need an edge (v,u). But u,v are non-adjacent.
                # So this logic is flawed.
                
                # Correct C6 detection:
                # A C6 is v1-v2-v3-v4-v5-v6-v1.
                # All edges exist.
                # My find_one_c6 was looking for a path of length 5 between non-adjacent nodes.
                # That's not a C6. That's just a path.
                # A C6 requires the closing edge.
                
                # Let's re-implement find_one_c6 correctly.
                # Or better, just use the existing would_create_c6 logic in a different way?
                # No, we need to find an existing C6 to break it.
                
                # Let's try a different approach for pruning:
                # Randomly remove edges until no C6 is found.
                # To check if a C6 exists, we can just try to add all possible edges? No.
                
                # Let's implement a proper find_c6.
                # C6: v0-v1-v2-v3-v4-v5-v0
                # Iterate v0, v1 (adjacent), v2 (adjacent to v1, not v0), v3 (adjacent to v2, not v0,v1),
                # v4 (adjacent to v3, not v0,v1,v2), v5 (adjacent to v4, not v0..v3, and adjacent to v0).
                
                # This is O(n^6) worst case, but with bitmasks it might be okay for small n.
                # However, we only need to find ONE.
                
                # Let's try to find one by random sampling first.
                # Pick 6 random distinct vertices. Check if they form a C6.
                # If yes, remove an edge.
                
                # If random sampling fails, do a structured search.
                
                # For now, let's just break the loop and use a simpler pruning strategy:
                # Iterate through edges. If removing an edge reduces the number of C6s?
                # Too expensive.
                
                # Let's go back to the "Dense Random Init + Prune" idea from Inspiration 1.
                # The inspiration 1 code had a `find_c6` that was cut off.
                # Let's implement a robust `find_c6`.
                
                # Actually, the current program's `dense_random_init` works by:
                # 1. Generating random edges.
                # 2. Iteratively checking each edge: remove it, see if re-adding creates C6.
                # This is O(E * C6_check).
                # C6_check is O(n^3) or so.
                # This is slow but works.
                
                # Let's stick to the current program's logic for pruning but maybe optimize it?
                # Or just use the current program's `dense_random_init` which is already there.
                
                # I will revert to using the existing `dense_random_init` from the current program
                # but I will integrate the "BQP/Spectral" idea by trying a slightly higher density
                # or different probabilities.
                
                # Actually, the directive says "Combine... take structural idea of one and refinement machinery of the other".
                # Current program: Greedy + Local Search + Perturbation.
                # Inspiration 1: BQP/Spectral (which effectively fell back to Dense Random + Prune).
                
                # Let's keep the Current Program's structure (Greedy/Local Search) but improve the initialization
                # using the "Dense Random Init" idea more aggressively, and perhaps add a "Simulated Annealing"
                # style acceptance in the local search (refinement machinery).
                
                # Let's refine the `local_search` to accept slightly worse solutions (lower edge count)
                # to escape local optima, then recover. This is a common refinement.
                
                # But first, let's fix the `find_one_c6` logic if I want to use it.
                # Actually, I don't strictly need `find_one_c6` if I use the `would_create_c6` based pruning.
                # The current `dense_random_init` is fine.
                
                # Let's just return the current state and rely on the main loop.
                pass
            else:
                break
        return edges

    # We will discard the broken find_one_c6/prune_graph above and stick to the working dense_random_init
    # but we will enhance the local search with simulated annealing.

    def dense_random_init(seed_val, prob=0.65):
        """
        Create a dense random initialization and prune C6s.
        """
        random.seed(seed_val)
        adj = [0] * n
        edges = set()
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < prob:
                    edges.add((i, j))
        
        adj = build_adj(edges)
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            for u, v in list(edges):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                if would_create_c6(adj, u, v):
                    to_remove.append((u, v))
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            for u, v in to_remove:
                edges.remove((u, v))
                changed = True
        
        return edges

    def run_greedy(start_edges, seed_val):
        random.seed(seed_val)
        
        adj = build_adj(start_edges)
        edges = set(start_edges)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 2:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges

    def local_search_sa(edges_set, seed_val, T_start=1.0, T_end=0.1, max_iter=200):
        """
        Simulated Annealing Local Search.
        State: edge set.
        Move: Add an edge, remove an edge, or swap (remove 1, add 1 or 2).
        Objective: Maximize edge count.
        Accept: Always accept improvement. Accept worse with probability exp((new-old)/T).
        """
        random.seed(seed_val)
        
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        current_score = len(current_edges)
        
        # Precompute non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        # We will perform a fixed number of iterations
        # Temperature schedule
        T = T_start
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.5 and non_edges:
                # Add an edge
                idx = random.randint(0, len(non_edges) - 1)
                u, v = non_edges[idx]
                
                # Check if valid
                if not would_create_c6(adj, u, v):
                    # Apply move
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    new_score = len(current_edges)
                    
                    # Always accept improvement (score increases by 1)
                    # Update non_edges list (remove this edge)
                    non_edges.pop(idx)
                    current_score = new_score
                else:
                    # Move not valid, so we can't apply it. 
                    # In SA, we need a valid state. So we can't "accept" an invalid state.
                    # So we just skip this move? Or try a different move?
                    # For simplicity, if the move is invalid, we don't apply it.
                    # This limits the search space to valid graphs.
                    # To escape local optima, we need to remove edges.
                    pass
            
            elif move_type < 0.8 and current_edges:
                # Remove an edge
                idx = random.randint(0, len(current_edges) - 1)
                e_rem = list(current_edges)[idx] # Inefficient, but okay for small n
                u, v = e_rem
                
                # Apply move
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges.remove(e_rem)
                new_score = len(current_edges)
                
                # Score decreases by 1. 
                # Accept with probability exp(-1/T)
                if random.random() < np.exp(-1.0 / T):
                    # Accepted
                    current_score = new_score
                    # Add to non_edges
                    non_edges.append(e_rem)
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add(e_rem)
            
            else:
                # Swap: Remove 1, Add 1
                if current_edges and non_edges:
                    idx_rem = random.randint(0, len(current_edges) - 1)
                    e_rem = list(current_edges)[idx_rem]
                    u_rem, v_rem = e_rem
                    
                    idx_add = random.randint(0, len(non_edges) - 1)
                    e_add = non_edges[idx_add]
                    u_add, v_add = e_add
                    
                    # Remove e_rem
                    adj[u_rem] &= ~(1 << v_rem)
                    adj[v_rem] &= ~(1 << u_rem)
                    current_edges.remove(e_rem)
                    
                    # Check if e_add is valid
                    if not would_create_c6(adj, u_add, v_add):
                        # Add e_add
                        adj[u_add] |= (1 << v_add)
                        adj[v_add] |= (1 << u_add)
                        current_edges.add(e_add)
                        
                        # Score stays same (remove 1, add 1)
                        # Accept with probability exp(0/T) = 1? 
                        # Or maybe we want to accept neutral moves to explore?
                        # Yes, accept neutral moves always to explore.
                        
                        # Update lists
                        non_edges.pop(idx_add)
                        non_edges.append(e_rem)
                    else:
                        # Revert removal
                        adj[u_rem] |= (1 << v_rem)
                        adj[v_rem] |= (1 << u_rem)
                        current_edges.add(e_rem)
            
            # Cool down
            T = max(T_end, T * 0.99)
            
            # Periodic greedy fill?
            if it % 50 == 0:
                # Try to add any valid non-edge
                added_any = False
                for i in range(len(non_edges)):
                    u, v = non_edges[i]
                    if not would_create_c6(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges.add((u, v))
                        non_edges.pop(i)
                        added_any = True
                        break
                if not added_any:
                    break # No more immediate improvements
            
        return current_edges

    def perturb_and_refine(edges_set, seed_val, num_remove):
        """
        Remove some edges randomly, then run greedy + local search to refine.
        """
        random.seed(seed_val)
        edges = set(edges_set)
        
        to_remove = random.sample(list(edges), min(num_remove, len(edges)))
        for e in to_remove:
            edges.remove(e)
        
        adj = build_adj(edges)
        
        # Run greedy
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
                edges.add((u, v))
        
        # Run SA Local Search
        edges = local_search_sa(edges, seed_val)
        
        return edges

    # Main search loop
    
    best_edges = set()
    best_count = 0
    
    # Try multiple dense random initializations
    if time.time() < deadline - 40:
        for seed_val in [42, 123, 777, 999, 555]:
            if time.time() > deadline - 35:
                break
            dense_edges = dense_random_init(seed_val)
            if len(dense_edges) > best_count:
                best_edges = dense_edges
                best_count = len(best_edges)
    
    # Initial greedy
    best_edges = run_greedy(best_edges, 42)
    
    # SA Local Search
    best_edges = local_search_sa(best_edges, 42)
    
    best_count = len(best_edges)
    
    # Multiple restarts with perturbation
    restart_seed = 100
    for i in range(50):
        if time.time() > deadline - 10:
            break
        
        num_remove = random.randint(2, 8)
        perturbed = perturb_and_refine(best_edges, restart_seed + i, num_remove)
        
        if len(perturbed) > best_count:
            best_edges = perturbed
            best_count = len(perturbed)
        
        restart_seed += 1
    
    # Final polish: try to add any remaining non-edges
    adj_final = build_adj(best_edges)
    final_non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj_final[i] & (1 << j)):
                final_non_edges.append((i, j))
    
    random.shuffle(final_non_edges)
    for u, v in final_non_edges:
        if time.time() > deadline - 2:
            break
        if not would_create_c6(adj_final, u, v):
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
            best_edges.add((u, v))
    
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
