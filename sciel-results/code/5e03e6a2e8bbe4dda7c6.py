import time
import random
import itertools

N = 18
DEADLINE = 145  # 150s limit, leave 5s margin

def construct():
    start_time = time.time()
    
    def time_ok():
        return time.time() - start_time < DEADLINE

    # --- K4-free check ---
    def is_k4_free(adj):
        for a in range(N):
            for b in range(a + 1, N):
                if not (adj[a] >> b) & 1:
                    continue
                common = adj[a] & adj[b]
                cb = common
                while cb:
                    lb = cb & (-cb)
                    c = lb.bit_length() - 1
                    cb ^= lb
                    if adj[c] & common:
                        return False
        return True

    def k4_free_after_add(adj, u, v):
        common = adj[u] & adj[v]
        cb = common
        while cb:
            lb = cb & (-cb)
            c = lb.bit_length() - 1
            cb ^= lb
            if adj[c] & common:
                return False
        return True

    # --- Exact B&B for final score verification (optional, but good for debugging/local check) ---
    # We might not need this if we trust the heuristic, but it's good to know where we stand.
    # However, running B&B multiple times is expensive. We will run it once at the very end.
    def max_tf_size_exact(adj):
        n = N
        best = [1]
        vertices = list(range(n))
        
        def bnb(idx, current_mask, current_size):
            if current_size > best[0]:
                best[0] = current_size
            if idx >= n:
                return
            if current_size + (n - idx) <= best[0]:
                return
            
            u = vertices[idx]
            
            # Option 1: exclude u
            bnb(idx + 1, current_mask, current_size)
            
            # Option 2: include u
            nu_in_set = adj[u] & current_mask
            ok = True
            nb = nu_in_set
            while nb:
                lb = nb & (-nb)
                a = lb.bit_length() - 1
                nb ^= lb
                if adj[a] & (nu_in_set & ~(1 << a)):
                    ok = False
                    break
            if ok:
                bnb(idx + 1, current_mask | (1 << u), current_size + 1)
        
        bnb(0, 0, 0)
        return best[0]

    # --- Setup for Heuristic Search ---
    # We will minimize the number of Triangle-Free subsets of size K.
    # Start with K=9. If we reach 0, try K=8.
    
    def run_search(K):
        if not time_ok():
            return None, None
            
        # Precompute subsets of size K
        subsets = []
        for comb in itertools.combinations(range(N), K):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets.append(mask)
        
        num_subsets = len(subsets)
        
        # Precompute indices of subsets containing each pair (u, v)
        # pair_to_subsets[u][v] = list of indices
        pair_to_subsets = [[[] for _ in range(N)] for _ in range(N)]
        for i, mask in enumerate(subsets):
            # Get vertices in mask
            verts = [v for v in range(N) if (mask >> v) & 1]
            for u in verts:
                for v in verts:
                    if u < v:
                        pair_to_subsets[u][v].append(i)
                        pair_to_subsets[v][u].append(i)

        # Initialize graph
        # Start with a random dense graph, then remove edges to make it K4-free
        adj = [0] * N
        for u in range(N):
            for v in range(u + 1, N):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Fix K4 violations
        changed = True
        while changed and time_ok():
            changed = False
            for u in range(N):
                for v in range(u + 1, N):
                    if (adj[u] >> v) & 1:
                        if not k4_free_after_add(adj, u, v):
                            # This edge is part of a K4, remove it
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            changed = True
        
        if not is_k4_free(adj):
            # Fallback to Turan
            adj = [0] * N
            for a in range(6):
                for b in range(6, 12):
                    adj[a] |= (1 << b); adj[b] |= (1 << a)
                for b in range(12, 18):
                    adj[a] |= (1 << b); adj[b] |= (1 << a)
            for a in range(6, 12):
                for b in range(12, 18):
                    adj[a] |= (1 << b); adj[b] |= (1 << a)

        # Calculate initial triangle counts for each subset
        tri_counts = [0] * num_subsets
        num_tf = 0
        
        # To speed up initialization, we can just iterate all triples and add to subsets containing them?
        # No, iterating all subsets and counting triangles inside is easier.
        # num_subsets ~ 48k. K=9. Triples in subset ~ 84.
        # 48k * 84 ~ 4M ops. Fast enough.
        
        for i, mask in enumerate(subsets):
            t_count = 0
            # Extract vertices
            verts = [v for v in range(N) if (mask >> v) & 1]
            for a_idx in range(len(verts)):
                a = verts[a_idx]
                for b_idx in range(a_idx + 1, len(verts)):
                    b = verts[b_idx]
                    if (adj[a] >> b) & 1:
                        # Check common neighbors in verts
                        common = adj[a] & adj[b] & mask
                        cb = common
                        while cb:
                            lb = cb & (-cb)
                            c = lb.bit_length() - 1
                            cb ^= lb
                            if c in verts: # c is in subset
                                t_count += 1
            tri_counts[i] = t_count
            if t_count == 0:
                num_tf += 1
        
        best_adj = adj[:]
        best_num_tf = num_tf
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.995
        current_num_tf = num_tf
        
        while time_ok() and T > 0.1:
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            
            has_edge = (adj[u] >> v) & 1
            
            # Calculate delta
            # Only subsets containing both u and v are affected
            indices = pair_to_subsets[u][v]
            
            delta = 0
            # We need to know the change in tri_counts for each index
            # Change is +/- popcount(common(u,v) & mask)
            
            common_uv = adj[u] & adj[v]
            
            # Pre-compute the k values for this move?
            # k_i = popcount(common_uv & subsets[i])
            
            # If adding edge: tri_counts[i] increases by k_i
            # If removing edge: tri_counts[i] decreases by k_i
            
            # We need to check if any tri_counts[i] crosses 0.
            
            # Optimization: 
            # If adding:
            #   If k_i == 0, tri_counts[i] doesn't change.
            #   If k_i > 0:
            #     If tri_counts[i] was 0, it becomes k_i (>0). num_tf decreases by 1.
            #     If tri_counts[i] was >0, it stays >0. num_tf unchanged.
            
            # If removing:
            #   If k_i == 0, no change.
            #   If k_i > 0:
            #     If tri_counts[i] was k_i, it becomes 0. num_tf increases by 1.
            #     If tri_counts[i] > k_i, it stays >0. num_tf unchanged.
            #     (Cannot be < k_i because k_i is the number of triangles using edge (u,v), 
            #      and tri_counts[i] >= k_i is not necessarily true? 
            #      Wait. tri_counts[i] is total triangles in S.
            #      k_i is triangles in S using (u,v).
            #      So tri_counts[i] >= k_i is TRUE.
            #      So if we remove, new_count = old_count - k_i.
            #      If old_count == k_i, new is 0.
            #      If old_count > k_i, new > 0.)

            # Let's calculate delta_num_tf
            d_num_tf = 0
            
            if not has_edge:
                # Adding edge
                if not k4_free_after_add(adj, u, v):
                    # Reject move (infinite cost)
                    continue
                
                for i in indices:
                    k = (common_uv & subsets[i]).bit_count()
                    if k > 0:
                        if tri_counts[i] == 0:
                            d_num_tf -= 1
                        # Update tri_counts temporarily? 
                        # We need to revert if we reject.
                        # Better to store changes or just revert immediately.
                        tri_counts[i] += k
                
                # Check if accepted
                if d_num_tf < 0 or (d_num_tf == 0 and random.random() < 0.5) or (d_num_tf > 0 and random.random() < (2.718 ** (-d_num_tf / T))):
                    # Accept
                    current_num_tf += d_num_tf
                    if current_num_tf < best_num_tf:
                        best_num_tf = current_num_tf
                        best_adj = adj[:]
                    # Commit edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    # Revert
                    for i in indices:
                        k = (common_uv & subsets[i]).bit_count()
                        if k > 0:
                            tri_counts[i] -= k
            else:
                # Removing edge
                for i in indices:
                    k = (common_uv & subsets[i]).bit_count()
                    if k > 0:
                        if tri_counts[i] == k:
                            d_num_tf += 1
                        tri_counts[i] -= k
                
                # Accept if better or SA
                if d_num_tf < 0 or (d_num_tf == 0 and random.random() < 0.5) or (d_num_tf > 0 and random.random() < (2.718 ** (-d_num_tf / T))):
                    # Accept
                    current_num_tf += d_num_tf
                    if current_num_tf < best_num_tf:
                        best_num_tf = current_num_tf
                        best_adj = adj[:]
                    # Commit removal
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # Revert
                    for i in indices:
                        k = (common_uv & subsets[i]).bit_count()
                        if k > 0:
                            tri_counts[i] += k
            
            T *= cooling
        
        return best_adj, best_num_tf

    # Strategy:
    # 1. Try to find graph with 0 TF subsets of size 9.
    # 2. If successful, try size 8.
    
    best_adj = None
    best_score_estimate = 9
    
    # Run for K=9
    adj9, tf9 = run_search(9)
    if adj9:
        best_adj = adj9
        if tf9 == 0:
            best_score_estimate = 8
            # Try K=8 if time permits
            if time_ok():
                # Re-initialize search with K=8, starting from adj9
                # The run_search function generates a random graph. 
                # We should modify it to accept an initial graph.
                # For now, just run another instance.
                adj8, tf8 = run_search(8)
                if adj8 and tf8 == 0:
                    best_adj = adj8
                    best_score_estimate = 7
                elif adj8:
                    # Compare? Hard to compare without exact score.
                    # Assume lower K target is better if both are 0? 
                    # If tf8 > 0, max TF is >= 8.
                    # If tf9 == 0, max TF <= 8.
                    # So adj9 is guaranteed <= 8.
                    # If adj8 has tf8 > 0, it might have max TF 8 or 9.
                    # Stick with adj9 if we can't guarantee adj8 is better.
                    pass
        else:
            # tf9 > 0, so max TF >= 9.
            # best_adj is still the one with min tf9.
            pass

    if best_adj is None:
        # Fallback
        adj = [0] * N
        for a in range(6):
            for b in range(6, 12):
                adj[a] |= (1 << b); adj[b] |= (1 << a)
            for b in range(12, 18):
                adj[a] |= (1 << b); adj[b] |= (1 << a)
        for a in range(6, 12):
            for b in range(12, 18):
                adj[a] |= (1 << b); adj[b] |= (1 << a)
        best_adj = adj

    # Final verification with exact B&B if time allows
    if time_ok():
        # We have ~10s left. B&B on 18 vertices might take a while.
        # Let's try. If it times out, we just return the graph.
        # The evaluator will run its own B&B.
        # We don't strictly need to know the score, but it's good for sanity.
        # Actually, if I return a graph that is K4-free, the evaluator will score it.
        # I should just return best_adj.
        
        # One last polish: local search on best_adj using exact score? 
        # No, too slow.
        
        pass

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
