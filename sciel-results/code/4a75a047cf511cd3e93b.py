"""Rewritten program for Maximum Cut task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    total_edges = adj.sum() // 2
    degs = adj.sum(axis=1)
    
    def cut_size(labels):
        s = 2 * labels - 1
        val = s @ (adj @ s)
        return total_edges - val // 2
    
    def compute_gains_1flip(labels):
        """Vectorized computation of 1-flip gains for all vertices."""
        s = 2 * labels - 1
        # gain[v] = 2 * (sum of adj[v,u] where s[u]==s[v]) - deg[v]
        # = 2 * (adj @ s) * s[v] - deg[v]
        # Actually: same_side_sum[v] = sum_u adj[v,u] * (s[u]==s[v])
        # = sum_u adj[v,u] * (1 if s[u]==s[v] else 0)
        # = (adj @ (s == s[:,None]).astype(int))[v] -- too slow
        # Better: same_side_sum[v] = (adj[v] @ (s == s[v]))
        # Vectorized: for each v, count neighbors on same side
        # s @ adj gives for each u: sum_v s[v]*adj[v,u]
        # Let's use: for vertex v with sign s[v], same_side = sum_u adj[v,u] where s[u]==s[v]
        # = 0.5 * (deg[v] + s[v] * (adj @ s)[v])
        # Because: sum_u adj[v,u]*s[u] = sum_{u:s[u]=s[v]} adj[v,u] - sum_{u:s[u]=-s[v]} adj[v,u]
        # And: same + diff = deg, same - diff = (adj@ s)[v]*s[v]... 
        # Actually (adj @ s)[v] = sum_u adj[v,u]*s[u]
        # same_side = sum_{u: s[u]=s[v]} adj[v,u]
        # diff_side = sum_{u: s[u]=-s[v]} adj[v,u]
        # same + diff = deg[v]
        # same - diff = s[v] * (adj @ s)[v]  ... let's verify
        # If s[v]=1: same-diff = sum_{s[u]=1}adj - sum_{s[u]=-1}adj = (adj@ s)[v] ✓
        # If s[v]=-1: same-diff = sum_{s[u]=-1}adj - sum_{s[u]=1}adj = -(adj@ s)[v] = s[v]*(adj@ s)[v] ✓
        # So: same = (deg[v] + s[v]*(adj@s)[v]) / 2
        # gain[v] = 2*same - deg = deg + s[v]*(adj@s)[v] - deg = s[v]*(adj@s)[v]
        
        adj_s = adj @ s
        gains = s * adj_s
        return gains
    
    def compute_gains_2flip(labels):
        """Vectorized computation of 2-flip gains for all pairs.
        
        For flipping vertices i and j simultaneously:
        gain(i,j) = gain_1(i) + gain_1(j) - 2*adj[i,j]*(s[i]+s[j])
        where s = 2*labels - 1
        
        Derivation:
        - Flipping i changes cut by: s[i] * (adj @ s)[i]
        - Flipping j changes cut by: s[j] * (adj @ s)[j]  
        - But the edge (i,j) is counted in both, so we need to correct
        - If s[i] == s[j] (same side): edge (i,j) contributes -1 to cut before, +1 after flip both
          Net change from that edge: +2. But in individual gains, each flip counts this edge.
          gain_i includes: s[i]*adj[i,j]*s[j] = s[i]*s[j]*adj[i,j] = adj[i,j] (since same sign)
          gain_j includes: s[j]*adj[j,i]*s[i] = s[i]*s[j]*adj[i,j] = adj[i,j]
          Total from edge in both gains: 2*adj[i,j]
          Actual change: +2*adj[i,j] (edge goes from internal to crossing)
          So no correction needed when s[i]==s[j]
        
        - If s[i] != s[j] (opposite sides): edge (i,j) contributes +1 to cut before, -1 after flip both
          Net change from that edge: -2.
          gain_i includes: s[i]*adj[i,j]*s[j] = -adj[i,j]
          gain_j includes: s[j]*adj[j,i]*s[i] = -adj[i,j]
          Total from edge in both gains: -2*adj[i,j]
          Actual change: -2*adj[i,j]
          So no correction needed when s[i]!=s[j] either!
        
        Wait, let me re-derive more carefully.
        
        Cut = sum_{i<j} adj[i,j] * (1 - s[i]*s[j]) / 2
        
        After flipping i and j (s[i]->-s[i], s[j]->-s[j]):
        New cut = sum_{i<j} adj[i,j] * (1 - s'[i]*s'[j]) / 2
        
        Change = Cut_new - Cut_old
        
        For edge (i,j): old contribution = adj[i,j]*(1-s[i]*s[j])/2, new = adj[i,j]*(1+s[i]*s[j])/2
        Change = adj[i,j]*s[i]*s[j]
        
        For edge (i,k) where k!=j: old = adj[i,k]*(1-s[i]*s[k])/2, new = adj[i,k]*(1+s[i]*s[k])/2
        Change = adj[i,k]*s[i]*s[k]
        
        For edge (j,k) where k!=i: old = adj[j,k]*(1-s[j]*s[k])/2, new = adj[j,k]*(1+s[j]*s[k])/2
        Change = adj[j,k]*s[j]*s[k]
        
        For edge (k,l) where k,l != i,j: no change.
        
        Total change = adj[i,j]*s[i]*s[j] + s[i]*sum_{k!=j}adj[i,k]*s[k] + s[j]*sum_{k!=i}adj[j,k]*s[k]
        
        Now, (adj@s)[i] = sum_k adj[i,k]*s[k] = adj[i,j]*s[j] + sum_{k!=j}adj[i,k]*s[k]
        So sum_{k!=j}adj[i,k]*s[k] = (adj@s)[i] - adj[i,j]*s[j]
        
        Similarly: sum_{k!=i}adj[j,k]*s[k] = (adj@s)[j] - adj[i,j]*s[i]
        
        Total change = adj[i,j]*s[i]*s[j] + s[i]*((adj@s)[i] - adj[i,j]*s[j]) + s[j]*((adj@s)[j] - adj[i,j]*s[i])
        = adj[i,j]*s[i]*s[j] + s[i]*(adj@s)[i] - adj[i,j]*s[i]*s[j] + s[j]*(adj@s)[j] - adj[i,j]*s[i]*s[j]
        = s[i]*(adj@s)[i] + s[j]*(adj@s)[j] - adj[i,j]*s[i]*s[j]
        
        So: gain_2flip(i,j) = s[i]*(adj@s)[i] + s[j]*(adj@s)[j] - adj[i,j]*s[i]*s[j]
        
        The first two terms are just the 1-flip gains for i and j.
        The correction is: -adj[i,j]*s[i]*s[j]
        
        If s[i]==s[j]: correction = -adj[i,j] (since s[i]*s[j]=1)
        If s[i]!=s[j]: correction = +adj[i,j] (since s[i]*s[j]=-1)
        """
        s = 2 * labels - 1
        adj_s = adj @ s
        g1 = s * adj_s  # 1-flip gains
        
        # 2-flip gain matrix: G[i,j] = g1[i] + g1[j] - adj[i,j]*s[i]*s[j]
        # = g1[i] + g1[j] - adj[i,j] * (s @ s^T)[i,j]
        ss = s[:, None] * s[None, :]  # outer product, n x n
        correction = adj * ss  # element-wise
        G = g1[:, None] + g1[None, :] - correction
        
        # Only consider pairs i < j
        # Set diagonal to -inf
        np.fill_diagonal(G, -np.inf)
        # Only upper triangle
        G = np.triu(G, k=1)
        
        return G, g1
    
    def local_search_1flip(labels, time_limit=None, max_iters=50000):
        """Fast 1-flip local search."""
        labels = labels.copy()
        iters = 0
        
        while iters < max_iters:
            if time.time() > time_limit:
                break
            iters += 1
            
            gains = compute_gains_1flip(labels)
            best_v = np.argmax(gains)
            best_gain = gains[best_v]
            
            if best_gain <= 0:
                break
            
            labels[best_v] = 1 - labels[best_v]
        
        return labels
    
    def local_search_2flip(labels, time_limit=None, max_iters=20000):
        """2-flip local search: explore pairwise swaps to escape 1-flip local optima."""
        labels = labels.copy()
        iters = 0
        
        while iters < max_iters:
            if time.time() > time_limit:
                break
            iters += 1
            
            # First try 1-flip
            g1 = compute_gains_1flip(labels)
            best_1 = np.argmax(g1)
            if g1[best_1] > 0:
                labels[best_1] = 1 - labels[best_1]
                continue
            
            # No improving 1-flip, try 2-flip
            G, _ = compute_gains_2flip(labels)
            idx = np.unravel_index(np.argmax(G), G.shape)
            best_2_gain = G[idx]
            
            if best_2_gain <= 0:
                break
            
            i, j = idx
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
        
        return labels
    
    def sa_from_start(labels, adj, n, t_start, t_end, iters, time_limit=None):
        """Simulated annealing with 1-flip and occasional 2-flip moves."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max(iters, 1))
        
        for it in range(iters):
            if time.time() > time_limit:
                break
            t *= cooling
            
            # 80% chance 1-flip, 20% chance 2-flip
            if random.random() < 0.8:
                v = random.randint(0, n - 1)
                s = 2 * labels - 1
                gain = s[v] * (adj @ s)[v]
                
                if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                    labels[v] = 1 - labels[v]
                    current_cut += gain
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
            else:
                # 2-flip: pick two random distinct vertices
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                while j == i:
                    j = random.randint(0, n - 1)
                
                s = 2 * labels - 1
                adj_s = adj @ s
                gain = s[i]*adj_s[i] + s[j]*adj_s[j] - adj[i,j]*s[i]*s[j]
                
                if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    current_cut += gain
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
        
        return best_labels, best_cut
    
    def greedy_init(adj, n):
        """Greedy initialization: add vertices one by one to the side that maximizes cut."""
        labels = np.zeros(n, dtype=int)
        order = np.arange(n)
        np.random.shuffle(order)
        
        for v in order:
            left_count = (adj[v] * (labels == 0)).sum()
            right_count = (adj[v] * (labels == 1)).sum()
            if right_count > left_count:
                labels[v] = 0
            else:
                labels[v] = 1
        return labels
    
    def spectral_init(adj, n):
        """Spectral initialization using Fiedler vector."""
        try:
            from scipy.sparse.csgraph import laplacian
            from scipy.sparse.linalg import eigsh
            import scipy.sparse as sp
            
            A = sp.csr_matrix(adj)
            L = laplacian(A)
            # Get second smallest eigenvector (Fiedler vector)
            vals, vecs = eigsh(L, k=2, which='SM')
            fiedler = vecs[:, 1]
            # Threshold at median
            median_val = np.median(fiedler)
            labels = (fiedler > median_val).astype(int)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Multiple diverse starts with local search
    starts = []
    
    # Start 1: alternating
    starts.append(np.arange(n) % 2)
    
    # Start 2: greedy
    starts.append(greedy_init(adj, n))
    
    # Start 3: spectral
    starts.append(spectral_init(adj, n))
    
    # Starts 4-10: random
    for _ in range(7):
        starts.append(np.random.randint(0, 2, size=n))
    
    for i, init_labels in enumerate(starts):
        if time.time() > deadline:
            break
        
        # 1-flip local search first
        ls_labels = local_search_1flip(init_labels, time_limit=deadline)
        c = cut_size(ls_labels)
        
        # Then 2-flip local search to escape local optima
        ls2_labels = local_search_2flip(ls_labels, time_limit=deadline)
        c2 = cut_size(ls2_labels)
        
        if c2 > c:
            c = c2
            ls_labels = ls2_labels
        
        if c > best_cut:
            best_cut = c
            best_labels = ls_labels.copy()
    
    # Phase 2: Simulated annealing from best solution (with 2-flip moves)
    if time.time() < deadline - 15:
        sa_time_limit = deadline - 15
        sa_labels, sa_cut = sa_from_start(
            best_labels, adj, n, 
            t_start=3.0, t_end=0.01, 
            iters=200000, 
            time_limit=sa_time_limit
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Phase 3: Final local search polish (1-flip then 2-flip)
    if time.time() < deadline:
        final_labels = local_search_1flip(best_labels, time_limit=deadline)
        final2_labels = local_search_2flip(final_labels, time_limit=deadline)
        final_cut = cut_size(final2_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final2_labels
    
    # Phase 4: One more SA pass if time allows
    if time.time() < deadline - 5:
        sa2_labels, sa2_cut = sa_from_start(
            best_labels, adj, n,
            t_start=1.0, t_end=0.005,
            iters=100000,
            time_limit=deadline - 5
        )
        if sa2_cut > best_cut:
            best_cut = sa2_cut
            best_labels = sa2_labels.copy()
        
        # Final polish
        if time.time() < deadline:
            final3 = local_search_1flip(best_labels, time_limit=deadline)
            final4 = local_search_2flip(final3, time_limit=deadline)
            final4_cut = cut_size(final4)
            if final4_cut > best_cut:
                best_labels = final4
    
    return best_labels
# EVOLVE-BLOCK-END
