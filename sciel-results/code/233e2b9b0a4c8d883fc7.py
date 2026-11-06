"""Rewritten Maximum Cut program with vectorized local search, 2-flip moves, and aggressive multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    degree = adj.sum(axis=1)
    total_edges = int(adj.sum()) // 2
    
    # Deadline: 150s wall-clock, leave 25s safety margin
    start_time = time.time()
    deadline = start_time + 120.0
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        s = 2 * labels - 1
        val = int(s @ (adj @ s))
        return total_edges - val // 2
    
    def spectral_init(adj):
        """Generate initial labeling using Fiedler vector of graph Laplacian."""
        D = sp.diags(degree)
        A_sparse = sp.csr_matrix(adj)
        L = D - A_sparse
        
        try:
            eigenvalues, eigenvectors = spla.eigsh(L, k=2, which='SM')
            idx = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx[1]]
            labels = (fiedler > 0).astype(int)
            if np.all(labels == labels[0]):
                labels = np.random.randint(0, 2, size=n)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n)
    
    def vectorized_gain(labels, adj):
        """
        Compute gain for each vertex if flipped.
        gain[v] = s[v] * (adj @ s)[v] where s = 2*labels - 1
        """
        s = 2 * labels - 1
        adj_s = adj @ s
        gains = s * adj_s
        return gains
    
    def local_search_pq(labels, adj, max_iters=30000, time_limit=None):
        """Local search using priority queue for efficient gain tracking."""
        labels = labels.copy()
        s = 2 * labels - 1
        adj_s = adj @ s
        gains = s * adj_s
        
        # Build max-heap (negate for min-heap)
        pq = [(-int(gains[i]), i) for i in range(n)]
        heapq.heapify(pq)
        
        iters = 0
        while iters < max_iters:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            
            # Pop best gain
            while pq:
                neg_gain, v = heapq.heappop(pq)
                gain = -neg_gain
                # Verify gain is still valid
                current_gain = int(s[v] * adj_s[v])
                if gain != current_gain:
                    continue
                break
            else:
                break
            
            if gain <= 0:
                break
            
            # Flip vertex v
            s[v] = -s[v]
            labels[v] = 1 - labels[v]
            
            # Update adj_s and gains for neighbors of v
            neighbors = np.where(adj[v] != 0)[0]
            for u in neighbors:
                adj_s[u] -= 2 * s[v]  # s[v] changed sign, so adj[v,u]*s[v] changes by -2*old_s[v] = -2*(-s[v])... 
                # Actually: old contribution was adj[v,u]*old_s[v], new is adj[v,u]*new_s[v] = adj[v,u]*(-old_s[v])
                # Change = adj[v,u]*(-old_s[v] - old_s[v]) = -2*adj[v,u]*old_s[v] = -2*adj[v,u]*(-new_s[v]) = 2*adj[v,u]*new_s[v]
                # Wait, let me redo: old_s[v] = -new_s[v]
                # adj_s[u] += adj[v,u]*(new_s[v] - old_s[v]) = adj[v,u]*(new_s[v] + new_s[v]) = 2*adj[v,u]*new_s[v]
                adj_s[u] += 2 * int(adj[v, u]) * s[v]
                
                # Update gain for u
                new_gain_u = int(s[u] * adj_s[u])
                heapq.heappush(pq, (-new_gain_u, u))
            
            # Also update gain for v itself
            # adj_s[v] changed because s[v] flipped
            # adj_s[v] = sum_u adj[v,u]*s[u], this doesn't change when s[v] flips
            # But gain[v] = s[v]*adj_s[v] changes sign
            new_gain_v = int(s[v] * adj_s[v])
            heapq.heappush(pq, (-new_gain_v, v))
        
        return labels
    
    def two_flip_local_search(labels, adj, max_rounds=50, time_limit=None):
        """2-flip local search to escape 1-flip local optima."""
        labels = labels.copy()
        s = 2 * labels - 1
        current_cut = cut_size(labels)
        
        for r in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Compute current gains
            adj_s = adj @ s
            gains = s * adj_s
            
            # Find best pair of flips
            # Try all pairs where both have negative or zero gain individually
            # but combined gain is positive
            best_pair_gain = 0
            best_pair = None
            
            # For efficiency, only consider pairs where at least one has gain > -degree
            # gain of flipping both i and j:
            # = gain[i] + gain[j] - 2*adj[i,j]*(s[i]*s[j] > 0 ? 1 : 0) * ... 
            # Let's derive: flipping i changes cut by gain[i], flipping j changes by gain[j]
            # But if i and j are adjacent, flipping both affects the edge between them twice
            # Actually, the combined gain is:
            # Let s' = s with s[i] and s[j] flipped
            # cut(s') - cut(s) = gain[i] + gain[j] - 2*adj[i,j]*s[i]*s[j]
            # Wait, let me verify with a simple case.
            # If i,j adjacent and same label: flipping both makes them same label still (both flipped)
            # Edge between them was not crossing, still not crossing. No change from that edge.
            # gain[i] includes -1 for edge (i,j) if same label... 
            # Let me just compute directly for candidate pairs.
            
            # For 100 nodes, O(n^2) = 10000 pairs, which is fine
            for i in range(n):
                if gains[i] <= 0:  # Only consider flipping vertices that don't improve alone
                    for j in range(i + 1, n):
                        if gains[j] <= 0:
                            # Combined gain
                            if adj[i, j] != 0:
                                # They are adjacent
                                # If same label: edge not crossing before, not crossing after (both flip)
                                #   gain from edge: 0
                                # If different label: edge crossing before, not crossing after
                                #   gain from edge: -1 (we lose a crossing edge)
                                # Actually let me compute: 
                                # combined_gain = gains[i] + gains[j] - 2*adj[i,j]*s[i]*s[j]
                                # If s[i]==s[j]: s[i]*s[j]=1, so -2*adj[i,j]
                                # If s[i]!=s[j]: s[i]*s[j]=-1, so +2*adj[i,j]
                                combined = gains[i] + gains[j] - 2 * int(adj[i, j]) * int(s[i] * s[j])
                            else:
                                combined = gains[i] + gains[j]
                            
                            if combined > best_pair_gain:
                                best_pair_gain = combined
                                best_pair = (i, j)
            
            if best_pair is None or best_pair_gain <= 0:
                break
            
            i, j = best_pair
            s[i] = -s[i]
            s[j] = -s[j]
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            current_cut += best_pair_gain
        
        return labels
    
    def simulated_annealing(labels, adj, iters=100000, t_start=1.0, t_end=0.001, time_limit=None):
        """Optimized simulated annealing for max cut."""
        labels = labels.copy()
        s = 2 * labels - 1
        adj_s = adj @ s
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / iters)
        
        for it in range(iters):
            if time_limit is not None and time.time() > time_limit:
                break
            t *= cooling
            
            v = random.randint(0, n - 1)
            gain = int(s[v] * adj_s[v])
            
            if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                # Flip vertex v
                s[v] = -s[v]
                labels[v] = 1 - labels[v]
                current_cut += gain
                
                # Update adj_s for neighbors
                neighbors = np.where(adj[v] != 0)[0]
                for u in neighbors:
                    adj_s[u] += 2 * int(adj[v, u]) * s[v]
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    # Multi-start with different initializations
    best_labels = None
    best_cut = -1
    
    def try_solution(labels):
        nonlocal best_labels, best_cut
        # 1-flip local search
        labels = local_search_pq(labels, adj, max_iters=30000, time_limit=deadline)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
        # 2-flip local search
        if time.time() < deadline - 20:
            labels2 = two_flip_local_search(labels, adj, max_rounds=30, time_limit=deadline - 20)
            c2 = cut_size(labels2)
            if c2 > best_cut:
                best_cut = c2
                best_labels = labels2.copy()
    
    # Start 1: spectral initialization
    labels = spectral_init(adj)
    try_solution(labels)
    
    # Start 2: alternating
    labels = np.arange(n) % 2
    try_solution(labels)
    
    # Start 3-10: random starts
    for start_iter in range(8):
        if time.time() > deadline - 25:
            break
        labels = np.random.randint(0, 2, size=n)
        try_solution(labels)
    
    # Simulated annealing from best solution found so far
    if time.time() < deadline - 15:
        sa_labels, sa_cut = simulated_annealing(
            best_labels, adj, iters=100000, t_start=1.0, t_end=0.001,
            time_limit=deadline - 15
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Final polish: 1-flip then 2-flip
    if time.time() < deadline - 5:
        best_labels = local_search_pq(best_labels, adj, max_iters=50000, time_limit=deadline - 5)
        c = cut_size(best_labels)
        if c > best_cut:
            best_cut = c
        if time.time() < deadline - 5:
            best_labels = two_flip_local_search(best_labels, adj, max_rounds=50, time_limit=deadline - 5)
            c = cut_size(best_labels)
            if c > best_cut:
                best_cut = c
    
    return best_labels
# EVOLVE-BLOCK-END
