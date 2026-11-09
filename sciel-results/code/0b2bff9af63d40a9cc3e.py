"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    start_time = time.time()
    deadline = start_time + 140.0

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def greedy_1flip(labels, max_passes=10):
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        return labels

    def fast_2swap(labels, max_passes=100):
        """Fast 2-swap local search with incremental O(n) updates per accepted swap."""
        labels = labels.copy()
        
        # Initialize sum_same and sum_other
        sum_same = np.zeros(n, dtype=np.int32)
        sum_other = np.zeros(n, dtype=np.int32)
        for v in range(n):
            sum_same[v] = adj[v][labels == labels[v]].sum()
            sum_other[v] = adj[v][labels != labels[v]].sum()
        
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            # Shuffle order to avoid bias
            order = list(range(n))
            np.random.shuffle(order)
            for i_idx in range(n):
                if time.time() > deadline:
                    break
                i = order[i_idx]
                li = labels[i]
                for j_idx in range(n):
                    if j_idx == i_idx:
                        continue
                    j = order[j_idx]
                    if labels[i] == labels[j]:
                        continue
                    
                    a = int(adj[i, j])
                    si = int(sum_same[i])
                    oi = int(sum_other[i])
                    sj = int(sum_same[j])
                    oj = int(sum_other[j])
                    
                    gain = si + sj - oi - oj + 2 * a
                    
                    if gain > 0:
                        # Accept swap
                        lj = labels[j]
                        labels[i], labels[j] = lj, li
                        
                        # Update sum_same/sum_other incrementally
                        sum_same[i], sum_other[i] = oi, si
                        sum_same[j], sum_other[j] = oj, sj
                        
                        # Update all other nodes k
                        mask_li = (labels == li)
                        mask_lj = (labels == lj)
                        
                        # Exclude i and j from masks (they already updated)
                        mask_li[i] = False
                        mask_li[j] = False
                        mask_lj[i] = False
                        mask_lj[j] = False
                        
                        delta_li = adj[:, j] - adj[:, i]  # for nodes with label li
                        delta_lj = adj[:, i] - adj[:, j]  # for nodes with label lj
                        
                        sum_same[mask_li] += delta_li[mask_li]
                        sum_same[mask_lj] += delta_lj[mask_lj]
                        sum_other[mask_li] -= delta_li[mask_li]
                        sum_other[mask_lj] -= delta_lj[mask_lj]
                        
                        improved = True
        
        return labels

    def lpt_construct():
        """LPT-style constructive heuristic: assign vertices by degree (largest first),
        placing each vertex in the partition with smaller current load (degree imbalance)."""
        labels = np.full(n, -1, dtype=np.int8)
        degrees = adj.sum(axis=1)
        
        # Process vertices in order of decreasing degree (LPT heuristic)
        order = list(range(n))
        order.sort(key=lambda v: -degrees[v])
        
        # Track load: number of edges from each partition to unassigned vertices
        load_0 = np.zeros(n, dtype=np.int32)  # edges from partition 0 to unassigned
        load_1 = np.zeros(n, dtype=np.int32)  # edges from partition 1 to unassigned
        
        # Initialize loads based on edges between vertices
        for v in order:
            for u in order:
                if u != v and adj[v, u]:
                    # When v is assigned, its edges to already-assigned vertices
                    # contribute to the cut if they go to different partitions
                    pass
        
        # Simpler approach: track current cut contribution and balance
        cut_contribution = 0
        size_0 = 0
        size_1 = 0
        
        for v in order:
            if time.time() > deadline:
                break
            
            # Count edges from v to already-assigned vertices in each partition
            edges_to_0 = 0
            edges_to_1 = 0
            for u in range(n):
                if labels[u] != -1 and adj[v, u]:
                    if labels[u] == 0:
                        edges_to_0 += 1
                    else:
                        edges_to_1 += 1
            
            # If we put v in partition 0, we gain edges_to_1 crossing edges
            # If we put v in partition 1, we gain edges_to_0 crossing edges
            # LPT: put in partition with smaller current "load" (fewer assigned neighbors)
            # This balances the partition sizes while maximizing cut
            
            # Use a combination: prefer the partition that gives more crossing edges,
            # but with a tie-breaker toward the smaller partition
            gain_0 = edges_to_1
            gain_1 = edges_to_0
            
            # Add a small bias toward the smaller partition to maintain balance
            bias = 0.1 * (size_1 - size_0)  # positive if partition 0 is smaller
            adjusted_gain_0 = gain_0 + bias
            adjusted_gain_1 = gain_1 - bias
            
            if adjusted_gain_0 >= adjusted_gain_1:
                labels[v] = 0
                cut_contribution += edges_to_1
                size_0 += 1
            else:
                labels[v] = 1
                cut_contribution += edges_to_0
                size_1 += 1
        
        # Handle any unassigned vertices
        for v in range(n):
            if labels[v] == -1:
                labels[v] = np.random.randint(2)
        
        return labels

    def tabu_search(labels, max_iters=200, tabu_len=10):
        """Tabu search to escape local optima."""
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_cut_local = current_cut
        best_labels_local = labels.copy()
        
        # Tabu list: store (vertex, iteration) for recently flipped vertices
        tabu_list = {}
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Remove old tabu entries
            for v in list(tabu_list.keys()):
                if it - tabu_list[v] >= tabu_len:
                    del tabu_list[v]
            
            best_gain = -np.inf
            best_vertex = -1
            
            # Find best non-tabu move
            for v in range(n):
                if v in tabu_list:
                    continue
                
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                gain = other - same
                
                if gain > best_gain:
                    best_gain = gain
                    best_vertex = v
            
            # Aspiration: accept if it improves best even if tabu
            if best_vertex == -1:
                # All moves are tabu, pick best anyway
                for v in range(n):
                    same = adj[v][labels == labels[v]].sum()
                    other = adj[v][labels != labels[v]].sum()
                    gain = other - same
                    if gain > best_gain:
                        best_gain = gain
                        best_vertex = v
            
            if best_vertex == -1:
                break
            
            # Apply the move
            v = best_vertex
            labels[v] = 1 - labels[v]
            tabu_list[v] = it
            
            new_cut = compute_cut(labels)
            
            if new_cut > best_cut_local:
                best_cut_local = new_cut
                best_labels_local = labels.copy()
        
        return best_labels_local, best_cut_local

    # Strategy 1: LPT construction + 1-flip + 2-swap
    if time.time() < deadline:
        labels = lpt_construct()
        labels = greedy_1flip(labels, max_passes=10)
        labels = fast_2swap(labels, max_passes=50)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Strategy 2: Diverse restarts with LPT, random, and go-with-the-flow
    for restart in range(1, 51):
        if time.time() > deadline:
            break
        
        # Diverse initialization strategies
        if restart % 4 == 1:
            labels = lpt_construct()
        elif restart % 4 == 2:
            labels = np.random.randint(0, 2, size=n)
        elif restart % 4 == 3:
            labels = np.arange(n) % 2
        else:
            # Spectral-like: use eigenvector of adjacency
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(adj)
                # Use the eigenvector corresponding to the largest eigenvalue
                idx = np.argmax(eigenvalues)
                labels = (eigenvectors[:, idx] > 0).astype(np.int8)
            except:
                labels = np.random.randint(0, 2, size=n)
        
        labels = greedy_1flip(labels, max_passes=5)
        labels = fast_2swap(labels, max_passes=30)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Strategy 3: Tabu search from best solution
    if best_labels is not None and time.time() < deadline:
        for tabu_restart in range(3):
            if time.time() > deadline:
                break
            # Add some randomness to tabu search starting point
            start_labels = best_labels.copy()
            if tabu_restart > 0:
                # Flip a few random vertices to diversify
                num_flips = min(5, n // 2)
                flip_indices = np.random.choice(n, size=num_flips, replace=False)
                start_labels[flip_indices] = 1 - start_labels[flip_indices]
            
            tabu_labels, tabu_cut = tabu_search(start_labels, max_iters=100, tabu_len=8 + tabu_restart * 3)
            
            if tabu_cut > best_cut:
                best_cut = tabu_cut
                best_labels = tabu_labels.copy()
            
            # Polish with local search
            if time.time() < deadline:
                polished = greedy_1flip(tabu_labels, max_passes=5)
                polished = fast_2swap(polished, max_passes=30)
                polished_cut = compute_cut(polished)
                if polished_cut > best_cut:
                    best_cut = polished_cut
                    best_labels = polished.copy()

    # Final polish with multiple rounds of 1-flip + 2-swap
    if best_labels is not None and time.time() < deadline:
        final_labels = best_labels.copy()
        for _ in range(3):
            if time.time() > deadline:
                break
            final_labels = greedy_1flip(final_labels, max_passes=10)
            final_labels = fast_2swap(final_labels, max_passes=50)
        final_cut = compute_cut(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
