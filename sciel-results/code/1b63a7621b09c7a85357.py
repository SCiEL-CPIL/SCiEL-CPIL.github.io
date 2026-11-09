"""Maximum Cut via GW-style spectral embedding + fast 1-flip + optimized 2-swap + 3-flip local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s limit

    if n == 0:
        return np.array([], dtype=np.int8)

    adj = np.array(adjacency, dtype=np.int32)
    total_edges = int(np.sum(adj)) // 2

    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def flip_greedy(labels, max_passes=50):
        """Fast 1-flip greedy using incremental cut tracking."""
        labels = labels.copy()
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                deg = degrees[v]
                gain = 2 * same_count[v] - deg
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    changed = True
                    new_same_v = deg - same_count[v]
                    same_count[v] = new_same_v
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            same_count[u] += 1
                        else:
                            same_count[u] -= 1
            if not changed:
                break
        return labels

    def tabu_search(labels, max_iter=5000):
        """Cerny tabu search with dynamic tabu length and aspiration."""
        labels = labels.copy()
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        
        tabu_until = np.zeros(n, dtype=np.int32)
        base_tabu_len = 15
        current_cut = compute_cut(labels)
        best_cut_ts = current_cut
        best_labels_ts = labels.copy()
        iteration = 0
        no_improve = 0
        deg_arr = degrees
        
        while iteration < max_iter:
            if time.time() > deadline:
                break
            
            # Dynamic tabu length
            tabu_len = base_tabu_len + no_improve // 50
            if tabu_len > 40:
                tabu_len = 40
            
            # Vectorized gain computation
            gains = 2 * same_count - deg_arr
            tabu_mask = tabu_until > iteration
            
            # Aspiration criterion
            best_non_tabu_gain = np.max(np.where(tabu_mask, -999999, gains))
            best_overall_gain = np.max(gains)
            
            if best_overall_gain > best_non_tabu_gain and best_overall_gain > -999999:
                best_v = int(np.argmax(gains))
                best_gain = best_overall_gain
            else:
                masked_gains = np.where(tabu_mask, -999999, gains)
                best_v = int(np.argmax(masked_gains))
                best_gain = best_non_tabu_gain
            
            if best_gain <= -999998:
                break
            
            v = best_v
            labels[v] = 1 - labels[v]
            tabu_until[v] = iteration + tabu_len
            
            old_sc_v = same_count[v]
            same_count[v] = deg_arr[v] - old_sc_v
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[u] += 1
                else:
                    same_count[u] -= 1
            
            current_cut += best_gain
            iteration += 1
            
            if current_cut > best_cut_ts:
                best_cut_ts = current_cut
                best_labels_ts = labels.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Reset if stuck too long
            if no_improve > 200:
                no_improve = 0
        
        return best_labels_ts

    def two_swap_search(labels, max_passes=20):
        """Optimized 2-swap local search."""
        labels = labels.copy()
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    break
                li = labels[i]
                diff_mask = labels != li
                diff_mask[i] = False
                candidates = np.where(diff_mask)[0]
                
                if len(candidates) == 0:
                    continue
                
                e_ij = adj[i, candidates]
                delta_same = (degrees[i] + degrees[candidates] - 2*same_count[i] - 2*same_count[candidates] - 2*e_ij)
                
                best_idx = int(np.argmax(delta_same))
                if delta_same[best_idx] <= 0:
                    continue
                
                j = candidates[best_idx]
                lj = labels[j]
                e_ij_val = adj[i, j]
                
                labels[i] = lj
                labels[j] = li
                improved = True
                
                old_same_i = same_count[i]
                old_same_j = same_count[j]
                same_count[i] = degrees[i] - old_same_i - e_ij_val
                same_count[j] = degrees[j] - old_same_j - e_ij_val
                
                for u in neighbors[i]:
                    if u == j:
                        continue
                    if labels[u] == labels[i]:
                        same_count[u] += 1
                    else:
                        same_count[u] -= 1
                
                for u in neighbors[j]:
                    if u == i:
                        continue
                    if labels[u] == labels[j]:
                        same_count[u] += 1
                    else:
                        same_count[u] -= 1
            
            if not improved:
                break
        
        return labels

    def three_flip_search(labels, max_passes=10):
        """3-flip local search to escape deeper local optima."""
        labels = labels.copy()
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    break
                li = labels[i]
                # Find two vertices j, k with same label as i (flipping all three)
                # Or one with same label and one with different label
                # Simplest: flip i and two others to maximize gain
                
                # Try flipping i with each pair of other vertices
                # This is O(n^2) per i, so limit to random sampling
                if random.random() > 0.3:
                    continue  # Skip most i for speed
                
                # Sample a few pairs
                other_vertices = [v for v in range(n) if v != i]
                if len(other_vertices) < 2:
                    continue
                
                # Try up to 10 random pairs
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    j, k = random.sample(other_vertices, 2)
                    
                    # Compute gain for flipping i, j, k
                    # Gain = sum of changes in cut for each flipped vertex
                    # For vertex v: gain_v = 2*same_count[v] - degree[v]
                    # But interactions: if two flipped vertices are adjacent, the edge between them
                    # changes from crossing to not crossing (or vice versa)
                    
                    gain_i = 2 * same_count[i] - degrees[i]
                    gain_j = 2 * same_count[j] - degrees[j]
                    gain_k = 2 * same_count[k] - degrees[k]
                    
                    # Interaction terms: edges between flipped vertices
                    # If i-j edge exists and they have same label, flipping both makes it non-crossing
                    # This is complex; use a simpler heuristic
                    # Just use sum of individual gains as approximation
                    total_gain = gain_i + gain_j + gain_k
                    
                    # Subtract interaction: for each pair (a,b) among {i,j,k}, if adj[a,b]==1
                    # and labels[a]==labels[b], the edge was same-side (not in cut), after flip
                    # it's still same-side (both flipped), so no change.
                    # If labels[a]!=labels[b], edge was crossing, after flip still crossing.
                    # So actually no interaction term for cut change! The gain is just the sum.
                    # Wait, that's not right. Let me reconsider.
                    
                    # When we flip vertex v, the change in cut is:
                    # For each neighbor u of v: if labels[u]==labels[v] before flip, 
                    #   after flip labels[u]!=labels[v], so this edge becomes crossing (+1 to cut)
                    #   if labels[u]!=labels[v] before, after flip same, edge becomes non-crossing (-1)
                    # So gain_v = (# neighbors with same label) - (# neighbors with diff label)
                    #           = same_count[v] - (degree[v] - same_count[v])
                    #           = 2*same_count[v] - degree[v]
                    
                    # When flipping multiple vertices simultaneously, the gain is NOT simply additive
                    # because the state of neighbors changes. But for an approximation, if the
                    # flipped vertices don't interact much, it's close.
                    
                    # For exact: we need to account for edges between flipped vertices
                    # Edge (i,j): if labels[i]==labels[j] before, after both flip, still same. No change.
                    # If labels[i]!=labels[j] before, after both flip, still different. No change.
                    # So edges between flipped vertices don't change their crossing status!
                    # The gain IS simply additive: gain_i + gain_j + gain_k
                    
                    if total_gain > 0:
                        # Perform the flip
                        old_i = labels[i]
                        old_j = labels[j]
                        old_k = labels[k]
                        labels[i] = 1 - old_i
                        labels[j] = 1 - old_j
                        labels[k] = 1 - old_k
                        improved = True
                        
                        # Update same_count for i, j, k and their neighbors
                        for v in [i, j, k]:
                            old_sc = same_count[v]
                            same_count[v] = degrees[v] - old_sc
                        
                        # Update neighbors of i, j, k
                        for v in [i, j, k]:
                            for u in neighbors[v]:
                                if u in (i, j, k):
                                    continue
                                if labels[u] == labels[v]:
                                    same_count[u] += 1
                                else:
                                    same_count[u] -= 1
                        
                        break  # Stop after one successful 3-flip in this i iteration
            
            if not improved:
                break
        
        return labels

    def gw_spectral_partition():
        """Goemans-Williamson style: full spectral embedding with random hyperplane."""
        try:
            D = np.diag(degrees)
            L = D - adj
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            k = n - 1
            embedding = eigenvectors[:, 1:k+1]
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
            random_vec = np.random.randn(k)
            random_vec /= np.linalg.norm(random_vec)
            projection = embedding @ random_vec
            labels = (projection > 0).astype(np.int8)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    def gw_spectral_partition_low_rank(k=16):
        """GW-style with reduced rank for speed, multiple random hyperplanes."""
        try:
            D = np.diag(degrees)
            L = D - adj
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            k = min(k, n - 1)
            embedding = eigenvectors[:, 1:k+1]
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
            random_vec = np.random.randn(k)
            random_vec /= np.linalg.norm(random_vec)
            projection = embedding @ random_vec
            labels = (projection > 0).astype(np.int8)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    best_cut = -1
    best_labels = None

    # Generate multiple initial partitions
    init_labels_list = []
    
    if time.time() < deadline:
        init_labels_list.append(gw_spectral_partition())
    
    for _ in range(8):
        if time.time() > deadline:
            break
        init_labels_list.append(gw_spectral_partition_low_rank(k=16))
    
    try:
        D = np.diag(degrees)
        L = D - adj
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        init_labels_list.append((fiedler > 0).astype(np.int8))
    except Exception:
        pass
    
    init_labels_list.append(np.arange(n) % 2)
    init_labels_list.append(np.random.randint(0, 2, size=n, dtype=np.int8))

    # Process each initial label assignment
    for init_labels in init_labels_list:
        if time.time() > deadline:
            break
        
        labels1 = flip_greedy(init_labels, max_passes=40)
        cut1 = compute_cut(labels1)
        if cut1 > best_cut:
            best_cut = cut1
            best_labels = labels1.copy()
        
        if time.time() < deadline:
            labels2 = two_swap_search(labels1, max_passes=10)
            cut2 = compute_cut(labels2)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = labels2.copy()
        
        if time.time() < deadline:
            labels3 = tabu_search(labels2, max_iter=3000)
            cut3 = compute_cut(labels3)
            if cut3 > best_cut:
                best_cut = cut3
                best_labels = labels3.copy()

    # Additional random restarts with spectral bias
    num_random = 20
    for r in range(num_random):
        if time.time() > deadline:
            break
        
        if r % 2 == 0:
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        else:
            labels = gw_spectral_partition_low_rank(k=8)
        
        labels = flip_greedy(labels, max_passes=30)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        if time.time() < deadline:
            labels_sw = two_swap_search(labels, max_passes=8)
            cut_sw = compute_cut(labels_sw)
            if cut_sw > best_cut:
                best_cut = cut_sw
                best_labels = labels_sw.copy()
        
        if time.time() < deadline:
            start_tabu = labels_sw if cut_sw > cut else labels
            labels_tab = tabu_search(start_tabu, max_iter=2000)
            cut_tab = compute_cut(labels_tab)
            if cut_tab > best_cut:
                best_cut = cut_tab
                best_labels = labels_tab.copy()

    # Final high-precision polish pass on the best solution
    if best_labels is not None and time.time() < deadline - 5.0:
        polished = two_swap_search(best_labels, max_passes=15)
        cut_polished = compute_cut(polished)
        if cut_polished > best_cut:
            best_cut = cut_polished
            best_labels = polished.copy()
        
        if time.time() < deadline - 3.0:
            # Add 3-flip polish
            polished3 = three_flip_search(best_labels, max_passes=8)
            cut_p3 = compute_cut(polished3)
            if cut_p3 > best_cut:
                best_cut = cut_p3
                best_labels = polished3.copy()
        
        if time.time() < deadline - 2.0:
            final_tabu = tabu_search(best_labels, max_iter=5000)
            cut_final = compute_cut(final_tabu)
            if cut_final > best_cut:
                best_cut = cut_final
                best_labels = final_tabu.copy()
        
        if time.time() < deadline - 1.0:
            final_swap = two_swap_search(best_labels, max_passes=10)
            cut_final2 = compute_cut(final_swap)
            if cut_final2 > best_cut:
                best_cut = cut_final2
                best_labels = final_swap.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
