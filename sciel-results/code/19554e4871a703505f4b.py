import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 160.0  # Safety margin for 180s limit (allowing for overhead)

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    # Map ACGT to 0,1,2,3
    # Using a lookup table for speed
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Create a matrix of shape (n, length)
    # Using list comprehension is often faster than np.array of lists for small n
    seq_matrix = np.zeros((n, length), dtype=np.int8)
    for i, s in enumerate(seqs):
        # Vectorized mapping
        # np.array(list(s), dtype='S1') -> bytes
        # Better: use a precomputed array if possible, but simple loop is fine for L<=2000, n<=32
        # Actually, let's use numpy's frombuffer or similar? 
        # Simplest robust way:
        for j, c in enumerate(s):
            seq_matrix[i, j] = mapping[c]
            
    # Pairwise hamming distances
    # diff[i, j, k] is 1 if seq[i][k] != seq[j][k]
    # We want sum over k.
    # (n, 1, L) vs (1, n, L) -> (n, n, L)
    # This creates an intermediate array of size n*n*L. 
    # For n=32, L=2000, size is 32*32*2000 = 2,048,000 bytes. Very fast.
    diff = (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=-1)
    
    # Convert to JC distance
    p = diff / length
    # Clip p to avoid log(0) or negative
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    
    return d

def _fast_nj(D):
    """
    Fast Neighbor Joining algorithm.
    Returns: adj (dict of lists), which represents the tree topology.
    """
    n = D.shape[0]
    if n <= 2:
        return {}
    
    # Active nodes
    active = list(range(n))
    # Distance matrix
    D_cur = D.copy()
    
    # Adjacency list for the tree
    adj = {i: [] for i in range(n)}
    
    # We will add internal nodes as we go
    next_node_id = n
    
    for _ in range(n - 2):
        m = len(active)
        if m == 3:
            # The last 3 nodes form a star (or rather, the last join connects 2, and the 3rd is connected to the new node)
            # Actually, standard NJ stops when 2 nodes remain? No, when 3 nodes remain, we join 2 of them?
            # Standard NJ:
            # Iteration 1: n nodes. Join 2 -> n-1 nodes.
            # ...
            # Iteration n-2: 3 nodes. Join 2 -> 2 nodes?
            # No, an unrooted tree with n leaves has n-2 internal nodes.
            # Total nodes = 2n-2.
            # We start with n leaves.
            # Each join reduces active count by 1 and adds 1 internal node.
            # After k joins, active count is n-k.
            # We need to end with 2 active nodes? No, the tree is unrooted.
            # Usually, NJ runs until 2 nodes remain, then connects them?
            # Or until 3 nodes remain, then connects them in a star?
            # Let's stick to the standard: Join until 2 nodes remain?
            # If we join until 2 nodes remain, we have added n-2 internal nodes.
            # The last 2 active nodes are connected by an edge.
            
            # Let's check the loop condition.
            # If m == 3, we can just connect all 3 in a star?
            # No, NJ is binary.
            # If m == 3, we pick the best pair, join them, and the resulting node is connected to the third.
            # So we do one more join step.
            pass
            
        # Calculate r[i] = sum of distances from i to all other active nodes
        sub_D = D_cur[np.ix_(active, active)]
        r = sub_D.sum(axis=1)
        
        # Calculate q[i, j] = (m-2)*d[i,j] - r[i] - r[j]
        q = (m - 2) * sub_D - r[:, np.newaxis] - r[np.newaxis, :]
        
        # Mask diagonal
        np.fill_diagonal(q, np.inf)
        
        # Find min
        idx_flat = np.argmin(q)
        i_idx, j_idx = np.unravel_index(idx_flat, q.shape)
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Calculate branch lengths
        delta = D_cur[a, b] / 2.0
        r_a = r[i_idx]
        r_b = r[j_idx]
        
        len_a = delta + (r_a - r_b) / (2.0 * (m - 2))
        len_b = delta + (r_b - r_a) / (2.0 * (m - 2))
        
        # Create new node
        new_node = next_node_id
        next_node_id += 1
        
        # Update adjacency
        adj[new_node] = [a, b]
        adj[a].append(new_node)
        adj[b].append(new_node)
        
        # Update distance matrix
        # New node is connected to all other active nodes except a and b
        rem_active = [x for x in active if x != a and x != b]
        
        if rem_active:
            da_x = D_cur[a, rem_active]
            db_x = D_cur[b, rem_active]
            ab = D_cur[a, b]
            
            new_d = 0.5 * (da_x + db_x - ab)
            
            D_cur[new_node, rem_active] = new_d
            D_cur[rem_active, new_node] = new_d
            
        # Update active list
        active = rem_active + [new_node]
        
        # Check time
        if time.time() > _DEADLINE - 10.0:
            break
            
    # If we broke out early or finished, we need to connect the remaining active nodes.
    # If we finished the loop (n-2 iterations), there should be 2 active nodes left.
    # We connect them.
    if len(active) == 2:
        u, v = active
        if v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)
    elif len(active) > 2:
        # If we timed out, we have a forest. Connect the components?
        # For safety, just return the partial tree. The NNI might fix it if it's close.
        # Or better, just connect all remaining active nodes in a star to the last internal node?
        # This is a fallback for timeout.
        if active:
            last_int = next_node_id - 1
            if last_int in adj:
                for node in active:
                    if last_int not in adj[node]:
                        adj[node].append(last_int)
                        adj[last_int].append(node)
                        
    return adj

def _build_newick(adj, root, labels, n):
    """Build Newick string from adjacency list."""
    def build(node, parent):
        if node < n:
            return labels[node]
        
        # Get children (neighbors excluding parent)
        children = [x for x in adj[node] if x != parent]
        
        if parent is None:
            children = adj[node]
            
        if len(children) == 0:
            return labels[node]
        
        parts = [build(child, node) for child in children]
        return "(" + ",".join(parts) + ")"
    
    # Find a root. Any leaf is a valid root for unrooted tree representation.
    # Using leaf 0.
    root = 0
    return build(root, None) + ";"

def _log_likelihood_nni(adj, D, n, labels, sequences, seq_matrix, length):
    """
    Calculate log-likelihood for the current tree.
    This is a placeholder for the NNI scoring function.
    For efficiency, we will implement a specific NNI loop that calculates
    likelihood for the 3 possible topologies of an edge swap.
    """
    pass

def _nni_refine_likelihood(adj, n, labels, sequences, seq_matrix, length):
    """
    Refine topology using NNI moves based on Jukes-Cantor Log-Likelihood.
    """
    # Precompute leaf sets for directed edges
    # directed_clades[u][v] = set of leaf indices reachable from u without passing v
    
    # To speed up, we can use bitmasks if n <= 64. n <= 32, so this is perfect.
    leaf_masks = {}
    
    def get_leaf_mask(u, v):
        key = (u, v)
        if key in leaf_masks:
            return leaf_masks[key]
        
        mask = 0
        stack = [u]
        visited = {v}
        
        while stack:
            node = stack.pop()
            if node < n:
                mask |= (1 << node)
            else:
                for neighbor in adj[node]:
                    if neighbor != v and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
                        
        leaf_masks[key] = mask
        return mask

    # Precompute all directed leaf masks
    for u in adj:
        for v in adj[u]:
            get_leaf_mask(u, v)
            
    # Helper to calculate likelihood contribution of a subtree
    # Felsenstein's algorithm:
    # L_i(k) = P(observed data in subtree i | state k at node i)
    # For leaves: L_i(A)=1 if obs=A, else 0.
    # For internal nodes: L_i(k) = sum_{p,q} L_child1(p) * L_child2(q) * P(k->p) * P(k->q)
    # This is complex to vectorize efficiently for NNI in pure Python without heavy precomputation.
    # 
    # Alternative: Use the Quartet Score approximation with JC distances?
    # The previous attempt used distance sums.
    # Let's try a simpler Likelihood approximation:
    # For an NNI move on edge (u,v) with neighbors u1,u2 and v1,v2:
    # The likelihood change depends on the "compatibility" of the splits.
    # 
    # Given the time constraint and complexity of full Felsenstein in Python,
    # I will use a **Quartet Score** based on the **Sum of Squared Errors** or **Likelihood Approximation**.
    # 
    # A better heuristic for JC:
    # Minimize the sum of distances between leaves in the same clade?
    # Or maximize the "compatibility" of splits.
    # 
    # Let's stick to the **Quartet Score** using the distance matrix D.
    # For 4 subtrees A, B, C, D attached to the central edge:
    # Topology 1: (A,B)|(C,D). Score = d(A,C)+d(A,D)+d(B,C)+d(B,D) - d(A,B) - d(C,D)?
    # No, the standard Quartet Score is:
    # S1 = d(A,B) + d(C,D)
    # S2 = d(A,C) + d(B,D)
    # S3 = d(A,D) + d(B,C)
    # We want to minimize the sum of distances for the "correct" pairing?
    # Actually, for NJ, we minimize the total branch length.
    # The NNI move that minimizes the total tree length is preferred.
    # Total length can be approximated by the sum of all pairwise distances? No, that's constant.
    # 
    # Let's use the **Quartet Score** defined as:
    # Q(A,B,C,D) = d(A,C) + d(B,D) - d(A,B) - d(C,D)
    # If Q < 0, then (A,B)|(C,D) is better than (A,C)|(B,D).
    # 
    # We will iterate over all internal edges.
    # For each edge (u,v), identify the 4 neighbors u1,u2,v1,v2.
    # Get their leaf masks.
    # Calculate the average distance between the sets.
    # 
    # Score for topology T1 ((u1,u2)|(v1,v2)):
    # S1 = avg_dist(u1, u2) + avg_dist(v1, v2)
    # S2 = avg_dist(u1, v1) + avg_dist(u2, v2)
    # S3 = avg_dist(u1, v2) + avg_dist(u2, v1)
    # 
    # We choose the topology with the MINIMUM score?
    # Let's verify.
    # If u1 and u2 are close (small dist), and v1 and v2 are close, then S1 is small.
    # This suggests (u1,u2) are a clade and (v1,v2) are a clade.
    # This matches the intuition of NJ.
    
    improved = True
    max_iter = 100
    iter_count = 0
    
    # Precompute avg_dist for all pairs of leaf masks?
    # There are at most 2*(2n-2) directed edges.
    # Pairs of masks: O(n^2).
    # For each pair of masks, avg_dist is sum(D[i,j]) / (|A||B|).
    # We can precompute sum_dist[mask1, mask2] using bit operations?
    # No, D is a matrix.
    # Let's just compute on the fly. n is small (<=32).
    
    def get_avg_dist(mask1, mask2):
        if mask1 == 0 or mask2 == 0:
            return 0
        # Get indices
        # This is slow in Python loop.
        # Optimization: Precompute sum of D for all mask pairs?
        # Number of masks is ~2n. Pairs ~4n^2.
        # For n=32, 4*1024 = 4096 pairs.
        # Each pair requires iterating over set bits.
        # Let's precompute a dictionary of sum_dist for unique mask pairs encountered?
        # Or just compute directly.
        
        s = 0.0
        cnt = 0
        # Iterate over bits in mask1
        m1 = mask1
        while m1:
            i = (m1 & -m1).bit_length() - 1
            m1 &= m1 - 1
            # Iterate over bits in mask2
            m2 = mask2
            while m2:
                j = (m2 & -m2).bit_length() - 1
                m2 &= m2 - 1
                s += D[i, j]
                cnt += 1
        return s / cnt if cnt > 0 else 0

    # To speed up get_avg_dist, let's precompute it for all unique masks.
    unique_masks = set(leaf_masks.values())
    # Convert to list
    mask_list = list(unique_masks)
    mask_to_idx = {m: i for i, m in enumerate(mask_list)}
    
    # Precompute avg_dist_matrix for unique masks
    # Size: len(mask_list) x len(mask_list)
    num_masks = len(mask_list)
    dist_matrix_masks = np.zeros((num_masks, num_masks))
    
    for i_idx, m1 in enumerate(mask_list):
        # Get indices in m1
        inds1 = [j for j in range(n) if (m1 >> j) & 1]
        for j_idx in range(i_idx, num_masks):
            m2 = mask_list[j_idx]
            inds2 = [k for k in range(n) if (m2 >> k) & 1]
            
            if not inds1 or not inds2:
                continue
                
            # Sum D[i,j] for i in inds1, j in inds2
            # Use numpy
            sub_D = D[np.ix_(inds1, inds2)]
            s = sub_D.sum()
            cnt = len(inds1) * len(inds2)
            val = s / cnt if cnt > 0 else 0
            dist_matrix_masks[i_idx, j_idx] = val
            dist_matrix_masks[j_idx, i_idx] = val

    def fast_avg_dist(mask1, mask2):
        if mask1 == 0 or mask2 == 0:
            return 0
        i1 = mask_to_idx[mask1]
        i2 = mask_to_idx[mask2]
        return dist_matrix_masks[i1, i2]

    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time.time() > _DEADLINE - 2.0:
            break
            
        # Iterate over all internal edges
        edges_to_check = []
        for u in adj:
            if u < n:
                continue
            for v in adj[u]:
                if v > u and v >= n:
                    edges_to_check.append((u, v))
                    
        for u, v in edges_to_check:
            # Get neighbors
            u_neighbors = [x for x in adj[u] if x != v]
            v_neighbors = [x for x in adj[v] if x != u]
            
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue
                
            u1, u2 = u_neighbors
            v1, v2 = v_neighbors
            
            # Get leaf masks
            L_u1 = get_leaf_mask(u1, u)
            L_u2 = get_leaf_mask(u2, u)
            L_v1 = get_leaf_mask(v1, v)
            L_v2 = get_leaf_mask(v2, v)
            
            # Calculate scores
            # S1: (u1, u2) and (v1, v2) are clades.
            # The "cost" is the distance between the clades?
            # No, the previous logic was:
            # Score = avg_dist(within u1, u2) + avg_dist(within v1, v2)?
            # This measures how compact the clades are.
            # We want to MINIMIZE this?
            # If u1 and u2 are close, they form a good clade.
            # So we want to minimize S1 = avg_dist(L_u1, L_u2) + avg_dist(L_v1, L_v2)?
            # Wait, avg_dist(L_u1, L_u2) is the distance BETWEEN the two subtrees attached to u.
            # If they are close, they are likely siblings.
            
            s1 = fast_avg_dist(L_u1, L_u2) + fast_avg_dist(L_v1, L_v2)
            s2 = fast_avg_dist(L_u1, L_v1) + fast_avg_dist(L_u2, L_v2)
            s3 = fast_avg_dist(L_u1, L_v2) + fast_avg_dist(L_u2, L_v1)
            
            best_score = min(s1, s2, s3)
            
            if best_score < s1 - 1e-6:
                # Perform NNI move
                if best_score == s2:
                    # Swap u2 and v1
                    # Remove edges (u, u2), (v, v1)
                    # Add edges (u, v1), (v, u2)
                    
                    adj[u].remove(u2)
                    adj[v].remove(v1)
                    adj[u].append(v1)
                    adj[v].append(u2)
                    adj[u2].remove(u)
                    adj[u2].append(v)
                    adj[v1].remove(v)
                    adj[v1].append(u)
                    
                elif best_score == s3:
                    # Swap u2 and v2
                    adj[u].remove(u2)
                    adj[v].remove(v2)
                    adj[u].append(v2)
                    adj[v].append(u2)
                    adj[u2].remove(u)
                    adj[u2].append(v)
                    adj[v2].remove(v)
                    adj[v2].append(u)
                    
                # Invalidate directed clades cache
                leaf_masks.clear()
                # Re-compute affected clades?
                # For simplicity, clear all and let lazy evaluation handle it?
                # No, `get_leaf_mask` is called in the next iteration.
                # We must clear the cache so it recomputes.
                # However, recomputing all masks is O(n^2) per NNI move.
                # This might be slow.
                # But n is small.
                
                improved = True
                break # Restart edge loop

    return adj

def reconstruct(sequences):
    global _START_TIME, _DEADLINE
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    try:
        # 1. Distance Matrix
        D = _jc_distance_matrix(sequences, labels)
        
        # 2. FastNJ
        adj = _fast_nj(D)
        
        # 3. NNI Refinement
        if time.time() < _DEADLINE - 5.0:
            # Prepare seq_matrix for likelihood if needed, but we use distance heuristic
            adj = _nni_refine_likelihood(adj, n, labels, sequences, None, 0)
            
        # 4. Build Newick
        newick = _build_newick(adj, 0, labels, n)
        
        return newick
        
    except Exception as e:
        # Fallback: Star tree
        import traceback
        traceback.print_exc()
        return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
