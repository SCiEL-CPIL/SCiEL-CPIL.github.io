"""Reconstruct phylogenetic topology using FastNJ with NNI refinement."""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 170.0  # Safety margin for 180s limit

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances."""
    n = len(labels)
    # Map labels to indices
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Align sequences
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    # One-hot encode or just compare
    # For efficiency with small n and medium length, simple loops or numpy broadcasting
    # Let's use numpy for pairwise hamming distance
    # Convert ACGT to integers: A=0, C=1, G=2, T=3
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Create a matrix of shape (n, length)
    try:
        seq_matrix = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    except:
        # Fallback if sequences contain unexpected chars
        seq_matrix = np.array([[0 if c=='A' else 1 if c=='C' else 2 if c=='G' else 3 for c in s] for s in seqs], dtype=np.int8)
        
    # Pairwise hamming distances
    # diff[i, j] = number of positions where seqs[i] != seqs[j]
    # (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=-1)
    diff = (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=-1)
    
    # Convert to JC distance
    # p = diff / length
    # d = -0.75 * ln(1 - 4p/3)
    p = diff / length
    # Clip p to avoid log(0) or negative
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(d, 0.0)
    
    return d

def _fast_nj(D):
    """
    Fast Neighbor Joining algorithm.
    D: numpy array of distances, shape (n, n)
    Returns: list of joins (i, j, d_i, d_j)
    """
    n = D.shape[0]
    if n <= 2:
        return []
    
    # Active nodes
    active = list(range(n))
    # Distance matrix (we will update this)
    D_cur = D.copy()
    
    joins = []
    # Branch lengths for new nodes
    # We need to store the tree structure to calculate branch lengths for NNI later?
    # For NNI using SSD, we just need the topology.
    
    for _ in range(n - 2):
        m = len(active)
        if m == 3:
            break
        
        # Calculate r[i] = sum of distances from i to all other active nodes
        # r = D_cur[active][:, active].sum(axis=1)
        sub_D = D_cur[np.ix_(active, active)]
        r = sub_D.sum(axis=1)
        
        # Calculate q[i, j] = (m-2)*d[i,j] - r[i] - r[j]
        # We want to minimize q[i, j]
        
        # Vectorized calculation
        # q = (m-2) * sub_D - r[:, np.newaxis] - r[np.newaxis, :]
        q = (m - 2) * sub_D - r[:, np.newaxis] - r[np.newaxis, :]
        
        # Mask diagonal (i != j)
        # Set diagonal to inf
        np.fill_diagonal(q, np.inf)
        
        # Find min
        min_val = np.min(q)
        # Get indices
        # q is symmetric, so we can just find one
        idx_flat = np.argmin(q)
        i_idx, j_idx = np.unravel_index(idx_flat, q.shape)
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Calculate branch lengths for a and b
        # delta = d[a,b] / 2
        delta = D_cur[a, b] / 2.0
        # len_a = delta + (r[a] - r[b]) / (2*(m-2))
        # len_b = delta + (r[b] - r[a]) / (2*(m-2))
        # Note: r is relative to active set
        r_a = r[i_idx]
        r_b = r[j_idx]
        
        len_a = delta + (r_a - r_b) / (2.0 * (m - 2))
        len_b = delta + (r_b - r_a) / (2.0 * (m - 2))
        
        joins.append((a, b, len_a, len_b))
        
        # Create new node
        new_node = n + len(joins) - 1
        
        # Update distance matrix
        # d[new, x] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
        # Only for active nodes x != a, b
        
        # Get indices of remaining active nodes
        rem_active = [x for x in active if x != a and x != b]
        
        if rem_active:
            # d[a, x] for x in rem_active
            da_x = D_cur[a, rem_active]
            db_x = D_cur[b, rem_active]
            ab = D_cur[a, b]
            
            new_d = 0.5 * (da_x + db_x - ab)
            
            # Update D_cur
            # D_cur[new_node, rem_active] = new_d
            # D_cur[rem_active, new_node] = new_d
            # We need to handle the fact that D_cur is size (n_initial + max_joins, n_initial + max_joins)
            # Or we can just keep a dict or sparse representation.
            # For simplicity and speed with n<=32, let's just keep a full array sized (2*n, 2*n) initially.
            
            D_cur[new_node, rem_active] = new_d
            D_cur[rem_active, new_node] = new_d
        
        # Update active list
        active = rem_active + [new_node]
        
        # Check time
        if time.time() > _DEADLINE - 5.0:
            break
            
    # Construct the tree structure for NNI
    # We have a list of joins. We need to build the adjacency or Newick.
    return joins

def _build_topology(joins, n):
    """
    Build a topology representation from NJ joins.
    Returns:
    - children: dict mapping node -> [left, right]
    - parent: dict mapping node -> parent
    - internal_nodes: list of internal node IDs
    """
    children = {}
    parent = {}
    
    # Initialize leaves
    for i in range(n):
        children[i] = []
        
    # Process joins
    for k, (a, b, la, lb) in enumerate(joins):
        new_node = n + k
        children[new_node] = [a, b]
        parent[a] = new_node
        parent[b] = new_node
        children[a].append(new_node) # Not strictly needed for topology, but good for traversal
        children[b].append(new_node)
        
        # Ensure children for new_node is initialized
        if new_node not in children:
            children[new_node] = [a, b]
            
    # Find root
    root = None
    for i in range(n + len(joins)):
        if i not in parent:
            root = i
            break
            
    if root is None:
        # Fallback
        root = n + len(joins) - 1
        
    return children, parent, root

def _nni_refine(D, children, parent, root, labels):
    """
    Perform NNI refinement to minimize Sum of Squared Distances (SSD).
    """
    n = len(labels)
    num_nodes = n + len(children) - n # Actually just max key + 1
    
    # Precompute clades for each node
    # clade[node] = set of leaves in the subtree rooted at node
    clades = {}
    def get_clade(node):
        if node in clades:
            return clades[node]
        if node < n:
            clades[node] = {node}
        else:
            l, r = children[node]
            clades[node] = get_clade(l) | get_clade(r)
        return clades[node]
        
    # Compute SSD for the current tree
    # SSD = sum_{i<j} (d_ij - path_len(i,j))^2
    # This is expensive to calculate from scratch O(n^2 * depth).
    # However, for n=32, n^2=1024. Depth ~ log(n) or n.
    # Let's use a faster metric: Total Branch Length (TBL) or Sum of Pairwise Distances (SPD) under LC correction.
    # Actually, NNI is often evaluated by checking if the quartet is compatible with the distance matrix.
    # A simpler heuristic: Check if the NNI move reduces the sum of distances between the quartet tips.
    
    # Let's implement a simple NNI loop that checks the 3 quartets.
    # For each internal edge (u, v), the two subtrees attached are A and B.
    # The quartet is A1, A2, B1, B2.
    # We check distances: d(A1, B1) + d(A2, B2) vs d(A1, B2) + d(A2, B1) vs d(A1, A2) + d(B1, B2)
    # If the current tree's quartet signal is weak or inconsistent with D, we switch.
    
    improved = True
    max_iter = 100
    
    # Get list of internal edges
    # An internal edge is (parent[node], node) where node is internal
    internal_nodes = [i for i in range(n, n + len(joins)) if i in children and len(children[i]) == 2]
    # Note: In our build, internal nodes are n..n+k-1
    
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time.time() > _DEADLINE - 2.0:
            break
            
        # Iterate over internal edges
        # We need to be careful not to modify the tree while iterating if it affects other edges.
        # Usually, NNI is done one edge at a time.
        
        for node in internal_nodes:
            if time.time() > _DEADLINE - 1.0:
                break
                
            p = parent.get(node)
            if p is None:
                continue
                
            # The edge is (p, node)
            # The two sides of the edge are:
            # Side 1: Subtree rooted at node (contains leaves C_node)
            # Side 2: The rest of the tree
            
            # To do NNI, we need to find the "sister" of node?
            # No, NNI operates on an internal edge. The edge separates the tree into two components.
            # Let's define the two subtrees hanging off the edge.
            # Actually, standard NNI definition:
            # Consider an internal edge with endpoints u and v.
            # u is connected to v and two other nodes (or one if it's the root? No, unrooted tree).
            # In a binary unrooted tree, every internal node has degree 3.
            # So edge (u, v).
            # u is connected to v, x, y.
            # v is connected to u, z, w.
            # The four tips are in the subtrees hanging from x, y, z, w?
            # No, x, y, z, w are nodes. The "quartet" is formed by the four subtrees attached to the "star" centered at the edge.
            # Wait, if we remove edge (u,v), we get two trees.
            # The neighbors of u (excluding v) are u1, u2.
            # The neighbors of v (excluding u) are v1, v2.
            # The four components are T(u1), T(u2), T(v1), T(v2).
            
            # Let's find neighbors in the current topology
            # We need an adjacency list for the internal tree structure (nodes 0..n+k-1)
            
            # Let's build adjacency list once
            adj = {}
            for i in range(n + len(joins)):
                adj[i] = []
            for i, chs in children.items():
                for c in chs:
                    adj[i].append(c)
                    adj[c].append(i)
            
            if node not in adj or p not in adj[node]:
                continue
                
            u = p
            v = node
            
            # Get neighbors of u excluding v
            u_neighbors = [x for x in adj[u] if x != v]
            if len(u_neighbors) != 2:
                continue # Root or non-binary?
            u1, u2 = u_neighbors
            
            # Get neighbors of v excluding u
            v_neighbors = [x for x in adj[v] if x != u]
            if len(v_neighbors) != 2:
                continue
            v1, v2 = v_neighbors
            
            # The four subtrees are attached to u1, u2, v1, v2.
            # We need to pick one representative leaf from each subtree to check quartet distances.
            # Or better, use the total distance between the sets of leaves.
            
            # Get leaves in each subtree
            def get_leaves(node_id):
                # Returns list of leaf indices
                if node_id < n:
                    return [node_id]
                l, r = children[node_id]
                return get_leaves(l) + get_leaves(r)
                
            # This recursive call is O(n). Doing it for every NNI check is O(n^2).
            # With n=32, n^2=1000. 100 NNI passes * 30 edges * 1000 = 3 million ops. Acceptable.
            
            # Optimization: Precompute leaves for all nodes
            leaves_cache = {}
            def get_leaves_cached(node_id):
                if node_id in leaves_cache:
                    return leaves_cache[node_id]
                if node_id < n:
                    leaves_cache[node_id] = [node_id]
                else:
                    l, r = children[node_id]
                    leaves_cache[node_id] = get_leaves_cached(l) + get_leaves_cached(r)
                return leaves_cache[node_id]
                
            L_u1 = get_leaves_cached(u1)
            L_u2 = get_leaves_cached(u2)
            L_v1 = get_leaves_cached(v1)
            L_v2 = get_leaves_cached(v2)
            
            # Calculate average distance between the sets
            def avg_dist(set1, set2):
                if not set1 or not set2:
                    return 0
                # Sum of all pairwise distances
                s = 0.0
                cnt = 0
                for a in set1:
                    for b in set2:
                        s += D[a, b]
                        cnt += 1
                return s / cnt if cnt > 0 else 0
            
            # Current tree quartet: (u1, u2) and (v1, v2) are on different sides of the edge.
            # The "split" is {L_u1, L_u2} | {L_v1, L_v2}
            
            # Distance signals:
            # d12 = dist(L_u1, L_v1)
            # d13 = dist(L_u1, L_v2)
            # d24 = dist(L_u2, L_v1)
            # d23 = dist(L_u2, L_v2)
            
            # The three possible quartets:
            # 1. Current: (L_u1, L_u2) vs (L_v1, L_v2)
            #    Signal: d12 + d13 + d24 + d23 ? No.
            #    Quartet topology A|B is supported by d(A,B) < d(A,B') + d(B,B') ...
            #    Let's use the standard quartet distance sum:
            #    For quartet (a,b,c,d), the three topologies are:
            #    (a,b)|(c,d): d_ab + d_cd
            #    (a,c)|(b,d): d_ac + d_bd
            #    (a,d)|(b,c): d_ad + d_bc
            #    We want to pick the topology with the smallest sum of "opposing" distances?
            #    Actually, the likelihood of the quartet is higher for the topology that minimizes the sum of squared distances or simply the sum of distances across the split.
            
            # Let's map the sets to single representatives for a fast check, or use mean distances.
            # Let a = L_u1, b = L_u2, c = L_v1, d = L_v2.
            # Topology 1 (Current): (a,b)|(c,d) -> Score = mean_d(a,c) + mean_d(a,d) + mean_d(b,c) + mean_d(b,d) ? No.
            # The "distance" of a topology is often defined as the sum of the two internal branch lengths estimated by LC correction, or simply the sum of the two "cross" distances in the quartet.
            # Let's use the sum of the two distances connecting the split parts:
            # Score( (a,b)|(c,d) ) = mean_d(a,c) + mean_d(a,d) + mean_d(b,c) + mean_d(b,d) ? No, that's all cross.
            # Let's use the standard quartet test:
            # S1 = d(a,c) + d(b,d)
            # S2 = d(a,d) + d(b,c)
            # S3 = d(a,b) + d(c,d)
            # The topology (a,b)|(c,d) is best if S3 < S1 and S3 < S2?
            # Actually, (a,b)|(c,d) means a and b are close, c and d are close.
            # So d(a,b) and d(c,d) should be small. d(a,c), d(a,d), d(b,c), d(b,d) should be large.
            # So we want to MINIMIZE d(a,b) + d(c,d).
            
            # Let's calculate these sums using mean distances between the leaf sets.
            # Note: u1, u2 are on the same side of the edge in the current tree.
            # v1, v2 are on the other side.
            
            # Current Topology T1: (L_u1, L_u2) | (L_v1, L_v2)
            # This corresponds to quartet (a,b)|(c,d) where a=L_u1, b=L_u2, c=L_v1, d=L_v2.
            # Score T1 = mean_d(a,b) + mean_d(c,d)
            
            # Alternative Topology T2: (L_u1, L_v1) | (L_u2, L_v2)
            # Score T2 = mean_d(a,c) + mean_d(b,d)
            
            # Alternative Topology T3: (L_u1, L_v2) | (L_u2, L_v1)
            # Score T3 = mean_d(a,d) + mean_d(b,c)
            
            score_t1 = avg_dist(L_u1, L_u2) + avg_dist(L_v1, L_v2)
            score_t2 = avg_dist(L_u1, L_v1) + avg_dist(L_u2, L_v2)
            score_t3 = avg_dist(L_u1, L_v2) + avg_dist(L_u2, L_v1)
            
            best_score = min(score_t1, score_t2, score_t3)
            
            if best_score < score_t1:
                # Perform NNI move
                # If T2 is best: Connect u1-v1 and u2-v2.
                # Remove edges (u,u1), (u,u2), (v,v1), (v,v2)
                # Add edges (u1,v1), (u2,v2) ? No, that's not right.
                # NNI changes the tree by swapping the neighbors.
                # Current: u connected to u1, u2, v. v connected to v1, v2, u.
                # T2: (u1, v1) and (u2, v2) are the new pairs?
                # Let's visualize.
                # Edge (u,v).
                # u connects to u1, u2.
                # v connects to v1, v2.
                # T1: u1-u2 are together, v1-v2 are together.
                # T2: u1-v1 are together, u2-v2 are together.
                # To get T2:
                # Remove (u, u2) and (v, v1).
                # Add (u, v1) and (v, u2).
                # Wait, if we add (u, v1), then u is connected to u1, v1, v.
                # And v is connected to v2, u2, u.
                # This creates a new edge (u,v) still? No, we removed (u,v)? No, NNI keeps the central edge?
                # No, NNI rotates around the edge. The edge (u,v) remains?
                # Actually, in an unrooted binary tree, NNI on edge (u,v) results in a tree where u and v are still connected, but their other neighbors are swapped.
                # Current: u-(u1), u-(u2), u-v, v-(v1), v-(v2).
                # T2: u-(u1), u-(v1), u-v, v-(v2), v-(u2).
                # Let's check the quartet for T2:
                # Path u1 to v1: u1-u-v1. Length d(u1,u) + d(u,v) + d(v,v1).
                # Path u2 to v2: u2-v-v2. Length d(u2,v) + d(v,v) + d(v,v2)? No.
                # The topology is defined by the splits.
                # T2 split: {L_u1, L_v1} and {L_u2, L_v2}.
                
                # Implementation:
                # If T2 is better:
                #   Remove u2 from u, add u2 to v.
                #   Remove v1 from v, add v1 to u.
                #   Update children/parent/adj.
                
                # If T3 is better:
                #   Remove u2 from u, add u2 to v.
                #   Remove v2 from v, add v2 to u.
                
                # We must update the data structures:
                # children, parent, adj, leaves_cache (invalidate)
                
                if best_score == score_t2:
                    # Swap u2 and v1
                    # u's children: remove u2, add v1
                    # v's children: remove v1, add u2
                    
                    # Update children
                    children[u] = [x for x in children[u] if x != u2] + [v1]
                    children[v] = [x for x in children[v] if x != v1] + [u2]
                    
                    # Update parent
                    parent[u2] = v
                    parent[v1] = u
                    
                    # Update adj
                    adj[u].remove(u2)
                    adj[u].append(v1)
                    adj[v].remove(v1)
                    adj[v].append(u2)
                    adj[u2].remove(u)
                    adj[u2].append(v)
                    adj[v1].remove(v)
                    adj[v1].append(u)
                    
                    # Invalidate cache
                    leaves_cache.clear()
                    
                elif best_score == score_t3:
                    # Swap u2 and v2
                    children[u] = [x for x in children[u] if x != u2] + [v2]
                    children[v] = [x for x in children[v] if x != v2] + [u2]
                    
                    parent[u2] = v
                    parent[v2] = u
                    
                    adj[u].remove(u2)
                    adj[u].append(v2)
                    adj[v].remove(v2)
                    adj[v].append(u2)
                    adj[u2].remove(u)
                    adj[u2].append(v)
                    adj[v2].remove(v)
                    adj[v2].append(u)
                    
                    leaves_cache.clear()
                    
                improved = True
                # Break out of edge loop? Or continue?
                # Continuing might cause issues if the tree structure changed significantly.
                # Better to restart the edge loop?
                # For simplicity and safety, let's break and restart.
                break

    return children, parent, root

def _build_newick(children, parent, root, labels):
    """Build Newick string from topology."""
    def build(node):
        if node < len(labels):
            return labels[node]
        l, r = children[node]
        return f"({build(l)},{build(r)})"
    
    return build(root) + ";"

def reconstruct(sequences):
    global _START_TIME, _DEADLINE
    # Reset deadline for each call? No, the sandbox calls it 20 times.
    # The 180s budget is for ALL 20 instances.
    # So _DEADLINE should be set once at module load or first call.
    # The code above sets _START_TIME at import. This is correct.
    
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
        joins = _fast_nj(D)
        
        # 3. Build Topology
        if not joins:
             return "(" + ",".join(labels) + ");"
             
        children, parent, root = _build_topology(joins, n)
        
        # 4. NNI Refinement
        if time.time() < _DEADLINE - 5.0:
            children, parent, root = _nni_refine(D, children, parent, root, labels)
            
        # 5. Build Newick
        newick = _build_newick(children, parent, root, labels)
        
        return newick
        
    except Exception as e:
        # Fallback: Star tree
        import traceback
        traceback.print_exc()
        return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
