import numpy as np
import math
import time
from itertools import combinations

class Node:
    __slots__ = ['left', 'right', 'label', 'parent']
    
    def __init__(self, left=None, right=None, label=None):
        self.left = left
        self.right = right
        self.label = label
        self.parent = None
        
    def is_leaf(self):
        return self.label is not None

def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using vectorized operations."""
    n = len(labels)
    L = len(sequences[labels[0]])
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Encode sequences as integer matrix
    seq_matrix = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        # Vectorized encoding
        for j, base in enumerate(seq):
            seq_matrix[i, j] = base_map[base]
    
    # Compute pairwise Hamming distances vectorized
    # This creates an n x n x L boolean array, which is memory efficient for n<=32, L<=2000
    diff = (seq_matrix[:, None, :] != seq_matrix[None, :, :])
    hamming = diff.sum(axis=2)
    
    # JC correction
    p = hamming.astype(float) / L
    p = np.clip(p, 0, 0.74)  # Avoid log(0) or negative args
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(dist, 0.0)
    
    return dist, seq_matrix

def _quartet_scores(dist):
    """
    Compute quartet scores vectorized.
    Returns a dictionary of split tuples to total support score.
    """
    n = len(dist)
    if n < 4:
        return {}
    
    # Vectorized computation of all quartet scores
    # We want to sum max(s1, s2, s3) for each quartet where max > 0
    # and attribute it to the winning split.
    
    # To avoid huge memory usage for n=32 (32 choose 4 = 35960 quartets, which is small),
    # we can just loop over quartets but use numpy for the arithmetic if needed, 
    # or just use a fast python loop since n is small.
    # The previous failure was likely due to time or a bug in the loop logic/indexing.
    # Let's stick to a clean loop for clarity and correctness, as n=32 is small enough.
    
    scores = {}
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    s1 = dist[a,b] + dist[c,d] - dist[a,c] - dist[b,d]
                    s2 = dist[a,c] + dist[b,d] - dist[a,b] - dist[c,d]
                    s3 = dist[a,d] + dist[b,c] - dist[a,b] - dist[c,d]
                    
                    # Determine winner
                    if s1 >= s2 and s1 >= s3:
                        if s1 > 0:
                            # Split {a,b}|{c,d}
                            key = (tuple(sorted((a,b))), tuple(sorted((c,d))))
                            # Canonicalize key so (smaller_set, larger_set) or similar
                            if key[0] > key[1]:
                                key = (key[1], key[0])
                            scores[key] = scores.get(key, 0.0) + s1
                    elif s2 >= s1 and s2 >= s3:
                        if s2 > 0:
                            # Split {a,c}|{b,d}
                            key = (tuple(sorted((a,c))), tuple(sorted((b,d))))
                            if key[0] > key[1]:
                                key = (key[1], key[0])
                            scores[key] = scores.get(key, 0.0) + s2
                    else:
                        if s3 > 0:
                            # Split {a,d}|{b,c}
                            key = (tuple(sorted((a,d))), tuple(sorted((b,c))))
                            if key[0] > key[1]:
                                key = (key[1], key[0])
                            scores[key] = scores.get(key, 0.0) + s3
                            
    return scores

def _fdq_build_tree(dist, labels, quartet_scores):
    """
    Build tree using Fast Distance-based Quartet approach.
    Start with a star tree, then iteratively add internal edges
    that are most strongly supported by quartet scores.
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return Node(label=labels[0])
        elif n == 2:
            root = Node()
            l = Node(label=labels[0])
            r = Node(label=labels[1])
            l.parent = root
            r.parent = root
            root.left = l
            root.right = r
            return root
        else:
            # 3 taxa: any binary tree is fine
            root = Node()
            l = Node(label=labels[0])
            mid = Node()
            r1 = Node(label=labels[1])
            r2 = Node(label=labels[2])
            mid.left = r1
            mid.right = r2
            r1.parent = mid
            r2.parent = mid
            root.left = l
            root.right = mid
            l.parent = root
            mid.parent = root
            return root
    
    # Start with NJ tree as initial estimate, then refine with quartet info
    nj_tree = _nj_build_tree(dist, labels)
    
    # Use quartet scores to identify potential topological corrections
    # For each internal edge in the NJ tree, check if quartet evidence
    # supports a different topology nearby (NNI moves)
    
    refined_tree = _quartet_refine(nj_tree, dist, labels, quartet_scores)
    return refined_tree

def _nj_build_tree(dist, labels):
    """Standard Neighbor-Joining tree construction."""
    n = len(labels)
    if n == 1:
        return Node(label=labels[0])
    if n == 2:
        root = Node()
        l = Node(label=labels[0])
        r = Node(label=labels[1])
        l.parent = root
        r.parent = root
        root.left = l
        root.right = r
        return root
    
    # Use a list of active nodes and a full-size distance matrix that we update
    # This avoids the complex sub-matrix resizing which caused the index error.
    
    # Map from active node ID to Node object (or index)
    # We will build the tree by creating Node objects as we merge.
    
    # active_nodes: list of IDs. ID < n are leaves. ID >= n are internal nodes.
    active_nodes = list(range(n))
    # dist_matrix: full size n + (n-2) x n + (n-2)
    # Initialize with leaf distances
    max_nodes = 2 * n
    d_mat = np.zeros((max_nodes, max_nodes))
    d_mat[:n, :n] = dist
    
    # node_objects: map ID -> Node
    node_objects = {i: Node(label=labels[i]) for i in range(n)}
    
    next_id = n
    
    while len(active_nodes) > 2:
        k = len(active_nodes)
        
        # Compute row sums r
        # Only consider active nodes
        idxs = np.array(active_nodes)
        sub_d = d_mat[np.ix_(idxs, idxs)]
        r = sub_d.sum(axis=1)
        
        # Compute Q matrix
        # Q_ij = (k-2)*d_ij - r_i - r_j
        Q = (k - 2) * sub_d - r[:, None] - r[None, :]
        
        # Find min Q (excluding diagonal)
        np.fill_diagonal(Q, np.inf)
        min_idx = np.argmin(Q)
        i_local, j_local = np.unravel_index(min_idx, (k, k))
        
        if i_local == j_local:
            break # Should not happen
            
        i_global = active_nodes[i_local]
        j_global = active_nodes[j_local]
        
        d_ij = d_mat[i_global, j_global]
        r_i = r[i_local]
        r_j = r[j_local]
        
        # Branch lengths
        d_i = (r_i - r_j) / (2 * (k - 2))
        d_j = d_ij - d_i
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Create Node object
        node_i = node_objects[i_global]
        node_j = node_objects[j_global]
        
        new_node = Node()
        new_node.left = node_i
        new_node.right = node_j
        node_i.parent = new_node
        node_j.parent = new_node
        node_objects[new_id] = new_node
        
        # Update distances from new node to all other active nodes
        for m_idx in range(k):
            if m_idx == i_local or m_idx == j_local:
                continue
            m_global = active_nodes[m_idx]
            d_new_m = 0.5 * (d_mat[i_global, m_global] + d_mat[j_global, m_global] - d_ij)
            d_mat[new_id, m_global] = d_new_m
            d_mat[m_global, new_id] = d_new_m
        
        # Remove i_global and j_global from active, add new_id
        active_nodes.remove(i_global)
        active_nodes.remove(j_global)
        active_nodes.append(new_id)
        
    # Final two nodes
    if len(active_nodes) == 2:
        u, v = active_nodes
        d_uv = d_mat[u, v]
        # Create root
        root = Node()
        node_u = node_objects[u]
        node_v = node_objects[v]
        root.left = node_u
        root.right = node_v
        node_u.parent = root
        node_v.parent = root
        return root
    else:
        # Fallback
        return node_objects[active_nodes[0]]

def _get_internal_edges(node):
    """Get all internal edges (parent, child) where child is internal."""
    edges = []
    if node.is_leaf():
        return edges
    if node.left and not node.left.is_leaf():
        edges.append((node, node.left))
        edges.extend(_get_internal_edges(node.left))
    if node.right and not node.right.is_leaf():
        edges.append((node, node.right))
        edges.extend(_get_internal_edges(node.right))
    return edges

def _fitch_score(node, seq_masks, leaf_indices):
    """Compute Fitch parsimony score for a tree."""
    if node.is_leaf():
        idx = leaf_indices[node.label]
        return seq_masks[:, idx], 0
    
    left_masks, left_score = _fitch_score(node.left, seq_masks, leaf_indices)
    right_masks, right_score = _fitch_score(node.right, seq_masks, leaf_indices)
    
    intersect = left_masks & right_masks
    union = left_masks | right_masks
    steps = np.sum(intersect == 0)
    masks = np.where(intersect == 0, union, intersect)
    
    return masks, left_score + right_score + steps

def _quartet_refine(tree, dist, labels, quartet_scores, max_rounds=5):
    """
    Refine tree topology using quartet scores to guide NNI moves.
    For each NNI move, evaluate if it's supported by quartet evidence.
    """
    n = len(labels)
    if n < 4:
        return tree
    
    leaf_indices = {label: i for i, label in enumerate(labels)}
    
    current_tree = tree
    
    for round_num in range(max_rounds):
        improved = False
        edges = _get_internal_edges(current_tree)
        
        for u, v in edges:
            # Identify the four subtrees around the internal edge (u,v)
            if u.left == v:
                w = u.right
            else:
                w = u.left
            
            x = v.left
            y = v.right
            
            # Get leaf sets for each subtree
            def get_leaves(node):
                leaves = []
                if node.is_leaf():
                    leaves.append(node.label)
                else:
                    leaves.extend(get_leaves(node.left))
                    leaves.extend(get_leaves(node.right))
                return leaves
            
            leaves_w = get_leaves(w)
            leaves_x = get_leaves(x)
            leaves_y = get_leaves(y)
            
            # Use quartet scores to determine best swap
            # We need to check which split is best supported by the global quartet scores
            # However, quartet_scores are for specific quartets. 
            # A simpler heuristic: Use the distance matrix to score the NNI moves directly.
            
            # Calculate average distance for each potential pairing
            # Current: w is connected to u, x and y are connected to v.
            # The "internal edge" connects the group (w) from (x, y).
            # NNI 1: Swap w and x. New groups: (x) and (w, y).
            # NNI 2: Swap w and y. New groups: (y) and (w, x).
            
            # To evaluate, we can sum the quartet scores for all quartets formed by picking
            # one leaf from w, one from x, one from y, and one from the rest (or just use the 3-leaf metric if possible).
            # Given time constraints, we use a simple distance-based score for the NNI move.
            
            def nni_score(move_type):
                # move_type 1: (w,x) vs (y,rest) -> implies w and x are closer
                # move_type 2: (w,y) vs (x,rest) -> implies w and y are closer
                
                # Sample a few quartets
                import random
                random.seed(42 + round_num)
                
                score = 0
                count = 0
                all_leaves = set(labels)
                rest = all_leaves - set(leaves_w + leaves_x + leaves_y)
                if not rest:
                    return 0.0
                
                sample_size = min(20, len(leaves_w) * len(leaves_x) * len(leaves_y) * len(rest))
                
                for _ in range(sample_size):
                    lw = random.choice(leaves_w)
                    lx = random.choice(leaves_x)
                    ly = random.choice(leaves_y)
                    lz = random.choice(list(rest))
                    
                    i_w, i_x, i_y, i_z = leaf_indices[lw], leaf_indices[lx], leaf_indices[ly], leaf_indices[lz]
                    
                    s_wx = dist[i_w, i_x] + dist[i_y, i_z]
                    s_wy = dist[i_w, i_y] + dist[i_x, i_z]
                    
                    if move_type == 1:
                        # Support for (w,x)
                        if s_wx < s_wy:
                            score += 1
                        else:
                            score -= 1
                    else:
                        # Support for (w,y)
                        if s_wy < s_wx:
                            score += 1
                        else:
                            score -= 1
                    count += 1
                    
                return score / max(count, 1)
            
            support1 = nni_score(1)
            support2 = nni_score(2)
            
            if support1 > 0.2 and support1 > support2:
                # Apply NNI move 1: swap w and x
                v.left = w
                v.right = y
                w.parent = v
                y.parent = v
                
                u.left = x
                u.right = v
                x.parent = u
                v.parent = u
                
                improved = True
                break
            elif support2 > 0.2 and support2 > support1:
                # Apply NNI move 2: swap w and y
                v.left = w
                v.right = x
                w.parent = v
                x.parent = v
                
                u.left = y
                u.right = v
                y.parent = u
                v.parent = u
                
                improved = True
                break
        
        if not improved:
            break
    
    return current_tree

def _node_to_newick(node):
    """Convert tree to Newick string."""
    if node.is_leaf():
        return node.label
    left = _node_to_newick(node.left)
    right = _node_to_newick(node.right)
    return f"({left},{right})"

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 240  # Safety margin before 270s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        return f"({labels[0]},{labels[1]})"
    
    # 1. Compute distance matrix
    dist, seq_matrix = _jc_distance_matrix(sequences, labels)
    
    # 2. Compute quartet scores
    quartet_scores = _quartet_scores(dist)
    
    # 3. Build initial tree using NJ
    tree = _nj_build_tree(dist, labels)
    
    # 4. Refine using quartet-guided NNI
    if time.time() < deadline - 10:
        tree = _quartet_refine(tree, dist, labels, quartet_scores, max_rounds=3)
    
    # 5. Final NNI refinement using Fitch parsimony as tiebreaker
    if time.time() < deadline - 5:
        tree = _fitch_nni_refine(tree, seq_matrix, labels)
    
    return _node_to_newick(tree) + ";"

def _fitch_nni_refine(tree, seq_matrix, labels, max_rounds=2):
    """Final NNI refinement using Fitch parsimony score."""
    n = len(labels)
    if n < 4:
        return tree
    
    leaf_indices = {label: i for i, label in enumerate(labels)}
    
    # Compute initial score
    _, current_score = _fitch_score(tree, seq_matrix, leaf_indices)
    
    for round_num in range(max_rounds):
        improved = False
        edges = _get_internal_edges(tree)
        
        for u, v in edges:
            if u.left == v:
                w = u.right
            else:
                w = u.left
            
            x = v.left
            y = v.right
            
            # Move 1: swap w and x
            # Save state
            u_left_orig, u_right_orig = u.left, u.right
            v_left_orig, v_right_orig = v.left, v.right
            
            # Apply move 1
            v.left = w
            v.right = y
            w.parent = v
            y.parent = v
            u.left = x
            u.right = v
            x.parent = u
            v.parent = u
            
            _, score1 = _fitch_score(tree, seq_matrix, leaf_indices)
            gain1 = current_score - score1
            
            # Revert
            u.left, u.right = u_left_orig, u_right_orig
            v.left, v.right = v_left_orig, v_right_orig
            w.parent, x.parent, y.parent = u, v, v
            v.parent = u
            
            # Move 2: swap w and y
            v.left = w
            v.right = x
            w.parent = v
            x.parent = v
            u.left = y
            u.right = v
            y.parent = u
            v.parent = u
            
            _, score2 = _fitch_score(tree, seq_matrix, leaf_indices)
            gain2 = current_score - score2
            
            # Revert
            u.left, u.right = u_left_orig, u_right_orig
            v.left, v.right = v_left_orig, v_right_orig
            w.parent, x.parent, y.parent = u, v, v
            v.parent = u
            
            # Apply best move
            if gain1 > gain2 and gain1 > 0:
                v.left = w
                v.right = y
                w.parent = v
                y.parent = v
                u.left = x
                u.right = v
                x.parent = u
                v.parent = u
                current_score -= gain1
                improved = True
                break
            elif gain2 > 0:
                v.left = w
                v.right = x
                w.parent = v
                x.parent = v
                u.left = y
                u.right = v
                y.parent = u
                v.parent = u
                current_score -= gain2
                improved = True
                break
        
        if not improved:
            break
    
    return tree
