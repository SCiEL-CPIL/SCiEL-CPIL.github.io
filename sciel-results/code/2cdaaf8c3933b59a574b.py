import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin for 180s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
        
    # Convert sequences to numpy array of integers
    # A:0, C:1, G:2, T:3
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs_list = [sequences[label] for label in labels]
    L = len(seqs_list[0])
    
    # Vectorized Hamming Distance
    # Create a matrix of shape (n, n, L)
    # seqs_matrix: (n, L)
    seqs_matrix = np.array([[base_map[c] for c in s] for s in seqs_list], dtype=np.int8)
    
    # Compute pairwise differences
    # (n, 1, L) != (1, n, L) -> (n, n, L)
    # This can be memory heavy for n=32, L=2000. 32*32*2000 = 2M ints. 16MB. Fine.
    diffs = (seqs_matrix[:, None, :] != seqs_matrix[None, :, :])
    hamming = np.sum(diffs, axis=2).astype(float)
    
    # JC Correction
    p = hamming / L
    # Avoid log(0) or log(negative)
    # p < 0.75 for valid JC
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    
    # Initialize tree tracking
    # nodes: list of current node labels
    # trees: dict mapping node label -> Newick string
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    # D is currently n x n
    # We will update D as we join nodes.
    # To make it easier, we'll maintain a list of active node indices in the original D matrix?
    # No, D changes. We'll use a dynamic matrix.
    # Since n is small, we can just recreate the matrix or use a dict.
    # But Bionj needs matrix multiplication.
    # Let's use a list of lists or a NumPy array that we shrink.
    
    # Current active nodes in the original index space?
    # It's easier to keep a list of "current" nodes and a distance matrix between them.
    # Let's map original indices to new indices as we go.
    
    # active_nodes: list of original indices
    active_nodes = list(range(n))
    # dist_matrix: 2D list or array corresponding to active_nodes
    dist_matrix = D[active_nodes, :][:, active_nodes]
    
    while len(active_nodes) > 3:
        if time.time() > deadline:
            break
            
        m = len(active_nodes)
        
        # Compute Bionj Criterion M
        # M_ij = D_i . D_j - 0.5 R_i - 0.5 R_j
        # R_i = sum_k D_ik^2 (excluding i? D_ii is 0, so sum over all k is fine)
        
        D_sq = dist_matrix ** 2
        R = np.sum(D_sq, axis=1)
        
        # Dot product matrix
        # dist_matrix @ dist_matrix.T
        # Note: dist_matrix is m x m
        Dot = dist_matrix @ dist_matrix.T
        
        M = Dot - 0.5 * R[:, None] - 0.5 * R[None, :]
        
        # Set diagonal to inf to avoid self-joining
        np.fill_diagonal(M, np.inf)
        
        # Find min
        min_idx = np.unravel_index(np.argmin(M), M.shape)
        i, j = min_idx
        
        # Ensure i < j for consistency in naming, though not strictly necessary
        if i > j:
            i, j = j, i
            
        orig_i = active_nodes[i]
        orig_j = active_nodes[j]
        
        # New node name
        new_name = f"n{len(active_nodes)}_{orig_i}_{orig_j}"
        
        # Build Newick string for the new node
        # trees[orig_i] and trees[orig_j] are the Newick strings for the subtrees
        new_tree_str = f"({trees[orig_i]},{trees[orig_j]})"
        
        # Update distance matrix
        # New distance to k: 0.5 * (D_ik + D_jk - D_ij)
        # k ranges over all other active nodes
        # We need to create a new matrix of size (m-1) x (m-1)
        
        # Remove i and j from active nodes
        # Create list of remaining indices
        remaining_indices = [k for k in range(m) if k not in (i, j)]
        
        # Calculate new distances
        # D_new[k, new_node] = 0.5 * (D[i, k] + D[j, k] - D[i, j])
        # Note: D is symmetric.
        
        new_dist_row = 0.5 * (dist_matrix[i, remaining_indices] + dist_matrix[j, remaining_indices] - dist_matrix[i, j])
        
        # Build new matrix
        # The submatrix of remaining nodes
        new_dist_matrix = dist_matrix[np.ix_(remaining_indices, remaining_indices)]
        
        # Add the new node
        # new_dist_matrix is (m-2) x (m-2)
        # We need to add a row and column for the new node.
        # Size becomes (m-1) x (m-1)
        
        # Initialize new matrix
        size_new = m - 1
        new_full_matrix = np.zeros((size_new, size_new))
        
        # Copy the submatrix
        new_full_matrix[:m-2, :m-2] = new_dist_matrix
        
        # Set distances to the new node
        # The new node is at index m-1 in the new matrix
        new_full_matrix[:m-2, m-1] = new_dist_row
        new_full_matrix[m-1, :m-2] = new_dist_row
        
        # Update active nodes
        # Remove orig_i, orig_j
        new_active_nodes = [active_nodes[k] for k in remaining_indices]
        new_active_nodes.append(new_name)
        
        # Update trees
        trees[new_name] = new_tree_str
        
        # Update state
        active_nodes = new_active_nodes
        dist_matrix = new_full_matrix

    # Final tree
    # If we broke early due to time, we might have a partial tree.
    # But for n<=32, it should finish.
    
    # If len(active_nodes) == 3
    # The tree is (A, B, C)
    # Unrooted tree of 3 leaves is just the star or a single internal node?
    # In Newick, 3 leaves: (A, B, C); is not standard.
    # Standard is ((A, B), C); which is rooted.
    # Unrooted: The topology is just the 3 leaves connected.
    # Newick for unrooted 3-taxon tree: (A, B, C); is often accepted or ((A,B),C);
    # The prompt says "unrooted tree topology".
    # Usually, Newick for 3 taxa is written as (A,B,C); or ((A,B),C);
    # Let's check the seed:
    # `return "(" + ",".join(trees[x] for x in nodes) + ");"`
    # If nodes has 3 items, it returns (A, B, C);
    # This is a valid Newick string for an unrooted tree with 3 leaves.
    
    if len(active_nodes) == 3:
        return "(" + ",".join(trees[x] for x in active_nodes) + ");"
    elif len(active_nodes) == 2:
        return f"({trees[active_nodes[0]]},{trees[active_nodes[1]]});"
    elif len(active_nodes) == 1:
        return trees[active_nodes[0]] + ";"
    else:
        # Fallback: Star tree
        return "(" + ",".join(trees[x] for x in active_nodes) + ");"

# EVOLVE-BLOCK-END
