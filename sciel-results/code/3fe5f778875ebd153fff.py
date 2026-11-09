import numpy as np
import time
import math

def reconstruct(sequences):
    start_time = time.time()
    deadline = 250.0  # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return f"({','.join(labels)});"
        
    # 1. Compute Distance Matrix
    # Convert sequences to 0/1/2/3 integers
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in seq] for seq in [sequences[label] for label in labels]])
    L = seqs.shape[1]
    
    # Pairwise Hamming distances
    # seqs is (n, L)
    # diff is (n, n, L)
    # This might be memory heavy for large L, but L<=2000, n<=32. 32*32*2000 = 2M ints. ~16MB. OK.
    diff = seqs[:, np.newaxis, :] - seqs[np.newaxis, :, :]
    hamming = np.sum(diff != 0, axis=2)
    
    # JC Correction
    p = hamming / L
    # Handle p >= 0.75
    # np.where(p < 0.75, -0.75 * np.log(1 - 4/3 * p), 10.0)
    # Note: log(1 - 4/3 * 0.75) = log(0) = -inf.
    dist_matrix = np.where(p < 0.75, -0.75 * np.log(1 - (4.0/3.0) * p), 10.0)
    np.fill_diagonal(dist_matrix, 0.0)
    
    # 2. Neighbor Joining
    # Nodes: 0..n-1
    # We will maintain a list of active nodes
    active = list(range(n))
    # trees: dict mapping node id to Newick string? 
    # For NNI, we need a graph representation.
    # Let's build the tree as a list of edges (u, v, length)
    # And a mapping from node id to its "name" in the Newick output.
    
    # Standard NJ implementation
    # We need to track the topology.
    # Let's use a dictionary `node_map` where key is current node id, value is (newick_string, parent_id, branch_length_to_parent)
    # This is tricky. 
    # Better: Build the tree step-by-step, storing the structure.
    
    # Let's stick to the seed's approach for the NJ part, but store branch lengths.
    # The seed does not store branch lengths.
    # I will modify the NJ loop to store the tree structure in a way that allows NNI.
    
    # Tree representation for NNI:
    # Nodes: 0..2n-3
    # Edges: list of (u, v, length)
    # Parent: dict node -> parent (for rooted view, but tree is unrooted)
    # Let's just keep a list of edges.
    
    edges = []
    # active_nodes: list of current leaf/internal nodes
    active_nodes = list(range(n))
    # node_newick: dict node_id -> newick_string (for leaves, just label)
    node_newick = {i: labels[i] for i in range(n)}
    
    # dist_matrix is (n, n) initially. We need to expand it as we add nodes.
    # This is inefficient. 
    # Alternative: Keep the full distance matrix and update it.
    
    # Let's use a class or careful bookkeeping.
    # For n=32, we can just re-compute or update carefully.
    
    # I will implement a simple NJ that builds the tree and stores edges.
    
    # Initialize
    current_dist = dist_matrix.copy()
    node_count = n
    next_node_id = n
    
    while len(active_nodes) > 3:
        if time.time() > deadline:
            break
            
        k = len(active_nodes)
        # Compute r_i = sum_{j != i} d_{ij}
        r = np.zeros(k)
        for i in range(k):
            for j in range(k):
                if i != j:
                    r[i] += current_dist[active_nodes[i], active_nodes[j]]
        
        # Find best pair
        best_val = float('inf')
        best_i, best_j = 0, 1
        for i in range(k):
            for j in range(i+1, k):
                # Q = (k-2)*d_ij - r_i - r_j
                q = (k - 2) * current_dist[active_nodes[i], active_nodes[j]] - r[i] - r[j]
                if q < best_val:
                    best_val = q
                    best_i, best_j = i, j
                    
        u = active_nodes[best_i]
        v = active_nodes[best_j]
        
        # Calculate branch lengths
        # l_u = 0.5 * d_uv + (r_u - r_v) / (2*(k-2))
        # l_v = 0.5 * d_uv - (r_u - r_v) / (2*(k-2))
        d_uv = current_dist[u, v]
        l_u = 0.5 * d_uv + (r[best_i] - r[best_j]) / (2.0 * (k - 2))
        l_v = 0.5 * d_uv - (r[best_i] - r[best_j]) / (2.0 * (k - 2))
        
        # Ensure non-negative (optional, but good practice)
        if l_u < 0: l_u = 0
        if l_v < 0: l_v = 0
        
        # Create new node
        w = next_node_id
        next_node_id += 1
        
        # Add edges
        edges.append((u, w, l_u))
        edges.append((v, w, l_v))
        
        # Update Newick for w
        # We need to know the Newick strings for u and v.
        # If u or v is an internal node, its Newick string is already in node_newick.
        # If it's a leaf, it's the label.
        newick_u = node_newick[u]
        newick_v = node_newick[v]
        node_newick[w] = f"({newick_u},{newick_v})"
        
        # Update distance matrix
        # d_wx = 0.5 * (d_ux + d_vx - d_uv)
        # We need to add a row/col for w in current_dist?
        # current_dist is (n, n). We can't easily expand it.
        # Let's use a dictionary for distances or a larger matrix.
        # Since n is small, let's just use a dictionary `dist_dict` for active nodes.
        
        # Actually, let's restart the NJ loop with a dictionary-based distance matrix to make it cleaner.
        
        # Break and re-implement with dict?
        # No, let's just use the `current_dist` array but be careful.
        # The active nodes are a subset of 0..next_node_id-1.
        # It's easier to maintain a full matrix of size (2n, 2n).
        
        # Let's refactor to use a full matrix.
        
        # This inner loop is getting messy. Let's step back and write a cleaner NJ.
        
        # I will abort this specific implementation path in the thought process and write the final code carefully.
        
        pass

    # For the final code, I will implement:
    # 1. NumPy Distance
    # 2. NJ with full matrix expansion (pre-allocate 2n x 2n)
    # 3. NNI Polish
    
    return "placeholder"
