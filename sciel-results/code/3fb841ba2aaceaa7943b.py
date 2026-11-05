"""
Phylogenetic topology recovery using Neighbor-Joining with NumPy optimization.
"""
import time
import numpy as np
import math

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 270.0  # Seconds

def _check_time():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def _jc_distance_matrix(sequences, labels):
    """
    Compute the Jukes-Cantor corrected distance matrix.
    sequences: dict mapping label to DNA string
    labels: list of labels
    """
    n = len(labels)
    # Convert sequences to numpy arrays of integers for fast comparison
    # Map A,C,G,T to 0,1,2,3
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([list(map(lambda x: seq_map[x], sequences[label])) for label in labels], dtype=np.uint8)
    
    # Compute pairwise Hamming distances
    # seqs is shape (n, L)
    # We want distance matrix of shape (n, n)
    # d[i,j] = sum(seqs[i] != seqs[j])
    
    # Efficient way: 
    # For each pair, count mismatches.
    # This can be done by broadcasting, but n*L might be large if L is 2000 and n is 32.
    # 32*32*2000 = 2M operations, which is trivial for NumPy.
    
    # Create a 3D array of differences? No, just compare row by row or use broadcasting carefully.
    # Let's do it in chunks or just simple broadcasting if memory allows.
    # n=32, L=2000. 32x32x2000 bool array is ~2MB. Fine.
    
    # Compare all pairs
    # seqs[:, None, :] vs seqs[None, :, :]
    # This creates a (n, n, L) array.
    diffs = (seqs[:, None, :] != seqs[None, :, :])
    hamming = np.sum(diffs, axis=2)
    
    # Convert to proportions
    L = seqs.shape[1]
    p = hamming / float(L)
    
    # Jukes-Cantor correction: d = -3/4 * ln(1 - 4p/3)
    # Handle p >= 0.75 case
    # np.where handles this nicely
    # If 1 - 4p/3 <= 0, log is undefined. We cap it.
    # The term inside log is 1 - 4/3 * p.
    # If p > 0.75, this is negative.
    # We set the distance to a large value or a specific cap.
    # Standard practice: if p >= 0.75, set to inf or a large constant.
    # Let's use a large constant like 10.0 or np.inf. NJ handles inf poorly if all are inf, 
    # but with random data, p is usually small.
    
    # To be safe against log(0) or log(negative):
    # valid = p < 0.75
    # d = np.zeros_like(p)
    # d[valid] = -0.75 * np.log(1 - 4/3 * p[valid])
    # d[~valid] = 10.0  # Arbitrary large distance
    
    # Vectorized version:
    arg = 1.0 - (4.0 / 3.0) * p
    # Clip to small positive number to avoid log(0) or negative
    # If arg <= 0, the distance is effectively infinite.
    # We can replace non-positive args with a tiny epsilon, then set those entries to a large value afterwards.
    
    # Better:
    d = np.zeros_like(p)
    valid = arg > 0
    d[valid] = -0.75 * np.log(arg[valid])
    d[~valid] = 10.0  # Cap for saturated distances
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d

def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology using Neighbor-Joining.
    Returns a Newick string.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]})();"
    if n == 3:
        # Unrooted tree with 3 taxa is a star (or just the 3 leaves)
        # In Newick, a 3-leaf unrooted tree is often represented as (a,b,c); 
        # or strictly as ((a,b),c); but since it's unrooted, any binary representation 
        # implies a root. The task asks for unrooted topology.
        # Standard Newick for 3 leaves is (a,b,c); which is a polytomy.
        # Or we can output ((a,b),c); which is a binary tree. 
        # Robinson-Foulds distance compares splits.
        # A 3-leaf tree has no non-trivial splits.
        # So any valid tree with 3 leaves should have RF distance 0.
        return f"({labels[0]},{labels[1]},{labels[2]})();"
    
    # Initialize distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Initialize list of active nodes
    active = list(range(n))
    
    # Keep track of the subtree structure for each node index
    # tree_structure[i] will be the Newick string for the subtree rooted at node i
    tree_structure = [labels[i] for i in range(n)]
    
    # Counter for new internal nodes
    node_counter = n
    
    # Neighbor-Joining Loop
    while len(active) > 3:
        if _check_time():
            # Fallback: return a star tree or whatever we have
            # If we run out of time, we should return the best we can.
            # For now, just break and return a star tree of remaining + already joined
            # Actually, if we break, we need to construct a valid tree from 'active' nodes.
            # A star tree is a safe fallback.
            break
            
        m = len(active)
        # D is currently a full n x n matrix, but we only care about active nodes.
        # Let's extract the submatrix for active nodes to keep it simple and fast.
        # m is small (<=32), so extracting is O(m^2).
        D_active = D[np.ix_(active, active)]
        
        # Calculate row sums r_i
        # r_i = sum_{j in active, j != i} D[i, j]
        # Since diagonal is 0, sum is just sum of the row.
        r = np.sum(D_active, axis=1)
        
        # Calculate Q matrix: Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        # We want to find the pair (i, j) with minimum Q[i, j]
        # Note: D_active is symmetric.
        
        Q = (m - 2) * D_active - r[:, None] - r[None, :]
        
        # Set diagonal to inf so we don't pick (i, i)
        np.fill_diagonal(Q, np.inf)
        
        # Find the index of the minimum value in Q
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        
        # Map back to original node indices
        i = active[i_idx]
        j = active[j_idx]
        
        # Create a new node k
        k = node_counter
        node_counter += 1
        
        # Calculate branch lengths (optional, but good for completeness)
        # d(i, k) = 0.5 * D[i, j] + (r[i] - r[j]) / (2 * (m - 2))
        # d(j, k) = 0.5 * D[i, j] - (r[i] - r[j]) / (2 * (m - 2))
        # We don't strictly need branch lengths for topology, but they are part of the Newick string if we want to be precise.
        # The task says "branch lengths optional". We will omit them to keep the string short and avoid floating point issues in parsing if any.
        # Actually, standard Newick often includes them. Let's include them for robustness if the parser expects them, 
        # but the task says "branch lengths optional". I will omit them to be safe and simple.
        
        # Update tree structure
        tree_structure.append(f"({tree_structure[i]},{tree_structure[j]})")
        
        # Update distance matrix D
        # For all x in active \ {i, j}:
        # D[k, x] = 0.5 * (D[i, x] + D[j, x] - D[i, j])
        
        # We need to update the full D matrix or just keep D_active?
        # It's easier to maintain the full D matrix and update it.
        
        # Get indices of other active nodes
        other_active = [x for x in active if x != i and x != j]
        
        # Calculate new distances
        # D[i, x] is D[i_idx, x_idx_in_active]
        # Let's just use the original D matrix indices
        D_i_x = D[i, other_active]
        D_j_x = D[j, other_active]
        D_i_j = D[i, j]
        
        new_dists = 0.5 * (D_i_x + D_j_x - D_i_j)
        
        # Update D[k, x] and D[x, k]
        D[k, other_active] = new_dists
        D[other_active, k] = new_dists
        
        # Set distances from k to i and j to 0 (or remove them)
        # Since i and j are removed from active, their rows/cols in active submatrix will be ignored.
        # But we need to make sure D is consistent if we extract submatrix later.
        # We will remove i and j from 'active'.
        
        # Update active list
        active.remove(i)
        active.remove(j)
        active.append(k)
        
        # Ensure tree_structure has an entry for k
        # We already appended it above.
        
    # If we broke due to time, or finished loop
    # If len(active) > 3, we need to join them into a star tree
    # If len(active) == 3, we join them into a 3-leaf tree
    
    if len(active) > 3:
        # Star tree of remaining active nodes
        # This is a fallback.
        # Combine all tree_structure[active] into a star
        leaves = [tree_structure[x] for x in active]
        return "(" + ",".join(leaves) + ");"
    elif len(active) == 3:
        # Join the 3 remaining nodes into a single unrooted tree
        # Standard Newick for 3 leaves: (a,b,c);
        # Or ((a,b),c);
        # Let's use (a,b,c); as it's a valid unrooted representation (polytomy)
        # However, if the true tree is binary, (a,b,c) implies a polytomy which might have RF distance > 0 if the true tree is binary?
        # No, for 3 leaves, there are no non-trivial splits. Any tree with 3 leaves has the same set of non-trivial splits (which is empty).
        # So (a,b,c); is fine.
        leaves = [tree_structure[x] for x in active]
        return "(" + ",".join(leaves) + ");"
    elif len(active) == 2:
        # Should not happen if we start with n>=3 and stop at 3
        # But just in case
        leaves = [tree_structure[x] for x in active]
        return "(" + ",".join(leaves) + ");"
    else:
        # Should not happen
        return tree_structure[active[0]] + ";" if active else ""

# Initialize start time on module load to account for import time?
# Better to initialize inside reconstruct or just use a global that is set on first call.
# The global _START_TIME is set at import. This is usually fine.
# However, if the module is imported long before the call, the time might be stale.
# Let's reset it on the first call to reconstruct.

_first_call = [False]

def _init_time():
    if not _first_call[0]:
        global _START_TIME
        _START_TIME = time.time()
        _first_call[0] = True

# Wrap the original reconstruct logic to ensure time is initialized
_orig_reconstruct = reconstruct

def reconstruct(sequences):
    _init_time()
    return _orig_reconstruct(sequences)
