"""Phylogenetic topology recovery using Fast Distance-based Quartet Inference."""
import math
import time
import itertools
import numpy as np

def _jc_distance_matrix(sequences):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    # Convert to numpy array of indices 0-3
    # A:0, C:1, G:2, T:3
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[base_to_idx[base] for base in seq] for seq in (sequences[l] for l in labels)], dtype=np.uint8)
    
    # Compute Hamming distances
    # (n, L) vs (n, L) -> (n, n, L)
    # diff = seqs[:, None, :] != seqs[None, :, :]
    # This creates a large array. For n=32, L=2000, 32*32*2000 = 2M bools. OK.
    diff = seqs[:, None, :] != seqs[None, :, :]
    hamming = diff.sum(axis=2)
    
    # Jukes-Cantor correction
    p = hamming.astype(float) / seqs.shape[1]
    # Handle p >= 0.75
    p_clipped = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    return dist, labels

def _quartet_scores(dist, n):
    """
    Compute scores for all unique quartets.
    Returns a list of (score, split_mask)
    """
    scores = []
    # Iterate over all unique quartets
    for a, b, c, d in itertools.combinations(range(n), 4):
        dab = dist[a, b]
        dac = dist[a, c]
        dad = dist[a, d]
        dbc = dist[b, c]
        dbd = dist[b, d]
        dcd = dist[c, d]
        
        s1 = dab + dcd # (ab)(cd)
        s2 = dac + dbd # (ac)(bd)
        s3 = dad + dbc # (ad)(bc)
        
        # Find min and second min
        # Sort the three values
        vals = [s1, s2, s3]
        vals.sort()
        min_val = vals[0]
        second_min = vals[1]
        score = second_min - min_val
        
        if score <= 0:
            continue
            
        if min_val == s1:
            # Split {a,b} | {c,d}
            mask = (1 << a) | (1 << b)
        elif min_val == s2:
            # Split {a,c} | {b,d}
            mask = (1 << a) | (1 << c)
        else:
            # Split {a,d} | {b,c}
            mask = (1 << a) | (1 << d)
            
        scores.append((score, mask))
        
    return scores

def _build_tree_from_splits(labels, splits, n):
    """
    Build Newick string from a list of compatible splits (bitmasks).
    splits: list of (score, mask) or just masks? We assume they are compatible and sufficient.
    """
    # We need to map indices to labels
    # Recursively build
    def build(taxa_mask, candidate_splits):
        if taxa_mask == 0:
            return ""
        
        # Count bits
        bit_count = bin(taxa_mask).count("1")
        if bit_count == 1:
            # Find the index
            idx = (taxa_mask & -taxa_mask).bit_length() - 1
            return labels[idx]
        
        if bit_count == 2:
            # Find indices
            idx1 = (taxa_mask & -taxa_mask).bit_length() - 1
            mask_rem = taxa_mask ^ (1 << idx1)
            idx2 = (mask_rem & -mask_rem).bit_length() - 1
            return f"({labels[idx1]},{labels[idx2]})"
        
        # Find a split that partitions taxa_mask
        # A split S partitions T if (S & T) != 0 and ((~S) & T) != 0
        # And S is compatible with previous choices (handled by filtering)
        
        chosen_split = None
        # We want to pick a split that is "balanced" or just any valid one?
        # Greedy: pick the one with highest score?
        # But we already filtered compatible ones.
        # Let's pick the first one that partitions.
        
        for mask in candidate_splits:
            a = mask & taxa_mask
            b = (~mask) & taxa_mask
            if a != 0 and b != 0:
                # This split partitions the current set
                # Check if it's a "trivial" split of the current set?
                # If a == taxa_mask or b == taxa_mask, it doesn't split.
                # We ensured a!=0 and b!=0, so it splits.
                
                # Filter splits for children
                # Child A gets splits that are subsets of A
                # Child B gets splits that are subsets of B
                splits_a = [s for s in candidate_splits if (s & ~a) == 0]
                splits_b = [s for s in candidate_splits if (s & ~b) == 0]
                
                left = build(a, splits_a)
                right = build(b, splits_b)
                return f"({left},{right})"
        
        # If no split found, fallback to star or error
        # This should not happen if splits are sufficient
        indices = []
        m = taxa_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(labels[idx])
            m ^= lsb
        return "(" + ",".join(indices) + ")"
    
    full_mask = (1 << n) - 1
    # Extract masks from splits
    masks = [s[1] if isinstance(s, tuple) else s for s in splits]
    return build(full_mask, masks)

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260.0 # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    dist, labels = _jc_distance_matrix(sequences)
    
    # If n is small, NJ is fast and reliable.
    # For larger n, use Quartet Inference.
    # Let's try Quartet Inference for all n >= 4.
    
    if n < 4:
        # Handle small cases with NJ or simple logic
        # For n=3, any tree is the same topology
        return f"({labels[0]},({labels[1]},{labels[2]}))"
    
    # Generate quartet scores
    # This is O(n^4). For n=32, ~450k quartets.
    # Python loop might be slow.
    # Let's check time.
    
    # If n is large, we might need to be careful.
    # But 450k iterations is fast in Python (<1s).
    
    scores = _quartet_scores(dist, n)
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Greedy split selection
    selected_splits = []
    full_mask = (1 << n) - 1
    
    # We need n-3 splits
    target_splits = n - 3
    
    # Optimization: Only consider top K splits?
    # No, greedy needs to check compatibility.
    # But we can stop early if we find enough.
    
    # To speed up compatibility check, we can use a list of selected masks.
    # For each candidate, check against all selected.
    
    for score, mask in scores:
        if len(selected_splits) == target_splits:
            break
            
        if time.time() > deadline - 10.0:
            break
            
        # Check compatibility
        compatible = True
        for s_mask in selected_splits:
            # Check if mask and s_mask are compatible
            # A = mask, B = ~mask
            # C = s_mask, D = ~s_mask
            # Compatible if A&C==0 or A&D==0 or B&C==0 or B&D==0
            a = mask
            b = full_mask ^ mask
            c = s_mask
            d = full_mask ^ s_mask
            
            if (a & c) != 0 and (a & d) != 0 and (b & c) != 0 and (b & d) != 0:
                compatible = False
                break
                
        if compatible:
            selected_splits.append(mask)
            
    # If we didn't get enough splits, fallback to NJ
    if len(selected_splits) < target_splits:
        # Fallback to Neighbor Joining
        return _neighbor_joining(dist, labels)
        
    # Build tree
    tree_str = _build_tree_from_splits(labels, selected_splits, n)
    
    return tree_str

def _neighbor_joining(dist, labels):
    """Standard Neighbor Joining as fallback."""
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}))"
        
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = dist.copy()
    
    while len(nodes) > 3:
        k = len(nodes)
        r = np.zeros(k)
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j:
                    r[i] += d[a, b]
                    
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        
        for i in range(k):
            for j in range(i + 1, k):
                a = nodes[i]
                b = nodes[j]
                criterion = (k - 2) * d[a, b] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
                    
        a = nodes[best_i]
        b = nodes[best_j]
        
        # Create new node
        new_node = -1 - len(nodes) # Negative ID
        joined_tree = f"({trees[a]},{trees[b]})"
        
        # Update distance matrix
        # d[new, x] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
        # We need to add a row/col to d? Or just store in a dict?
        # Since we are removing a,b and adding new, we can create a new matrix.
        
        new_nodes = [x for idx, x in enumerate(nodes) if idx != best_i and idx != best_j]
        new_nodes.append(new_node)
        
        new_d = np.zeros((len(new_nodes), len(new_nodes)))
        for i, u in enumerate(new_nodes):
            for j, v in enumerate(new_nodes):
                if i == j:
                    continue
                if u == new_node:
                    # u is new, v is old
                    if v == a or v == b:
                        # Should not happen as a,b removed
                        pass
                    new_d[i, j] = 0.5 * (d[a, v] + d[b, v] - d[a, b])
                elif v == new_node:
                    new_d[i, j] = 0.5 * (d[u, a] + d[u, b] - d[a, b])
                else:
                    new_d[i, j] = d[u, v]
                    
        d = new_d
        nodes = new_nodes
        trees[new_node] = joined_tree
        
    # Combine remaining 3
    # Order doesn't matter for unrooted
    return f"({trees[nodes[0]]},{trees[nodes[1]]},{trees[nodes[2]]})"
