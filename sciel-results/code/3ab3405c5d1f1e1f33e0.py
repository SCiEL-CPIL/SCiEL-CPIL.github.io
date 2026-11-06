import math
import time
import numpy as np
from itertools import combinations

def _compute_jc_distance_matrix(sequences):
    """
    Compute the Jukes-Cantor distance matrix for the given sequences.
    sequences: dict mapping label to DNA string.
    Returns: NumPy array of shape (n, n) and list of labels.
    """
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n = len(labels)
    seq_list = [sequences[l] for l in labels]
    
    # Convert to integer array
    # Map A->0, C->1, G->2, T->3
    # Using numpy vectorized operations for speed
    # Create a 2D array of shape (n, L)
    # We can use np.array with a list of lists, or better, encode directly.
    
    # Efficient encoding:
    # 'A' -> 0, 'C' -> 1, 'G' -> 2, 'T' -> 3
    # We can use a lookup table or np.where chains.
    # For small n and L, even a simple loop is fast, but let's vectorize.
    
    # Convert strings to bytes or use char codes
    # A=65, C=67, G=71, T=84
    # Map: 65->0, 67->1, 71->2, 84->3
    # Others -> 0 (should not happen)
    
    # Create an array of shape (n, L)
    # We can use np.vectorize or manual mapping.
    # Let's use a simple mapping via np.select or similar.
    
    # Actually, the fastest way for small L is often just converting to a list of ints and then to numpy.
    # But let's try to be efficient.
    
    # Method:
    # 1. Join all sequences? No, they are separate.
    # 2. Use np.array([list(s) for s in seq_list]) -> shape (n, L) of characters.
    # 3. Map characters to ints.
    
    # Let's use the character array approach as it's clean.
    char_array = np.array([list(s) for s in seq_list], dtype='U1')
    
    # Map to ints
    # A=0, C=1, G=2, T=3
    # We can use np.isin or a lookup.
    # Since it's U1, we can compare directly.
    
    # Create a lookup table for all possible characters?
    # Or just:
    # A_mask = char_array == 'A'
    # C_mask = char_array == 'C'
    # G_mask = char_array == 'G'
    # T_mask = char_array == 'T'
    
    # int_array = 0*A_mask + 1*C_mask + 2*G_mask + 3*T_mask
    # This is 4 passes.
    
    # Alternatively, use np.vectorize with a mapping dict.
    # Or, since the alphabet is small, we can do:
    
    # Let's try the mask approach, it's vectorized.
    A_mask = (char_array == 'A')
    C_mask = (char_array == 'C')
    G_mask = (char_array == 'G')
    # T is the rest, but let's be explicit or assume valid input.
    # If input is valid ACGT, T = ~A & ~C & ~G
    
    int_array = np.zeros((n, len(seq_list[0])), dtype=np.int8)
    int_array[C_mask] = 1
    int_array[G_mask] = 2
    int_array[~A_mask & ~C_mask & ~G_mask] = 3 # T
    
    # Compute pairwise Hamming distances
    # A[:, None, :] != A[None, :, :]
    # Shape: (n, n, L)
    # This can be memory intensive if L is large, but L <= 2000, n <= 32.
    # 32 * 32 * 2000 = 2,048,000 elements. 8 bytes each -> 16 MB. Fine.
    
    # diff = (int_array[:, None, :] != int_array[None, :, :])
    # hamming = diff.sum(axis=2)
    
    # To save memory/time, we can do it in chunks or just let numpy handle it.
    # Let's use the direct comparison.
    
    # Note: For very large L, this might be slow. But L <= 2000.
    
    hamming = (int_array[:, None, :] != int_array[None, :, :]).sum(axis=2)
    
    L = len(seq_list[0])
    p = hamming.astype(np.float64) / L
    
    # Jukes-Cantor correction
    # d = -0.75 * ln(1 - 4/3 * p)
    # Handle p = 0.75 (log 0)
    p_clipped = np.clip(p, 0, 0.749999)
    
    D = -0.75 * np.log(1.0 - (4.0/3.0) * p_clipped)
    
    # If p was clipped, the distance is effectively infinite/undefined, 
    # but for topology, a large value is fine.
    # The formula gives -0.75 * ln(1 - 1) = inf.
    # np.log(0) is -inf. So D becomes inf.
    # We should handle inf carefully in NJ.
    # If p >= 0.75, set to a large value or handle separately.
    
    # Let's check for inf
    # If any p >= 0.75, set D to a large number like 10.0?
    # Or just let it be inf and hope NJ handles it?
    # NJ criterion: (m-2)D_ij - r_i - r_j.
    # If D_ij is inf, criterion is inf (unless r_i or r_j is inf with opposite sign, which won't happen).
    # So inf distances will never be chosen as the minimum. This is correct behavior.
    
    np.fill_diagonal(D, 0.0)
    
    return D, labels

def _parse_newick_to_splits(newick, n_leaves):
    """
    Parse a Newick string and return the set of non-trivial splits.
    n_leaves: number of leaves to expect.
    """
    # Remove trailing semicolon
    if newick.endswith(';'):
        newick = newick[:-1]
    
    splits = set()
    
    def get_leaves(node_str):
        """Extract leaf labels from a subtree string."""
        # This is a bit inefficient but n is small.
        # Better to parse properly.
        pass

    # Let's use a proper parser that returns the tree structure or splits directly.
    # We'll use a stack-based parser.
    
    # Each node in the stack is a list of children (leaves are strings, internal are lists).
    stack = []
    current = []
    i = 0
    s = newick
    
    # This is a standard Newick parser.
    # We want to collect the set of leaves for each internal node.
    
    # Let's build a tree of sets.
    # Root is a set of leaves.
    
    def parse_node():
        nonlocal i
        if i >= len(s):
            return set()
        
        if s[i] == '(':
            i += 1
            children = []
            while True:
                child = parse_node()
                children.append(child)
                if i < len(s):
                    if s[i] == ',':
                        i += 1
                    elif s[i] == ')':
                        i += 1
                        break
                    else:
                        break
            # Combine children into a set
            result = set()
            for child in children:
                result.update(child)
            return result
        else:
            # Leaf
            start = i
            while i < len(s) and s[i] not in ',);':
                i += 1
            label = s[start:i]
            # Skip branch length
            if i < len(s) and s[i] == ':':
                i += 1
                while i < len(s) and s[i] not in ',);':
                    i += 1
            return {label}
    
    all_leaves = parse_node()
    
    # Now, to get splits, we need to know the structure.
    # The set of leaves of any internal node defines a split.
    # However, the parser above just returns the set of leaves for the root.
    # We need to capture the intermediate sets.
    
    # Let's modify the parser to record all internal node leaf sets.
    
    splits_set = set()
    
    def parse_node_v2():
        nonlocal i
        if i >= len(s):
            return set()
        
        if s[i] == '(':
            i += 1
            children = []
            while True:
                child = parse_node_v2()
                children.append(child)
                if i < len(s):
                    if s[i] == ',':
                        i += 1
                    elif s[i] == ')':
                        i += 1
                        break
                    else:
                        break
            
            # Combine children
            result = set()
            for child in children:
                result.update(child)
            
            # Record this split
            # Only record if it's a non-trivial split (size > 1 and < n_leaves)
            if 1 < len(result) < n_leaves:
                # Normalize: use the smaller side
                if len(result) <= n_leaves // 2:
                    splits_set.add(frozenset(result))
                else:
                    splits_set.add(frozenset(all_leaves - result))
            
            return result
        else:
            start = i
            while i < len(s) and s[i] not in ',);':
                i += 1
            label = s[start:i]
            if i < len(s) and s[i] == ':':
                i += 1
                while i < len(s) and s[i] not in ',);':
                    i += 1
            return {label}
    
    i = 0
    all_leaves = parse_node_v2()
    
    return splits_set

def _splits_to_newick(splits, labels):
    """
    Reconstruct a Newick string from a set of splits.
    splits: set of frozensets.
    labels: list of leaf labels.
    """
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # We need to build the tree.
    # This is the hard part.
    # Given a set of compatible splits, we can build the tree recursively.
    
    # A split S partitions the leaves into S and L\S.
    # If we root the tree at an arbitrary leaf (or edge), we can find the root split.
    # Actually, for an unrooted tree, we can just find any split and recurse.
    
    # Let's use a recursive approach:
    # 1. Find a split that is "minimal" in some sense?
    # 2. Or, find the split that corresponds to the "root" if we root at a leaf.
    
    # Simpler:
    # We can build the tree by finding a leaf and its adjacent split.
    # Or, we can use the fact that the splits form a laminar family (if rooted).
    
    # Let's root the tree at an arbitrary leaf, say labels[0].
    # Then every internal node corresponds to a split S such that labels[0] in S.
    # The root of the rooted tree is the split containing labels[0] that is the "largest" (closest to root).
    # Actually, in a rooted tree, the root is the whole set L.
    # The children of the root are the maximal proper subsets of L that contain labels[0] and are in the split set?
    # No, the splits are unrooted.
    
    # Let's try a different approach:
    # We have a set of splits.
    # We want to output a Newick string.
    # We can pick any leaf, say 't0'.
    # The neighbors of 't0' in the tree are defined by the splits that contain 't0' and have size 2? No.
    
    # Let's use a standard algorithm to build a tree from splits.
    # 1. Create a node for each split.
    # 2. Connect nodes if one split is a subset of the other?
    
    # This is getting complex. Let's use a simpler heuristic for small n.
    # Since we are using this for n <= 7 (exhaustive search), we can afford a slower method.
    
    # For n <= 7, we can just try all possible Newick strings? No.
    
    # Let's use a recursive construction:
    # 1. If n == 1, return leaf.
    # 2. If n == 2, return (leaf1, leaf2).
    # 3. Find a split S in the set such that |S| = 2?
    #    If such a split exists, it corresponds to a cherry (two leaves joined by an internal node).
    #    We can replace the two leaves in S with a new node, and remove S from the split set?
    #    No, the other splits will now refer to the new node.
    
    # This "reduction" approach is standard.
    
    # Let's implement a reduction algorithm:
    # 1. Find a cherry: a split S with |S| = 2.
    #    If found, let S = {a, b}.
    #    Create a new leaf c.
    #    Replace a and b with c in all other splits.
    #    Remove S from the set.
    #    Recurse on the new set of leaves.
    #    Return (a, b, rest_of_tree).
    
    # 2. If no cherry is found (which can happen for non-binary trees or specific topologies),
    #    find a split S with minimum size > 1.
    #    This is trickier.
    
    # For binary trees, there is always a cherry.
    # NJ produces binary trees.
    # The exhaustive search for n<=5 produces binary trees.
    
    # So we can rely on the existence of a cherry.
    
    leaves = set(labels)
    current_splits = set(splits)
    
    def reduce(leaves, splits):
        n = len(leaves)
        if n == 1:
            return list(leaves)[0]
        if n == 2:
            return "(" + ",".join(sorted(leaves)) + ")"
        
        # Find a cherry
        cherry = None
        for s in splits:
            if len(s) == 2 and s.issubset(leaves):
                cherry = s
                break
        
        if cherry is None:
            # Fallback: return a star tree or raise error
            # For safety, return a star
            return "(" + ",".join(sorted(leaves)) + ")"
        
        a, b = list(cherry)
        new_leaf = f"node_{a}_{b}"
        
        new_leaves = (leaves - cherry) | {new_leaf}
        
        new_splits = set()
        for s in splits:
            if s == cherry:
                continue
            new_s = set()
            for x in s:
                if x == a or x == b:
                    new_s.add(new_leaf)
                else:
                    new_s.add(x)
            # Normalize new_s?
            # The splits are unrooted.
            # If new_s is empty, skip.
            if new_s:
                # We need to ensure we don't add duplicates or complements.
                # But since we are reducing, the structure is preserved.
                # We just add the set.
                # Note: We should keep the smaller side?
                # The input splits were normalized.
                # The new split might not be.
                # Let's just add it. The reduction logic handles it.
                new_splits.add(frozenset(new_s))
        
        # Recurse
        subtree = reduce(new_leaves, new_splits)
        
        # Reconstruct
        # The new leaf new_leaf corresponds to the cherry (a, b)
        # So we replace new_leaf in the subtree string with (a, b)
        # But new_leaf is a unique string, so we can just do a string replacement?
        # Careful: what if "a" or "b" appears elsewhere?
        # a and b are removed from new_leaves, so they don't appear in subtree.
        # So we can safely replace new_leaf with (a, b).
        
        return subtree.replace(new_leaf, f"({a},{b})")
    
    return reduce(leaves, current_splits)

def _neighbor_joining(dist_matrix, labels):
    """
    Perform Neighbor Joining.
    Returns Newick string.
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"
        
    D = dist_matrix.copy()
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    while len(nodes) > 3:
        m = len(nodes)
        r = np.zeros(m)
        for i_idx, i in enumerate(nodes):
            for j_idx, j in enumerate(nodes):
                if i != j:
                    r[i_idx] += D[i, j]
                    
        Q = np.zeros((m, m))
        for i_idx, i in enumerate(nodes):
            for j_idx, j in enumerate(nodes):
                if i != j:
                    Q[i_idx, j_idx] = (m - 2) * D[i, j] - r[i_idx] - r[j_idx]
                    
        min_val = float('inf')
        min_i_idx = -1
        min_j_idx = -1
        for i_idx in range(m):
            for j_idx in range(i_idx + 1, m):
                if Q[i_idx, j_idx] < min_val:
                    min_val = Q[i_idx, j_idx]
                    min_i_idx = i_idx
                    min_j_idx = j_idx
                    
        i = nodes[min_i_idx]
        j = nodes[min_j_idx]
        
        new_tree_str = "(" + trees[i] + "," + trees[j] + ")"
        
        new_nodes = [x for x in nodes if x != i and x != j]
        
        new_m = m - 1
        new_D = np.zeros((new_m, new_m))
        
        new_id = f"n{len(trees)}"
        trees[new_id] = new_tree_str
        
        # Map old indices to new indices
        # new_nodes contains the remaining original nodes.
        # The new node is appended at the end.
        
        # Fill new_D
        for a_idx, a in enumerate(new_nodes):
            for b_idx, b in enumerate(new_nodes):
                if a == b:
                    continue
                new_D[a_idx, b_idx] = D[a, b]
        
        # Distances to new node
        for a_idx, a in enumerate(new_nodes):
            d_new = 0.5 * (D[i, a] + D[j, a] - D[i, j])
            new_D[a_idx, new_m - 1] = d_new
            new_D[new_m - 1, a_idx] = d_new
        
        new_nodes.append(new_id)
        D = new_D
        nodes = new_nodes
        
    return "(" + ",".join(trees[node] for node in nodes) + ");"

def _quartet_score(dist_matrix, quartet):
    """
    Compute the quartet score for a given quartet topology.
    Returns the best quartet split as a frozenset of two leaves.
    """
    a, b, c, d = quartet
    # Three possible splits
    # Split 1: {a,b} | {c,d}
    score1 = dist_matrix[a, c] + dist_matrix[b, d] - dist_matrix[a, d] - dist_matrix[b, c]
    # Split 2: {a,c} | {b,d}
    score2 = dist_matrix[a, b] + dist_matrix[c, d] - dist_matrix[a, d] - dist_matrix[b, c]
    # Split 3: {a,d} | {b,c}
    score3 = dist_matrix[a, b] + dist_matrix[c, d] - dist_matrix[a, c] - dist_matrix[b, d]
    
    scores = [score1, score2, score3]
    best_idx = int(np.argmin(scores))
    
    if best_idx == 0:
        return frozenset({a, b})
    elif best_idx == 1:
        return frozenset({a, c})
    else:
        return frozenset({a, d})

def _exhaustive_quartet_search(dist_matrix, labels):
    """
    Exhaustive quartet-based topology search for small n (<= 7).
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        else:
            return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"
    
    # For n=4, 5, 6, 7
    # We can enumerate all binary tree topologies.
    # Number of topologies:
    # n=4: 3
    # n=5: 15
    # n=6: 105
    # n=7: 945
    
    # We'll generate all possible sets of splits.
    
    def all_binary_tree_splits(leaves):
        """
        Generate all sets of non-trivial splits for binary unrooted trees on the given leaves.
        """
        n_leaves = len(leaves)
        if n_leaves <= 2:
            yield set()
        else:
            # Fix leaf 0 to be in the first group to avoid duplicates.
            # Choose a subset S of leaves containing 0, with 1 <= |S| <= n_leaves - 1.
            # To avoid duplicates, we can restrict |S| <= n_leaves/2?
            # No, because the tree is unrooted.
            # If we choose S, the split is S | L\S.
            # We should only generate each split once.
            # So we can require that 0 in S.
            # And we can vary the size of S from 1 to n_leaves - 1.
            # But if we choose S and L\S, they define the same split.
            # So we can just iterate over all subsets S containing 0, with |S| < n_leaves.
            
            # However, to generate the *topology*, we need to combine splits from S and L\S.
            
            # Let's iterate over all subsets S containing 0.
            # To avoid generating the same topology twice (S and L\S),
            # we can impose an ordering, e.g., the smallest leaf in S is 0,
            # and we only consider S if the smallest leaf in L\S is > 0?
            # Actually, since 0 is always in S, L\S never contains 0.
            # So S and L\S are distinct.
            # But do they generate the same topology?
            # If we choose S, we generate splits for S and L\S, and add S.
            # If we choose L\S, we generate splits for L\S and S, and add L\S.
            # The set of splits is the same: Splits(S) U Splits(L\S) U {S}.
            # So yes, they are the same.
            
            # So we can just iterate over all subsets S containing 0, with 1 <= |S| <= n_leaves - 1.
            # And yield the combined splits.
            
            # To avoid duplicates, we can just pick one representative.
            # Let's pick S such that 0 in S.
            # This covers all splits exactly once?
            # No, a split is defined by the partition.
            # If we pick S, we get the split S|L\S.
            # If we pick L\S, we get the split L\S|S.
            # These are the same split.
            # So if we iterate over all S containing 0, we will generate each split exactly once
            # (because for any split, exactly one side contains 0).
            
            # So we can do:
            for k in range(1, n_leaves):
                # Choose k-1 other leaves to join 0 in S
                other_leaves = list(range(1, n_leaves))
                for subset in combinations(other_leaves, k - 1):
                    S = frozenset([0] + list(subset))
                    # S must be a proper subset
                    if S == frozenset(leaves):
                        continue
                    
                    # Generate splits for S
                    for splits_S in all_binary_tree_splits(list(S)):
                        # Generate splits for L\S
                        L_minus_S = list(frozenset(leaves) - S)
                        for splits_L in all_binary_tree_splits(L_minus_S):
                            # Combine
                            all_splits = set(splits_S) | set(splits_L)
                            all_splits.add(S)
                            # Normalize?
                            # The splits from recursion are already normalized?
                            # No, they are just sets.
                            # We should normalize them to be the smaller side?
                            # Or just keep them as is, as long as they are consistent.
                            # For the purpose of comparison, we should normalize.
                            
                            # Let's yield the set of splits.
                            yield all_splits
    
    # Score a tree
    def score_tree(tree_splits):
        """
        Score a tree topology given by its set of splits.
        """
        # For each quartet, check consistency
        consistent = 0
        total = 0
        
        for quartet in combinations(range(n), 4):
            a, b, c, d = quartet
            total += 1
            
            # Determine the split implied by the tree for this quartet
            # The tree implies a split {x, y} | {z, w} if {x, y} is a subset of some split in the tree?
            # Actually, we can check which of the 3 possible splits is in the tree.
            
            split_ab = frozenset({a, b})
            split_ac = frozenset({a, c})
            split_ad = frozenset({a, d})
            
            # Check if any of these (or their complements) are in the tree
            # Note: The tree splits are on the full leaf set.
            # The quartet splits are on the 4 leaves.
            # We need to check if the restriction of the tree splits to the quartet
            # matches one of the 3 possible quartet splits.
            
            # A split S in the tree restricts to S ∩ {a,b,c,d}.
            # If |S ∩ {a,b,c,d}| == 2, then it defines a split on the quartet.
            
            implied_split = None
            for s in tree_splits:
                inter = s.intersection(frozenset({a, b, c, d}))
                if len(inter) == 2:
                    implied_split = inter
                    break
            
            if implied_split is None:
                # This shouldn't happen for a valid binary tree
                continue
            
            # Compare with the best quartet split
            best_split = _quartet_score(dist_matrix, quartet)
            # Normalize best_split to be the smaller side?
            # _quartet_score returns a set of 2 leaves.
            # It might be {a,b}, {a,c}, or {a,d}.
            # These are all size 2.
            
            # Check if implied_split matches best_split
            if implied_split == best_split:
                consistent += 1
        
        return consistent / total if total > 0 else 0
    
    # Find the best tree
    best_score = -1
    best_tree = None
    
    for tree_splits in all_binary_tree_splits(list(range(n))):
        # Normalize the splits for comparison?
        # The splits are sets of leaf indices.
        # We should ensure they are consistent (no duplicates, no complements).
        # The generation logic should ensure this.
        
        score = score_tree(tree_splits)
        
        if score > best_score:
            best_score = score
            best_tree = tree_splits
    
    # Convert best_tree to Newick
    # We need to map indices to labels
    # best_tree contains sets of indices.
    # _splits_to_newick expects sets of labels.
    
    # Let's convert
    index_to_label = {i: labels[i] for i in range(n)}
    
    label_splits = set()
    for s in best_tree:
        label_s = frozenset(index_to_label[i] for i in s)
        # Normalize
        if len(label_s) <= n // 2:
            label_splits.add(label_s)
        else:
            label_splits.add(frozenset(set(labels) - set(label_s)))
    
    newick = _splits_to_newick(label_splits, labels)
    return newick

def _spectral_embedding(dist_matrix, labels):
    """
    Perform spectral embedding of the distance matrix.
    Returns coordinates of shape (n, k).
    """
    n = len(labels)
    D = dist_matrix.copy()
    
    # Center the distance matrix
    # C = H D^2 H, where H = I - (1/n) 11^T
    # For distances, we usually square them first if using classical MDS.
    # But for tree topology, we might just use the distances directly?
    # Classical MDS uses squared distances.
    
    D2 = D ** 2
    J = np.ones((n, n)) / n
    H = np.eye(n) - J
    C = H @ D2 @ H
    
    # Eigendecomposition
    # We want the top k eigenvectors
    k = min(n - 1, 5) # Use top 5 dimensions
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top k
    coords = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
    
    return coords

def _ward_clustering(coords, labels):
    """
    Perform Ward's linkage hierarchical clustering on the coordinates.
    Returns a Newick string.
    """
    n = len(coords)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # We need to implement Ward's linkage.
    # This is a bit involved.
    # Let's use a simple recursive bisection based on the first principal component?
    # Or use scipy.cluster.hierarchy?
    # The prompt says SciPy is available.
    
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    # We have coordinates, so we can compute the pairwise distance matrix from them.
    # Or just use the original distance matrix?
    # Ward's linkage works on a distance matrix.
    # Let's use the original distance matrix D.
    
    # But the inspiration suggests using the embedded coordinates.
    # So let's compute the distance matrix from the coordinates.
    
    # Actually, Ward's linkage minimizes the increase in total within-cluster variance.
    # It works on a distance matrix.
    
    # Let's use the original distance matrix D for Ward's linkage.
    # The spectral embedding is used to... wait, the strategy says "recovered by applying a hierarchical clustering algorithm (such as Ward's linkage) to the embedded coordinates".
    # So we should use the distances between the embedded points.
    
    # Compute pairwise Euclidean distances between coordinates
    diff = coords[:, None, :] - coords[None, :, :]
    euclidean_dist = np.sqrt((diff ** 2).sum(axis=2))
    
    # Ward's linkage
    # scipy.cluster.hierarchy.linkage expects a condensed distance matrix
    condensed = squareform(euclidean_dist)
    Z = linkage(condensed, method='ward')
    
    # Convert the linkage matrix to Newick
    # This is a standard task.
    
    def linkage_to_newick(Z, labels):
        n = len(labels)
        # Z has n-1 rows
        # Each row is [i, j, d, c]
        # i and j are cluster indices.
        # Clusters 0..n-1 are leaves.
        # Clusters n..2n-2 are internal nodes.
        
        tree_str = {i: labels[i] for i in range(n)}
        
        for i in range(n - 1):
            a = int(Z[i, 0])
            b = int(Z[i, 1])
            new_id = n + i
            tree_str[new_id] = f"({tree_str[a]},{tree_str[b]})"
        
        return tree_str[2 * n - 3]
    
    return linkage_to_newick(Z, labels)

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 250 # Safety margin
    
    D, labels = _compute_jc_distance_matrix(sequences)
    n = len(labels)
    
    # For small n (<= 7), use exhaustive quartet-based search
    if n <= 7:
        newick = _exhaustive_quartet_search(D, labels)
    else:
        # For larger n, use NJ
        # We can also try spectral embedding + Ward's linkage and compare?
        # But NJ is robust.
        # Let's stick with NJ for now, as spectral embedding might be less accurate for topology.
        newick = _neighbor_joining(D, labels)
    
    return newick
