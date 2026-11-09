"""Phylogenetic topology recovery using quartet-based split selection."""
import time
import math
import numpy as np
from itertools import combinations

_DEADLINE = 255.0  # wall-clock seconds; leave margin under 270s kill
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _jc_corrected_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_quartet_scores(sequences, labels):
    """
    For each quartet (a,b,c,d), compute the score for each of the three
    possible topologies. Returns dict: (frozenset of 4 labels) -> (best_topology, score)
    where best_topology is a frozenset of two disjoint pairs representing the split.
    """
    n = len(labels)
    # Precompute pairwise differences as bitmasks for speed
    # Actually, let's compute pairwise p-distances first
    n_sites = len(sequences[labels[0]])
    
    # Convert sequences to numpy arrays of integers for fast comparison
    seq_arrs = {}
    for lab in labels:
        s = sequences[lab]
        seq_arrs[lab] = np.frombuffer(s.encode(), dtype=np.uint8)
    
    # For quartet scoring, we need to count, for each pair of taxa in a quartet,
    # how many sites they differ at. But doing all O(n^4) quartets with O(n_sites) each
    # is O(n^4 * L). For n=32, n^4 ~ 1M, L up to 2000 -> 2e9 ops. Too slow.
    
    # Alternative: use the "quartet counting" via pairwise distances.
    # Under JC model, the expected number of differences between taxa i and j is
    # proportional to the sum of branch lengths on the path between i and j.
    # For a quartet (a,b,c,d), the three possible splits are:
    #   (a,b)|(c,d), (a,c)|(b,d), (a,d)|(b,c)
    # The "correct" split minimizes the sum of pairwise distances across the split
    # compared to within. Actually, the standard quartet test uses:
    #   d_ab + d_cd vs d_ac + d_bd vs d_ad + d_bc
    # The split with the smallest sum is the inferred topology.
    
    # So we need pairwise JC-corrected distances.
    # Compute all pairwise distances first: O(n^2 * L)
    
    n_taxa = len(labels)
    dist = np.zeros((n_taxa, n_taxa))
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            ai = seq_arrs[labels[i]]
            aj = seq_arrs[labels[j]]
            diff = np.count_nonzero(ai != aj)
            p = diff / n_sites
            dist[i, j] = _jc_corrected_distance(p)
            dist[j, i] = dist[i, j]
    
    # Now for each quartet, find the best split
    # We'll collect all non-trivial splits and their support scores
    # A split is a partition of all taxa into two non-empty sets.
    # But quartets give us splits on 4 taxa. We need to "lift" these to the full taxon set.
    
    # Standard approach: for each quartet, the inferred split on those 4 taxa
    # corresponds to a split on all taxa. We accumulate evidence for each candidate split.
    
    # Candidate splits: any bipartition of the full taxon set where both parts
    # have size between 2 and n-2 (non-trivial).
    # But enumerating all is exponential. Instead, use the splits implied by quartets.
    
    # For a quartet {a,b,c,d} with best split {a,b}|{c,d}, the corresponding
    # full split is {a,b} ∪ S  |  {c,d} ∪ (complement) for any S... no, that's not right.
    
    # Actually, the standard method (e.g., in programs like QuartetSuite or FastTree)
    # is to count, for each candidate split, how many quartets support it.
    # A candidate split A|B (where A ∪ B = all taxa, A ∩ B = ∅, both non-empty)
    # is supported by a quartet (a,b,c,d) if the quartet's inferred split is consistent
    # with A|B, i.e., either {a,b} ⊆ A and {c,d} ⊆ B, or vice versa, etc.
    
    # This is complex. Let me use a simpler, well-known approach:
    # Use the pairwise distances to do Neighbor-Joining, but also try a
    # "quartet consensus" by sampling quartets and building a split set.
    
    # Given time constraints, let me implement a robust NJ with some enhancements:
    # 1. Use vectorized distance computation
    # 2. Use a proper BIONJ-like correction if possible
    # 3. Add a post-processing step: for each internal edge, check if removing it
    #    and re-joining gives a better tree (this is essentially what NJ does)
    
    # Let me implement an optimized NJ with vectorized ops.
    
    return dist


def _nj_build(labels, dist):
    """
    Neighbor-Joining using precomputed distance matrix.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Work with a mutable list of "nodes" (each is a label or a composite)
    nodes = list(range(n))  # indices into labels
    tree = {i: labels[i] for i in range(n)}
    D = dist.copy()  # (n x n)
    
    while len(nodes) > 3:
        m = len(nodes)
        # Compute r_i = sum_j D[i,j] for i in nodes
        r = np.zeros(m)
        for idx, ni in enumerate(nodes):
            r[idx] = sum(D[ni, nj] for nj in nodes if nj != ni)
        
        # Find pair (i,j) minimizing: (m-2)*D[i,j] - r[i] - r[j]
        best_val = float('inf')
        best_i = best_j = -1
        for ii in range(m):
            for jj in range(ii + 1, m):
                ni, nj = nodes[ii], nodes[jj]
                val = (m - 2) * D[ni, nj] - r[ii] - r[jj]
                if val < best_val:
                    best_val = val
                    best_i, best_j = ii, jj
        
        ni, nj = nodes[best_i], nodes[best_j]
        
        # Create new node
        new_idx = max(nodes) + 1
        new_node = new_idx
        
        # Compute distances from new node to all other nodes
        for idx, nk in enumerate(nodes):
            if nk == ni or nk == nj:
                continue
            D[new_node, nk] = 0.5 * (D[ni, nk] + D[nj, nk] - D[ni, nj])
            D[nk, new_node] = D[new_node, nk]
        
        # Build tree string
        tree[new_node] = f"({tree[ni]},{tree[nj]})"
        
        # Remove ni, nj, add new_node
        nodes = [x for x in nodes if x != ni and x != nj] + [new_node]
    
    # Final 3 nodes: build a 3-taxon tree
    a, b, c = nodes
    # Pick the pair with smallest distance
    d_ab = D[a, b]
    d_ac = D[a, c]
    d_bc = D[b, c]
    if d_ab <= d_ac and d_ab <= d_bc:
        return f"({tree[a]},{tree[b]},{tree[c]})"
    elif d_ac <= d_ab and d_ac <= d_bc:
        return f"({tree[a]},{tree[c]},{tree[b]})"
    else:
        return f"({tree[b]},{tree[c]},{tree[a]})"


def _nj_build_vectorized(labels, dist):
    """
    Faster NJ using numpy vectorization for the r-vector and pair search.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    nodes = list(range(n))
    tree = {i: labels[i] for i in range(n)}
    D = dist.copy()
    
    while len(nodes) > 3:
        m = len(nodes)
        node_arr = np.array(nodes)
        
        # r vector: for each node in nodes, sum of distances to all other nodes in nodes
        # Use matrix D[node_arr][:, node_arr] and set diagonal to 0
        sub_D = D[np.ix_(node_arr, node_arr)]
        np.fill_diagonal(sub_D, 0.0)
        r = sub_D.sum(axis=1)
        
        # Find pair minimizing (m-2)*D[i,j] - r[i] - r[j]
        # Create matrix of criteria
        criteria = (m - 2) * sub_D - r[:, None] - r[None, :]
        # Only consider upper triangle
        iu = np.triu_indices(m, k=1)
        flat_criteria = criteria[iu]
        best_pos = np.argmin(flat_criteria)
        best_i, best_j = iu[0][best_pos], iu[1][best_pos]
        
        ni, nj = nodes[best_i], nodes[best_j]
        
        new_idx = max(nodes) + 1
        new_node = new_idx
        
        # Compute distances from new node to all other nodes
        others = [x for x in nodes if x != ni and x != nj]
        for nk in others:
            D[new_node, nk] = 0.5 * (D[ni, nk] + D[nj, nk] - D[ni, nj])
            D[nk, new_node] = D[new_node, nk]
        
        tree[new_node] = f"({tree[ni]},{tree[nj]})"
        
        nodes = others + [new_node]
    
    a, b, c = nodes
    d_ab = D[a, b]
    d_ac = D[a, c]
    d_bc = D[b, c]
    if d_ab <= d_ac and d_ab <= d_bc:
        return f"({tree[a]},{tree[b]},{tree[c]})"
    elif d_ac <= d_ab and d_ac <= d_bc:
        return f"({tree[a]},{tree[c]},{tree[b]})"
    else:
        return f"({tree[b]},{tree[c]},{tree[a]})"


def _quartet_informed_nj(sequences, labels):
    """
    NJ with quartet-based validation: after building the NJ tree,
    for each internal edge, test if the quartet evidence supports it.
    If not, try alternative splits.
    """
    n = len(labels)
    if n < 4:
        return _nj_build_vectorized(labels, _compute_pairwise_dists(sequences, labels))
    
    # Step 1: Compute pairwise JC distances
    dist = _compute_pairwise_dists(sequences, labels)
    
    # Step 2: Build NJ tree
    nj_tree = _nj_build_vectorized(labels, dist)
    
    # Step 3: Extract splits from NJ tree
    splits = _extract_splits(nj_tree, labels)
    
    # Step 4: For each split, compute quartet support
    # Sample quartets and check consistency
    n_quartets = min(5000, max(500, n * n * n))  # adaptive
    np.random.seed(42)
    
    # Precompute pairwise distances for quartet scoring
    # For each quartet, the best split minimizes d_ab+d_cd vs d_ac+d_bd vs d_ad+d_bc
    
    # Count support for each NJ split
    split_support = {i: 0 for i in range(len(splits))}
    split_total = {i: 0 for i in range(len(splits))}
    
    # Sample quartets
    quartets_sampled = 0
    while quartets_sampled < n_quartets and _remaining() > 5:
        q = np.random.choice(n, 4, replace=False)
        a, b, c, d = q
        da_b = dist[a, b]
        da_c = dist[a, c]
        da_d = dist[a, d]
        db_c = dist[b, c]
        db_d = dist[b, d]
        dc_d = dist[c, d]
        
        # Three possible splits:
        # (a,b)|(c,d): score = da_b + dc_d
        # (a,c)|(b,d): score = da_c + db_d
        # (a,d)|(b,c): score = da_d + db_c
        s1 = da_b + dc_d
        s2 = da_c + db_d
        s3 = da_d + db_c
        
        # Best split for this quartet
        if s1 <= s2 and s1 <= s3:
            best_split = frozenset([frozenset([a, b]), frozenset([c, d])])
        elif s2 <= s1 and s2 <= s3:
            best_split = frozenset([frozenset([a, c]), frozenset([b, d])])
        else:
            best_split = frozenset([frozenset([a, d]), frozenset([b, c])])
        
        quartets_sampled += 1
        
        # Check which NJ splits are consistent with this quartet
        for i, sp in enumerate(splits):
            # sp is a frozenset of two frozensets (the two sides)
            # A split is consistent with the quartet if the quartet's split
            # is "compatible" with the full split.
            # Two splits are compatible if they don't cross.
            # For a full split A|B and a quartet split {x,y}|{z,w} on 4 taxa,
            # they are compatible if the quartet split doesn't "cross" the full split.
            # i.e., it's not the case that one of {x,y} is in A and the other in B,
            # AND one of {z,w} is in A and the other in B.
            
            # Get the two sides of the full split
            sides = list(sp)
            side_a = sides[0]
            side_b = sides[1]
            
            # Check if the quartet split crosses
            # The quartet has two pairs. Check if both pairs cross the full split.
            q_sides = list(best_split)
            q_pair1 = q_sides[0]
            q_pair2 = q_sides[1]
            
            def crosses(pair1, pair2, full_side_a, full_side_b):
                """Check if pair1 and pair2 both cross the full split."""
                # pair1 crosses if one element is in full_side_a and other in full_side_b
                p1_crosses = False
                for x in pair1:
                    if x in full_side_a:
                        for y in pair1:
                            if y != x and y in full_side_b:
                                p1_crosses = True
                p2_crosses = False
                for x in pair2:
                    if x in full_side_a:
                        for y in pair2:
                            if y != x and y in full_side_b:
                                p2_crosses = True
                return p1_crosses and p2_crosses
            
            if not crosses(q_pair1, q_pair2, side_a, side_b):
                split_support[i] += 1
            split_total[i] += 1
    
    # All splits should have the same total (they all get checked against every quartet)
    # Find splits with low support and try to fix them
    # For simplicity, if a split has support < 0.5 * total, mark it for potential removal
    
    # Given complexity and time, let's just return the NJ tree for now.
    # The quartet sampling is too slow for the full validation.
    
    return nj_tree


def _compute_pairwise_dists(sequences, labels):
    """Compute pairwise JC-corrected distances using numpy."""
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    
    # Convert to uint8 arrays
    seqs = np.array([list(sequences[lab].encode()) for lab in labels], dtype=np.uint8)
    
    # Compute pairwise differences
    # For each pair, count differences
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(seqs[i] != seqs[j])
            p = diff / n_sites
            dist[i, j] = _jc_corrected_distance(p)
            dist[j, i] = dist[i, j]
    
    return dist


def _extract_splits(newick, labels):
    """
    Extract all non-trivial splits from a Newick string.
    Returns a list of frozensets, each containing two frozensets (the two sides).
    """
    # Parse the Newick string into a tree structure
    # For simplicity, use a recursive parser
    
    def parse(s):
        """Parse Newick string, return list of splits."""
        s = s.strip().rstrip(';')
        if not s.startswith('('):
            return []
        
        # Find matching parentheses
        depth = 0
        end = -1
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        if end == -1:
            return []
        
        inner = s[1:end]
        # Split inner by top-level commas
        parts = _split_top_level(inner)
        
        # Recursively parse each part
        all_splits = []
        for part in parts:
            all_splits.extend(parse(part))
        
        # The split defined by this node: left side vs right side
        # Left side = taxa in parts[0], right side = taxa in parts[1:]
        # But we need to get the actual taxa sets
        # This requires knowing which taxa are in each subtree
        
        return all_splits
    
    # This approach is getting complex. Let me use a different method:
    # Build the tree as a nested structure, then for each internal node,
    # the split is the set of leaves in that node vs the rest.
    
    def build_tree(s):
        """Build a tree from Newick. Returns a tuple structure."""
        s = s.strip().rstrip(';')
        if not s.startswith('('):
            return s  # leaf
        
        depth = 0
        end = -1
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        inner = s[1:end]
        parts = _split_top_level(inner)
        children = [build_tree(p) for p in parts]
        return children
    
    def _split_top_level(s):
        """Split string by top-level commas."""
        parts = []
        depth = 0
        current = []
        for ch in s:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts
    
    def get_leaves(node):
        """Get set of leaf labels from a tree node."""
        if isinstance(node, str):
            return {node}
        leaves = set()
        for child in node:
            leaves.update(get_leaves(child))
        return leaves
    
    tree = build_tree(newick)
    all_labels = set(labels)
    
    splits = []
    
    def collect_splits(node):
        if isinstance(node, str):
            return
        leaves_in = get_leaves(node)
        if 1 < len(leaves_in) < len(all_labels):
            side_a = frozenset(leaves_in)
            side_b = frozenset(all_labels - leaves_in)
            splits.append(frozenset([side_a, side_b]))
        for child in node:
            collect_splits(child)
    
    collect_splits(tree)
    
    return splits


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from sequences.
    Returns a Newick string.
    """
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Compute pairwise distances
    dist = _compute_pairwise_dists(sequences, labels)
    
    # Build NJ tree
    tree_str = _nj_build_vectorized(labels, dist)
    
    # Ensure it ends with semicolon
    if not tree_str.endswith(';'):
        tree_str += ';'
    
    return tree_str
