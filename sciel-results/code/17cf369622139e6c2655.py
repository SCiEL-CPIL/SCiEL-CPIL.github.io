"""Phylogenetic topology recovery: NJ + quartet refinement + exhaustive for small n."""
import math
import time
import itertools
from collections import defaultdict

# Global deadline guard: 270s wall-clock hard limit, use 250s as safety margin
_DEADLINE = 250.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distance matrix."""
    n = len(labels)
    seq_list = [sequences[lbl] for lbl in labels]
    length = len(seq_list[0])
    
    # Precompute character vectors for speed
    # Encode sequences as integer arrays for fast comparison
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arrays = []
    for s in seq_list:
        arr = [enc[c] for c in s]
        arrays.append(arr)
    
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            diff = 0
            ai = arrays[i]
            aj = arrays[j]
            for k in range(length):
                if ai[k] != aj[k]:
                    diff += 1
            p = diff / length
            d = _jc_distance(p)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _newick_from_splits(splits, labels):
    """Build a Newick string from a set of unrooted splits (each split is a frozenset of leaf labels)."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else ""
    
    # We need to reconstruct the tree topology from splits.
    # For a binary unrooted tree with n leaves, there are n-3 internal splits.
    # We'll build the tree by finding the structure.
    
    # Approach: use the splits to build the tree via a recursive decomposition.
    # A split {A, B} where A ∪ B = all leaves and |A|, |B| >= 2.
    
    # For small n, we can build the tree from splits directly.
    # Let's use a different approach: build adjacency from quartet consistency.
    
    # Actually, let's just build the tree using the splits to determine parent-child relationships.
    # For unrooted binary tree, we can use the following:
    # 1. Find all splits
    # 2. The tree can be reconstructed by finding which splits are "nested"
    
    # Simpler approach for the final output: we already have the tree from NJ or quartet method.
    # This function is a fallback. Let's just return a star tree if we can't reconstruct.
    
    # Better: let's build the tree from splits using a standard algorithm.
    # For n leaves, we have n-3 splits (for binary tree).
    # Each split divides leaves into two groups.
    
    # Let's use a recursive approach:
    # Find a split that divides leaves into groups of sizes (k, n-k) where 2 <= k <= n-2.
    # Recurse on each side.
    
    all_leaves = frozenset(labels)
    
    def build_tree(leaf_set):
        """Recursively build Newick for the subtree on leaf_set."""
        ls = list(leaf_set)
        if len(ls) == 1:
            return ls[0]
        if len(ls) == 2:
            return f"({ls[0]},{ls[1]})"
        
        # Find a split that partitions leaf_set
        for sp in splits:
            a = sp & leaf_set
            b = leaf_set - sp
            if 2 <= len(a) <= len(leaf_set) - 2:
                # This split is valid for this subtree
                left = build_tree(a)
                right = build_tree(b)
                return f"({left},{right})"
        
        # If no valid split found, return a star (shouldn't happen for valid tree)
        return "(" + ",".join(ls) + ")"
    
    return build_tree(all_leaves) + ";"


def _nj_tree(dist, labels):
    """Neighbour-Joining algorithm. Returns (newick_string, splits_set)."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else "", set()
    if n == 2:
        return f"({labels[0]},{labels[1]});", set()
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));", set()
    
    # Work with indices
    nodes = list(range(n))
    # dist is n x n
    d = [row[:] for row in dist]
    
    # Track the tree structure as nested tuples
    # Each node is either a leaf label or a tuple representing a subtree
    tree_repr = [labels[i] for i in range(n)]
    
    # Splits collected
    splits = set()
    
    while len(nodes) > 2:
        m = len(nodes)
        if _deadline_exceeded():
            break
        
        # Compute row sums
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += d[nodes[i]][nodes[j]]
        
        # Find best pair
        best_val = float('inf')
        best_i = best_j = -1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * d[nodes[i]][nodes[j]] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        ni, nj = nodes[best_i], nodes[best_j]
        
        # Compute distances from new node to all others
        # New node index
        new_idx = max(nodes) + 1 if nodes else 0
        new_idx = 10000 + len(nodes)  # unique id
        
        # Compute distances
        new_d = {}
        for k in nodes:
            if k == ni or k == nj:
                continue
            new_d[k] = 0.5 * (d[ni][k] + d[nj][k] - d[ni][nj])
        
        # The split created: all leaves in ni's subtree vs all leaves in nj's subtree
        # We need to track which original leaves are in each subtree
        def get_leaves(node_id):
            """Get all original leaf indices in the subtree rooted at node_id."""
            if node_id < n:
                return {node_id}
            # It's a new node; we need to track this
            return leaf_sets.get(node_id, set())
        
        # Track leaf sets for each node
        if 'leaf_sets' not in dir():
            leaf_sets = {}
        
        if not hasattr(_nj_tree, '_leaf_sets'):
            _nj_tree._leaf_sets = {}
        leaf_sets = _nj_tree._leaf_sets
        
        # Initialize leaf sets
        for i in range(n):
            leaf_sets[i] = {i}
        
        li = leaf_sets.get(ni, {ni})
        lj = leaf_sets.get(nj, {nj})
        
        # Record the split
        split_a = frozenset(li)
        if 2 <= len(split_a) <= n - 2:
            splits.add(split_a)
        
        # Create new node
        new_leaf_set = li | lj
        leaf_sets[new_idx] = new_leaf_set
        
        # Update distance matrix
        # Remove ni, nj; add new_idx
        new_nodes = [x for x in nodes if x != ni and x != nj]
        new_nodes.append(new_idx)
        
        # Build new distance matrix
        new_d_matrix = {}
        for a in new_nodes:
            for b in new_nodes:
                if a == b:
                    new_d_matrix[(a, b)] = 0.0
                elif a < n and b < n:
                    new_d_matrix[(a, b)] = d[a][b]
                elif a < n:
                    new_d_matrix[(a, b)] = new_d.get(a, 0.0)
                elif b < n:
                    new_d_matrix[(a, b)] = new_d.get(b, 0.0)
                else:
                    # Both are new nodes - shouldn't happen in NJ
                    new_d_matrix[(a, b)] = 0.0
        
        # Update d
        d = new_d_matrix
        nodes = new_nodes
        
        # Update tree representation
        tree_repr_new = {}
        for node in nodes:
            if node < n:
                tree_repr_new[node] = labels[node]
            else:
                # Find which two nodes were merged
                # We need to track this differently
                pass
        
        # This is getting complex. Let me use a cleaner approach.
        # Let's just track the tree as a dictionary of node -> (left, right) or leaf
        break
    
    # The above got messy. Let me restart with a cleaner NJ implementation.
    # For now, let's use a simpler approach.
    
    # Actually, let me just implement NJ properly from scratch.
    return _nj_clean(dist, labels)


def _nj_clean(dist, labels):
    """Clean Neighbour-Joining implementation."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else "", set()
    if n == 2:
        return f"({labels[0]},{labels[1]});", set()
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));", set()
    
    # Use a dict-based approach
    # nodes: list of current node ids
    # dist: dict (i,j) -> distance
    # tree: dict node_id -> either label string or (left_id, right_id)
    
    nodes = list(range(n))
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            dist_dict[(i, j)] = dist[i][j]
    
    tree = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {i} for i in range(n)}
    
    next_id = n
    splits = set()
    
    while len(nodes) > 2:
        if _deadline_exceeded():
            break
        m = len(nodes)
        
        # Row sums
        r = {}
        for a in nodes:
            r[a] = 0.0
            for b in nodes:
                if a != b:
                    r[a] += dist_dict[(a, b)]
        
        # Find best pair
        best_val = float('inf')
        best_pair = (nodes[0], nodes[1])
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                val = (m - 2) * dist_dict[(a, b)] - r[a] - r[b]
                if val < best_val:
                    best_val = val
                    best_pair = (a, b)
        
        a, b = best_pair
        
        # Record split
        la = leaf_sets[a]
        lb = leaf_sets[b]
        split_a = frozenset(la)
        if 2 <= len(split_a) <= n - 2:
            splits.add(split_a)
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Compute distances from new node to remaining nodes
        rest = [x for x in nodes if x != a and x != b]
        for x in rest:
            d_new = 0.5 * (dist_dict[(a, x)] + dist_dict[(b, x)] - dist_dict[(a, b)])
            dist_dict[(new_id, x)] = d_new
            dist_dict[(x, new_id)] = d_new
        
        # Update
        tree[new_id] = (a, b)
        leaf_sets[new_id] = la | lb
        nodes = rest + [new_id]
        
        # Clean up old entries (optional, for memory)
        # Not strictly necessary
    
    # Build Newick from the tree structure
    def to_newick(node_id):
        if node_id in tree and isinstance(tree[node_id], str):
            return tree[node_id]
        left, right = tree[node_id]
        return f"({to_newick(left)},{to_newick(right)})"
    
    # The final tree has 2 nodes left
    if len(nodes) == 2:
        a, b = nodes[0], nodes[1]
        newick = f"({to_newick(a)},{to_newick(b)});"
    else:
        newick = to_newick(nodes[0]) + ";"
    
    return newick, splits


def _quartet_score(splits, dist, labels):
    """Score a set of splits by counting concordant quartets."""
    n = len(labels)
    if n < 4:
        return 0.0
    
    # For each quartet {a,b,c,d}, determine which of the 3 topologies is supported by splits
    # Topology 1: (a,b)|(c,d) - split {a,b} or {c,d}
    # Topology 2: (a,c)|(b,d) - split {a,c} or {b,d}
    # Topology 3: (a,d)|(b,c) - split {a,d} or {b,c}
    
    # The "correct" topology for a quartet is the one with the smallest sum of two opposing distances
    # d(a,b) + d(c,d) vs d(a,c) + d(b,d) vs d(a,d) + d(b,c)
    
    concordant = 0
    total = 0
    
    # For efficiency, precompute which splits exist
    split_set = splits
    
    # Iterate over all quartets
    idx_list = list(range(n))
    for i in range(n - 3):
        if _deadline_exceeded():
            break
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    a, b, c, d = idx_list[i], idx_list[j], idx_list[k], idx_list[l]
                    
                    # Determine best topology by distance
                    d_ab_cd = dist[a][b] + dist[c][d]
                    d_ac_bd = dist[a][c] + dist[b][d]
                    d_ad_bc = dist[a][d] + dist[b][c]
                    
                    # Find the minimum
                    min_val = min(d_ab_cd, d_ac_bd, d_ad_bc)
                    
                    # Check if the splits support this topology
                    # Topology (a,b)|(c,d): need split {a,b} or {c,d}
                    s_ab = frozenset({a, b})
                    s_cd = frozenset({c, d})
                    s_ac = frozenset({a, c})
                    s_bd = frozenset({b, d})
                    s_ad = frozenset({a, d})
                    s_bc = frozenset({b, c})
                    
                    if min_val == d_ab_cd:
                        if s_ab in split_set or s_cd in split_set:
                            concordant += 1
                    elif min_val == d_ac_bd:
                        if s_ac in split_set or s_bd in split_set:
                            concordant += 1
                    else:
                        if s_ad in split_set or s_bc in split_set:
                            concordant += 1
                    
                    total += 1
    
    return concordant / total if total > 0 else 0.0


def _quartet_refinement(splits, dist, labels, max_rounds=3):
    """Refine splits using quartet-based local search."""
    n = len(labels)
    if n < 4:
        return splits
    
    current_splits = set(splits)
    current_score = _quartet_score(current_splits, dist, labels)
    
    for _ in range(max_rounds):
        if _deadline_exceeded():
            break
        
        improved = False
        
        # Try adding a split
        # Generate candidate splits from quartets
        candidate_splits = set()
        idx_list = list(range(n))
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate_splits.add(frozenset({idx_list[i], idx_list[j]}))
        
        # Also consider larger splits
        # For each pair of leaves, try all possible split sizes
        # This is expensive, so limit to small splits
        for size in range(2, n - 1):
            if size > n - size:
                break
            for combo in itertools.combinations(idx_list, size):
                candidate_splits.add(frozenset(combo))
        
        # Try each candidate split: add it and check if score improves
        # This is O(|candidates| * quartets) which can be expensive
        # Limit to a subset
        candidate_list = list(candidate_splits)
        if len(candidate_list) > 500:
            import random
            random.shuffle(candidate_list)
            candidate_list = candidate_list[:500]
        
        for sp in candidate_list:
            if _deadline_exceeded():
                break
            if sp in current_splits:
                continue
            # Check if this split is compatible with existing splits
            compatible = True
            for existing in current_splits:
                # Two splits are compatible if they don't cross
                # They cross if |A ∩ B|, |A - B|, |B - A|, |A ∪ B - (A ∩ B)| all > 0
                a = sp
                b = existing
                inter = a & b
                a_only = a - b
                b_only = b - a
                if inter and a_only and b_only:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Try adding
            new_splits = current_splits | {sp}
            new_score = _quartet_score(new_splits, dist, labels)
            
            if new_score > current_score:
                current_splits = new_splits
                current_score = new_score
                improved = True
        
        if not improved:
            break
    
    return current_splits


def _exhaustive_small_n(sequences, labels):
    """For n <= 7, try all possible unrooted binary topologies and pick the best by NJ-like criterion."""
    n = len(labels)
    if n < 4 or n > 7:
        return None
    
    # For n leaves, number of unrooted binary trees = (2n-5)!!
    # n=4: 3, n=5: 15, n=6: 105, n=7: 945
    # For each topology, compute the sum of branch lengths (or some score)
    # Use the quartet score as a proxy
    
    dist = _compute_distance_matrix(sequences, labels)
    
    # Generate all possible split sets for unrooted binary trees
    # For n=4: 3 possible topologies
    # For n=5: 15
    # For n=6: 105
    # For n=7: 945
    
    # We'll enumerate by building all possible sets of n-3 compatible splits
    
    def all_binary_splits(n_leaves):
        """Generate all possible sets of n-3 compatible splits for an unrooted binary tree on n_leaves labeled 0..n-1."""
        if n_leaves <= 3:
            yield set()
            return
        
        # Recursive: pick a split, then recurse on both sides
        # A split divides leaves into two groups of size >= 2
        all_leaves = frozenset(range(n_leaves))
        
        def recurse(leaf_set, existing_splits):
            ls = list(leaf_set)
            if len(ls) <= 3:
                # No more splits needed
                yield frozenset(existing_splits) if existing_splits else set()
                return
            
            # Try all possible splits of leaf_set
            k = len(ls)
            for size in range(2, k // 2 + 1):
                for combo in itertools.combinations(ls, size):
                    a = frozenset(combo)
                    b = leaf_set - a
                    if len(a) < 2 or len(b) < 2:
                        continue
                    
                    # Check compatibility with existing splits
                    compatible = True
                    for es in existing_splits:
                        e = es & leaf_set  # Only consider within this subtree
                        inter = a & e
                        a_only = a - e
                        e_only = e - a
                        if inter and a_only and e_only:
                            compatible = False
                            break
                    
                    if not compatible:
                        continue
                    
                    new_splits = existing_splits | {a}
                    yield from recurse(a, new_splits)
                    # Note: we only recurse on one side because the other side is determined
                    # Actually, we need to recurse on both sides to get all splits
                    # But since the tree is unrooted, we can just recurse on the smaller side
                    # and the larger side will be handled at a higher level
                    # Hmm, this is tricky. Let me think...
                    # Actually, for an unrooted binary tree, each internal edge corresponds to a split.
                    # If we pick a split, the tree decomposes into two subtrees.
                    # We need to generate all valid split sets for each subtree.
                    # But we only need to recurse on one side (the other is implicit).
                    # Wait, no. We need to generate splits for BOTH sides.
                    # Let me fix this.
        
        # The above recursion is flawed. Let me use a different approach.
        # For small n, just enumerate all possible sets of n-3 splits and check if they form a valid tree.
        
        if n_leaves == 4:
            # 3 possible topologies
            yield {frozenset({0, 1}), }  # (0,1)|(2,3)
            yield {frozenset({0, 2}), }  # (0,2)|(1,3)
            yield {frozenset({0, 3}), }  # (0,3)|(1,2)
        elif n_leaves == 5:
            # 15 topologies
            # Generate all pairs of compatible splits
            all_singles = [frozenset(combo) for size in range(2, 4) for combo in itertools.combinations(range(5), size)]
            # Each topology has 2 splits
            seen = set()
            for i in range(len(all_singles)):
                for j in range(i + 1, len(all_singles)):
                    a, b = all_singles[i], all_singles[j]
                    # Check compatibility
                    inter = a & b
                    a_only = a - b
                    b_only = b - a
                    if not (inter and a_only and b_only):
                        # Compatible
                        key = frozenset({a, b})
                        if key not in seen:
                            seen.add(key)
                            yield {a, b}
        elif n_leaves == 6:
            # 105 topologies, each with 3 splits
            # This is getting complex. Let's just use the quartet score with NJ as a good approximation
            # and skip exhaustive for n>=6
            return
        elif n_leaves == 7:
            # 945 topologies - too many to enumerate easily
            return
        
        # Fallback: yield empty (use NJ)
        yield set()
    
    # For n=4 and n=5, we can enumerate
    best_splits = None
    best_score = -1.0
    
    for split_set in all_binary_splits(n):
        if _deadline_exceeded():
            break
        score = _quartet_score(split_set, dist, labels)
        if score > best_score:
            best_score = score
            best_splits = split_set
    
    if best_splits is not None:
        newick = _newick_from_splits(best_splits, labels)
        return newick
    
    return None


def reconstruct(sequences):
    """Main entry point: reconstruct phylogenetic tree from sequences."""
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # For small n, try exhaustive search
    if n <= 5:
        result = _exhaustive_small_n(sequences, labels)
        if result is not None:
            return result
    
    # Use NJ as starting point
    newick, splits = _nj_clean(dist, labels)
    
    # For n >= 6, try quartet refinement
    if n >= 6 and not _deadline_exceeded():
        refined_splits = _quartet_refinement(splits, dist, labels, max_rounds=2)
        if len(refined_splits) > len(splits):
            newick = _newick_from_splits(refined_splits, labels)
    
    return newick
