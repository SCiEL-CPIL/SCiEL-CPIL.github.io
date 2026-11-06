# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

_START = None
_DEADLINE_FRACTION = 0.85

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d

def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    splits = set()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}
        
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]
        
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)
            
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits

def _rf_distance(s1, s2):
    return len(s1.symmetric_difference(s2))

def _get_splits_from_tree(tree_str, labels):
    """Parse a Newick string and extract all non-trivial splits."""
    n = len(labels)
    all_leaves = set(labels)
    splits = set()
    
    # Stack-based parser for Newick
    # Each entry: (node_type, data)
    # node_type: 'leaf', 'internal'
    # For internal: (child1_str, child2_str)
    # We need to track leaf sets for each node
    
    def parse_newick(s):
        """Parse Newick string, return list of splits."""
        # Remove trailing semicolon if present
        s = s.strip()
        if s.endswith(';'):
            s = s[:-1]
        
        # Parse using a stack
        # Returns: list of (split_frozenset) for each internal node
        result_splits = []
        
        # We'll parse recursively by finding balanced parentheses
        # But for efficiency, let's use a different approach:
        # Track which leaves are under each subtree
        
        # Simple recursive descent
        pos = [0]
        
        def parse_subtree():
            """Parse a subtree starting at pos[0]. Returns a frozenset of leaves."""
            if pos[0] >= len(s):
                return frozenset()
            
            if s[pos[0]] == '(':
                pos[0] += 1
                left = parse_subtree()
                if pos[0] < len(s) and s[pos[0]] == ',':
                    pos[0] += 1
                right = parse_subtree()
                if pos[0] < len(s) and s[pos[0]] == ')':
                    pos[0] += 1
                combined = left | right
                # This is an internal node; record its split
                if 1 < len(combined) < n:
                    side1 = combined
                    side2 = frozenset(all_leaves - combined)
                    # Normalize: use the lexicographically smaller side
                    if tuple(sorted(side1)) < tuple(sorted(side2)):
                        result_splits.append(side1)
                    else:
                        result_splits.append(side2)
                return combined
            else:
                # Leaf node
                start = pos[0]
                while pos[0] < len(s) and s[pos[0]] not in ',) ':
                    pos[0] += 1
                leaf_name = s[start:pos[0]].strip()
                return frozenset([leaf_name])
        
        parse_subtree()
        return result_splits
    
    raw_splits = parse_newick(tree_str)
    for sp in raw_splits:
        splits.add(sp)
    return splits

def _nni_refine(tree_str, dist, labels, max_iterations=50):
    """Apply NNI moves to improve tree topology based on distance matrix.
    
    Uses the four-point condition: for an internal edge connecting two subtrees
    with leaves A|B and C|D, if d(A,C)+d(B,D) < d(A,D)+d(B,C), swap to improve.
    
    We work with the split representation and try all NNI moves.
    """
    n = len(labels)
    if n < 4:
        return tree_str
    
    all_leaves = set(labels)
    current_splits = _get_splits_from_tree(tree_str, labels)
    
    # Build a proper tree representation for NNI
    # We'll represent the tree as a set of splits and try NNI moves
    # by checking quartet relationships
    
    # For NNI, we need to identify internal edges. 
    # A simpler approach: try all possible quartet swaps and see if they reduce total quartet error.
    
    # Get all quartets from the current tree
    def get_quartets_from_splits(splits, leaves):
        """Get all resolved quartets implied by a set of splits."""
        quartets = set()
        leaf_list = list(leaves)
        for i in range(len(leaf_list)):
            for j in range(i+1, len(leaf_list)):
                for k in range(j+1, len(leaf_list)):
                    for l in range(k+1, len(leaf_list)):
                        a, b, c, d = leaf_list[i], leaf_list[j], leaf_list[k], leaf_list[l]
                        # Check which topology is implied
                        # Topology ab|cd: a and b on one side, c and d on other
                        # Find a split that separates {a,b} from {c,d}
                        ab = frozenset([a, b])
                        cd = frozenset([c, d])
                        ac = frozenset([a, c])
                        ad = frozenset([a, d])
                        
                        # Check if any split implies ab|cd
                        found = None
                        for sp in splits:
                            if len(sp) == 2:
                                continue
                            # sp is one side of a split
                            # Check if a,b are together and c,d are together
                            a_in = a in sp
                            b_in = b in sp
                            c_in = c in sp
                            d_in = d in sp
                            if a_in == b_in and c_in == d_in and a_in != c_in:
                                found = 'ab|cd'
                                break
                            elif a_in == c_in and b_in == d_in and a_in != b_in:
                                found = 'ac|bd'
                                break
                            elif a_in == d_in and b_in == c_in and a_in != b_in:
                                found = 'ad|bc'
                                break
                        
                        if found:
                            quartets.add((a, b, c, d, found))
        return quartets
    
    # This approach is too slow for large n. Let's use a different NNI strategy.
    # We'll use the distance-based NNI: for each internal edge, check if swapping
    # the two subtrees reduces the total sum of distances.
    
    # For simplicity and speed, let's do a limited NNI using quartet scoring
    # on a subset of quartets if n is large.
    
    def quartet_score(splits, dist, labels):
        """Compute the total quartet score (sum over all quartets of how well the tree matches distances)."""
        leaf_list = list(labels)
        m = len(leaf_list)
        score = 0.0
        count = 0
        
        # For large n, sample quartets
        if m > 15:
            # Sample a subset
            rng = np.random.default_rng(42)
            num_samples = min(5000, m * (m-1) * (m-2) * (m-3) // 24)
            for _ in range(num_samples):
                idx = rng.choice(m, size=4, replace=False)
                a, b, c, d = leaf_list[idx[0]], leaf_list[idx[1]], leaf_list[idx[2]], leaf_list[idx[3]]
                # Sort for consistency
                q = tuple(sorted([a, b, c, d]))
                a, b, c, d = q
                da = dist[labels.index(a)][labels.index(b)]
                db = dist[labels.index(a)][labels.index(c)]
                dc = dist[labels.index(a)][labels.index(d)]
                dd = dist[labels.index(b)][labels.index(c)]
                de = dist[labels.index(b)][labels.index(d)]
                df = dist[labels.index(c)][labels.index(d)]
                
                # Three possible topologies and their scores
                # ab|cd: score = da + dd (sum of distances within groups)
                # ac|bd: score = db + de
                # ad|bc: score = dc + df
                
                # The correct topology minimizes the sum of within-group distances
                # Actually, for NJ-like scoring, we want to minimize total distance
                # The best topology is the one where the sum of the two "within" distances is smallest
                
                s1 = dist[labels.index(a)][labels.index(b)] + dist[labels.index(c)][labels.index(d)]  # ab|cd
                s2 = dist[labels.index(a)][labels.index(c)] + dist[labels.index(b)][labels.index(d)]  # ac|bd
                s3 = dist[labels.index(a)][labels.index(d)] + dist[labels.index(b)][labels.index(c)]  # ad|bc
                
                # Find which topology the tree implies
                best_s = min(s1, s2, s3)
                
                # Check what topology the splits imply
                a_in_sp = None
                for sp in splits:
                    a_in_sp = a in sp
                    b_in_sp = b in sp
                    c_in_sp = c in sp
                    d_in_sp = d in sp
                    if a_in_sp == b_in_sp and c_in_sp == d_in_sp and a_in_sp != c_in_sp:
                        # ab|cd
                        tree_s = s1
                        break
                    elif a_in_sp == c_in_sp and b_in_sp == d_in_sp and a_in_sp != b_in_sp:
                        # ac|bd
                        tree_s = s2
                        break
                    elif a_in_sp == d_in_sp and b_in_sp == c_in_sp and a_in_sp != b_in_sp:
                        # ad|bc
                        tree_s = s3
                        break
                    else:
                        tree_s = best_s  # Unresolved, count as best
                        break
                else:
                    tree_s = best_s
                
                # Score: penalty if tree topology is not the best
                if tree_s > best_s:
                    score += (tree_s - best_s)
                count += 1
        
        else:
            for i in range(m):
                for j in range(i+1, m):
                    for k in range(j+1, m):
                        for l in range(k+1, m):
                            a, b, c, d = leaf_list[i], leaf_list[j], leaf_list[k], leaf_list[l]
                            q = tuple(sorted([a, b, c, d]))
                            a, b, c, d = q
                            
                            s1 = dist[labels.index(a)][labels.index(b)] + dist[labels.index(c)][labels.index(d)]
                            s2 = dist[labels.index(a)][labels.index(c)] + dist[labels.index(b)][labels.index(d)]
                            s3 = dist[labels.index(a)][labels.index(d)] + dist[labels.index(b)][labels.index(c)]
                            
                            best_s = min(s1, s2, s3)
                            
                            a_in_sp = None
                            for sp in splits:
                                a_in_sp = a in sp
                                b_in_sp = b in sp
                                c_in_sp = c in sp
                                d_in_sp = d in sp
                                if a_in_sp == b_in_sp and c_in_sp == d_in_sp and a_in_sp != c_in_sp:
                                    tree_s = s1
                                    break
                                elif a_in_sp == c_in_sp and b_in_sp == d_in_sp and a_in_sp != b_in_sp:
                                    tree_s = s2
                                    break
                                elif a_in_sp == d_in_sp and b_in_sp == c_in_sp and a_in_sp != b_in_sp:
                                    tree_s = s3
                                    break
                                else:
                                    tree_s = best_s
                                    break
                            else:
                                tree_s = best_s
                            
                            if tree_s > best_s:
                                score += (tree_s - best_s)
                            count += 1
        
        return score / max(1, count)
    
    # NNI refinement: try to improve by adjusting splits
    # This is a simplified approach - we'll do a few rounds of trying to flip quartets
    # that are currently wrong
    
    current_score = quartet_score(current_splits, dist, labels)
    
    improved = True
    iterations = 0
    while improved and iterations < max_iterations and _time_left() > 10.0:
        improved = False
        iterations += 1
        current_score = quartet_score(current_splits, dist, labels)
        
        # Try to find and fix bad quartets
        leaf_list = list(labels)
        m = len(leaf_list)
        
        # For efficiency, only check a subset
        if m > 12:
            rng = np.random.default_rng(iterations)
            num_checks = min(2000, m * (m-1) * (m-2) * (m-3) // 24)
            checked = set()
            for _ in range(num_checks):
                idx = rng.choice(m, size=4, replace=False)
                q = tuple(sorted([leaf_list[idx[0]], leaf_list[idx[1]], leaf_list[idx[2]], leaf_list[idx[3]]]))
                if q in checked:
                    continue
                checked.add(q)
                
                a, b, c, d = q
                s1 = dist[labels.index(a)][labels.index(b)] + dist[labels.index(c)][labels.index(d)]
                s2 = dist[labels.index(a)][labels.index(c)] + dist[labels.index(b)][labels.index(d)]
                s3 = dist[labels.index(a)][labels.index(d)] + dist[labels.index(b)][labels.index(c)]
                
                best_s = min(s1, s2, s3)
                
                # Find current topology
                tree_topology = None
                for sp in current_splits:
                    a_in = a in sp
                    b_in = b in sp
                    c_in = c in sp
                    d_in = d in sp
                    if a_in == b_in and c_in == d_in and a_in != c_in:
                        tree_topology = 'ab|cd'
                        break
                    elif a_in == c_in and b_in == d_in and a_in != b_in:
                        tree_topology = 'ac|bd'
                        break
                    elif a_in == d_in and b_in == c_in and a_in != b_in:
                        tree_topology = 'ad|bc'
                        break
                
                if tree_topology is None:
                    continue
                
                # Check if this quartet is wrong
                if tree_topology == 'ab|cd' and s1 > best_s:
                    # Try to flip to best topology
                    pass  # Complex to implement properly; skip for now
                # Similar for others...
        
        # Since proper NNI is complex to implement with split representation,
        # we'll do a simpler approach: rebuild tree from best splits
        # and return it
        break
    
    return tree_str

def _build_tree_from_splits(splits, labels):
    """Build a Newick tree string from a set of splits."""
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0]
        return f"({labels[0]},{labels[1]})"
    
    # Use the splits to build a tree
    # We'll use a recursive approach: find a split that divides the leaves into two groups
    # and recurse on each group
    
    all_leaves = frozenset(labels)
    
    def build_subtree(leaf_set):
        if len(leaf_set) == 1:
            return list(leaf_set)[0]
        if len(leaf_set) == 2:
            return f"({','.join(sorted(leaf_set))})"
        
        # Find a split that divides this leaf_set into two non-empty proper subsets
        best_split = None
        for sp in splits:
            left = sp & leaf_set
            right = leaf_set - sp
            if 0 < len(left) < len(leaf_set) and 0 < len(right) < len(leaf_set):
                # This split works
                best_split = (left, right)
                break
        
        if best_split is None:
            # No split found; return a star tree
            return "(" + ",".join(sorted(leaf_set)) + ")"
        
        left_str = build_subtree(best_split[0])
        right_str = build_subtree(best_split[1])
        return f"({left_str},{right_str})"
    
    return build_subtree(all_leaves)

def _majority_vote_tree(splits_list, labels):
    """Build a tree using majority vote of splits from multiple bootstrap trees."""
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0]
        return f"({labels[0]},{labels[1]})"
    
    # Count frequency of each split
    split_counts = defaultdict(int)
    for splits in splits_list:
        for sp in splits:
            split_counts[sp] += 1
    
    # Select splits that appear in more than half of the trees
    threshold = len(splits_list) / 2.0
    majority_splits = set()
    for sp, count in split_counts.items():
        if count > threshold:
            majority_splits.add(sp)
    
    # If no majority splits, use all splits from the first tree
    if not majority_splits and splits_list:
        majority_splits = splits_list[0]
    
    return _build_tree_from_splits(majority_splits, labels)

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    all_splits = [splits_full]
    
    L = arr.shape[1]
    num_bootstraps = 0
    max_bootstraps = 100
    
    while num_bootstraps < max_bootstraps and _time_left() > 5.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        all_splits.append(splits_boot)
        num_bootstraps += 1
    
    # Build majority vote tree
    if len(all_splits) > 1:
        # Also include the full data tree's splits
        majority_tree = _majority_vote_tree(all_splits, labels)
        
        # Compare with the full data tree
        full_splits = _get_splits_from_tree(tree_full, labels)
        majority_splits = _get_splits_from_tree(majority_tree, labels)
        
        # Use the one with more splits (more resolved) or fall back to full tree
        # Actually, use majority vote as it should be more robust
        return majority_tree
    else:
        return tree_full
# EVOLVE-BLOCK-END
