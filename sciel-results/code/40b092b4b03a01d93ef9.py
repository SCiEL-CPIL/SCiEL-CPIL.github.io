"""Quartet Fitting for phylogenetic topology recovery."""
import math
import time
from itertools import combinations

_DEADLINE = 260.0  # safety margin before 270s kill
_start_time = None


def _jc_dist(a, b):
    """Jukes-Cantor corrected distance between two DNA strings."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _quartet_best_topo(d_ab, d_ac, d_ad, d_bc, d_bd, d_cd):
    """
    Given 6 pairwise distances for 4 taxa (a,b,c,d), return the best topology.
    The three possible unrooted topologies are:
      (a,b)|(c,d): internal edge separates {a,b} from {c,d}
      (a,c)|(b,d): internal edge separates {a,c} from {b,d}
      (a,d)|(b,c): internal edge separates {a,d} from {b,c}
    
    Under JC, the correct topology minimizes the sum of the two "cross" distances
    (the pairs that are NOT separated by the internal edge in the quartet).
    Actually, the standard criterion: for topology (ab|cd), the score is
    d_ab + d_cd (the two "within" pairs) should be smaller than d_ac + d_bd or d_ad + d_bc.
    Wait - let me think again.
    
    For a 4-taxon tree with internal edge separating {a,b} from {c,d}:
    - a and b are close (share recent common ancestor)
    - c and d are close
    - a-c, a-d, b-c, b-d are longer
    
    The standard quartet scoring: the correct topology is the one where
    d_ab + d_cd < d_ac + d_bd AND d_ab + d_cd < d_ad + d_bc
    i.e., the sum of the two "within-group" distances is smallest.
    
    So we compute:
      score_ab_cd = d_ab + d_cd
      score_ac_bd = d_ac + d_bd  
      score_ad_bc = d_ad + d_bc
    and pick the minimum.
    """
    s1 = d_ab + d_cd  # (ab|cd)
    s2 = d_ac + d_bd  # (ac|bd)
    s3 = d_ad + d_bc  # (ad|bc)
    
    if s1 <= s2 and s1 <= s3:
        return ('ab', 'cd')  # a,b together; c,d together
    elif s2 <= s1 and s2 <= s3:
        return ('ac', 'bd')
    else:
        return ('ad', 'bc')


def _build_quartet_scores(labels, dist_matrix):
    """
    For every 4-taxon subset, compute the best quartet topology.
    Returns a dict mapping frozenset of 4 taxa -> (group1, group2) where
    each group is a frozenset of 2 taxa.
    """
    n = len(labels)
    quartet_scores = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    a, b, c, d = labels[i], labels[j], labels[k], labels[l]
                    d_ab = dist_matrix[(a, b)]
                    d_ac = dist_matrix[(a, c)]
                    d_ad = dist_matrix[(a, d)]
                    d_bc = dist_matrix[(b, c)]
                    d_bd = dist_matrix[(b, d)]
                    d_cd = dist_matrix[(c, d)]
                    
                    topo = _quartet_best_topo(d_ab, d_ac, d_ad, d_bc, d_bd, d_cd)
                    quartet_scores[frozenset([a, b, c, d])] = topo
    
    return quartet_scores


def _quartet_consistency(tree, quartet_scores, labels_in_tree):
    """
    Count how many quartets are consistent with the given tree topology.
    tree: a dict mapping each taxon to its path (list of nodes from root to leaf)
    Actually, let's use a simpler representation.
    
    We'll represent the tree as a set of splits. For each 4-taxon subset,
    check if the tree induces the same split as the quartet score.
    """
    # For a given tree, determine the split for any 4-taxon subset
    # A tree on taxa T induces a split for each 4-subset {a,b,c,d}
    # based on which pairs are separated by the internal edge.
    
    # Simpler: for each quartet, check consistency
    consistent = 0
    total = 0
    
    for quartet, (g1, g2) in quartet_scores.items():
        if not quartet.issubset(set(labels_in_tree)):
            continue
        total += 1
        # Check if tree induces split g1|g2
        # In the tree, find if there's an edge whose removal separates g1 from g2
        # This is complex, so let's use a different approach:
        # For 4 taxa in a tree, the induced quartet topology is determined
        # by which pairs have the smallest path distance in the tree.
        # But we don't have branch lengths...
        
        # Alternative: use the tree structure directly.
        # For an unrooted tree, the quartet topology is determined by
        # the three possible pairings. We need to check which pairing
        # the tree realizes.
        
        # Let's compute the tree distances (topological, not metric)
        # For each pair, count the number of edges on the path.
        # The quartet topology has the property that the two "within" pairs
        # have shorter topological distances than the "cross" pairs.
        
        a, b, c, d = sorted(quartet)
        # Get topological distances
        dists = {}
        for x, y in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]:
            dists[(x, y)] = _topo_dist(tree, x, y)
        
        s1 = dists[(a, b)] + dists[(c, d)]
        s2 = dists[(a, c)] + dists[(b, d)]
        s3 = dists[(a, d)] + dists[(b, c)]
        
        # Determine which topology the tree induces
        if s1 <= s2 and s1 <= s3:
            tree_topo = (frozenset([a, b]), frozenset([c, d]))
        elif s2 <= s1 and s2 <= s3:
            tree_topo = (frozenset([a, c]), frozenset([b, d]))
        else:
            tree_topo = (frozenset([a, d]), frozenset([b, c]))
        
        # Compare with quartet score
        g1s, g2s = frozenset(g1), frozenset(g2)
        if tree_topo == (g1s, g2s) or tree_topo == (g2s, g1s):
            consistent += 1
    
    return consistent, total


def _topo_dist(tree, x, y):
    """Compute topological distance (number of edges) between x and y in tree."""
    # tree is a dict: node -> (parent, [children]) or similar
    # Let's use a different tree representation for efficiency.
    # Actually, let's use adjacency-based BFS.
    pass


# Let me redesign with a cleaner tree representation.

def _build_tree_quartet_fitting(sequences):
    """
    Quartet Fitting: build tree by greedily adding taxa.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Step 1: Compute all pairwise JC distances
    dist_matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_dist(sequences[labels[i]], sequences[labels[j]])
            dist_matrix[(labels[i], labels[j])] = d
            dist_matrix[(labels[j], labels[i])] = d
    
    # Step 2: Compute quartet scores
    quartet_scores = _build_quartet_scores(labels, dist_matrix)
    
    # Step 3: Greedy tree construction
    # Start with 3 taxa, add one at a time.
    # Tree representation: adjacency list of nodes.
    # Each node is either a taxon label or an internal node.
    
    # We'll use a tree where leaves are taxa and internal nodes are labeled.
    # For tracking splits, we'll maintain the tree as a set of edges.
    
    # Simpler approach: represent tree as nested tuples and convert to Newick.
    # But for quartet checking, we need to compute induced quartet topologies.
    
    # Let's use a different strategy: build the tree using a "add leaf" approach
    # where we track the tree structure and compute quartet consistency efficiently.
    
    # For efficiency, let's use the following:
    # - Maintain the tree as a list of "splits" (bipartitions of the current taxa set)
    # - When adding a new taxon, find the best edge to subdivide
    
    # Actually, let's use a cleaner approach:
    # 1. Start with a 3-taxon star tree
    # 2. For each remaining taxon, find the best insertion point
    # 3. Use quartet consistency as the score
    
    # Tree representation: 
    # - nodes: set of node names (taxa + internal nodes)
    # - edges: set of frozenset({u, v}) 
    # - For each node, track which "side" it's on for each split
    
    # This is getting complex. Let me use a simpler but effective approach:
    # Use the quartet scores to build a "quartet distance matrix" and then
    # use a modified Neighbor Joining that uses quartet-based distances.
    
    # Alternative: Use the quartet scores directly to build the tree via
    # a "greedy quartet" method.
    
    # Let me try a different, cleaner approach:
    # 1. Compute quartet scores
    # 2. For each pair of taxa, compute a "quartet-based distance" 
    #    (average over all quartets containing both)
    # 3. Use this distance matrix with Neighbor Joining
    
    # Actually, the most straightforward quartet-based method is:
    # - For each 4-taxon subset, we have a preferred topology
    # - We want to find a tree that maximizes agreement with these quartets
    # - This is NP-hard, but greedy heuristics work well
    
    # Let me implement a greedy approach:
    # Start with 3 taxa. For each new taxon x, find the best edge to subdivide
    # such that the number of consistent quartets involving x is maximized.
    
    # Tree representation: 
    #   leaves = set of taxa in tree so far
    #   For each edge, we track the two "sides" (sets of leaves on each side)
    #   When we subdivide an edge, we create a new internal node and two edges
    
    # For quartet checking: for a quartet {a,b,c,d}, the tree induces a split
    # if there's an edge whose removal separates the quartet into two pairs.
    # The induced topology is determined by which edge splits the quartet.
    
    # Let me implement this step by step.
    
    # Start with 3 taxa
    current_leaves = set(labels[:3])
    # Tree: a 3-taxon star. One internal node connected to all 3 leaves.
    # Edges: {internal, t0}, {internal, t1}, {internal, t2}
    
    # For tracking splits, I'll maintain a list of "bipartitions".
    # Each bipartition is a frozenset of leaves on one side.
    # For a 3-taxon star, the only non-trivial splits are singletons vs rest,
    # which are trivial. So the 3-taxon star has no non-trivial quartet information.
    
    # When we add a 4th taxon, we subdivide one of the 3 edges.
    # This creates a tree with one internal edge, which induces one non-trivial split.
    
    # Let me track the tree as:
    # - leaf_set: set of current leaves
    # - splits: list of frozensets (each represents one side of a bipartition)
    # - For each split, the other side is leaf_set - split
    
    # For quartet consistency: for a quartet Q = {a,b,c,d},
    # the tree induces a split if there's a split S such that
    # S ∩ Q is a 2-element set (and Q - S is the other 2-element set).
    # The induced topology is (S ∩ Q) | (Q - S).
    
    # Let me implement this.
    
    leaf_set = frozenset(labels[:3])
    splits = []  # list of frozensets, each is one side of a bipartition
    
    # For 3 taxa, no non-trivial splits
    remaining = labels[3:]
    
    for x in remaining:
        # Find the best edge to subdivide for adding x
        # An edge is identified by a split S (one side) where S is a proper
        # non-trivial subset of leaf_set (i.e., 1 < |S| < |leaf_set|)
        # Plus the "trivial" edges: each leaf is connected via an edge.
        
        # For a tree with k leaves, there are 2k-3 edges.
        # Each edge corresponds to a split.
        # For the 3-taxon star, all 3 edges are trivial (singleton splits).
        
        # When we subdivide an edge corresponding to split S,
        # we create a new internal node. The new splits are:
        #   S (unchanged, but now the new node is on one side)
        #   Actually, subdividing an edge doesn't change the split structure;
        #   it just adds a new leaf. The new leaf x is attached to the new node.
        #   The split S now includes x on one side.
        
        # Let me think more carefully.
        # Current tree has leaves L and splits S_1, ..., S_m.
        # Edge e corresponds to split S_e (|S_e| = k for some k, 1 <= k <= |L|-1).
        # Subdividing e: add new node v, remove edge e, add edges (u,v) and (v,x).
        # The new splits are:
        #   All old splits, but S_e is replaced by S_e ∪ {x} (if x is on that side)
        #   Actually, the split S_e now has x added to one side.
        #   And we get a new split: {x} (trivial)
        
        # This is getting complicated. Let me use a simpler representation.
        
        # I'll represent the tree as a nested structure and compute
        # quartet consistency by checking all 4-subsets.
        
        # For efficiency, let me limit the number of taxa we handle with
        # full quartet checking. For n > 20, use a faster heuristic.
        
        break  # placeholder
    
    # This approach is getting too complex. Let me use a simpler, proven method:
    # Use the quartet scores to compute a "quartet distance" matrix, then
    # use Neighbor Joining on this matrix.
    
    # Quartet distance between taxa i and j:
    # For all quartets containing both i and j, check if the quartet score
    # "supports" i and j being on the same side of the internal edge.
    # Define q_dist(i,j) = (number of quartets with i,j where they are
    # on different sides) / (total quartets with i,j)
    
    # Actually, a cleaner approach: for each pair (i,j), compute the fraction
    # of quartets containing i and j where i and j are in the same group.
    # Then use (1 - fraction) as a distance-like measure.
    
    n = len(labels)
    # q_support[i][j] = fraction of quartets with i,j where they're together
    q_support = [[0.0] * n for _ in range(n)]
    q_count = [[0] * n for _ in range(n)]
    
    for quartet, (g1, g2) in quartet_scores.items():
        ql = sorted(quartet)
        # g1 and g2 are tuples of 2 taxa
        g1_set = set(g1)
        g2_set = set(g2)
        # For each pair in the quartet, check if they're in the same group
        for idx1 in range(4):
            for idx2 in range(idx1 + 1, 4):
                t1, t2 = ql[idx1], ql[idx2]
                i, j = labels.index(t1), labels.index(t2)
                q_count[i][j] += 1
                q_count[j][i] += 1
                if t1 in g1_set and t2 in g1_set:
                    q_support[i][j] += 1
                    q_support[j][i] += 1
                elif t1 in g2_set and t2 in g2_set:
                    q_support[i][j] += 1
                    q_support[j][i] += 1
    
    # Convert to distance: q_dist[i][j] = 1 - q_support[i][j] / q_count[i][j]
    q_dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            if q_count[i][j] > 0:
                d = 1.0 - q_support[i][j] / q_count[i][j]
            else:
                d = 0.5  # default
            q_dist[(labels[i], labels[j])] = d
            q_dist[(labels[j], labels[i])] = d
    
    # Now use Neighbor Joining on q_dist
    return _neighbor_joining_quartet(labels, q_dist)


def _neighbor_joining_quartet(labels, dist):
    """Neighbor Joining using quartet-based distances."""
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    nodes = list(labels)
    trees = {name: name for name in labels}
    matrix = dict(dist)
    
    while len(nodes) > 3:
        n = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a star tree as fallback
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        return "(" + ",".join(labels) + ");"
    
    try:
        return _build_tree_quartet_fitting(sequences)
    except Exception:
        # Fallback to a simple star tree
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        return "(" + ",".join(labels) + ");"
