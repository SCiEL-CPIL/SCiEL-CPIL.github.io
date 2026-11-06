"""Phylogenetic topology recovery: quartet-counting for small n, NJ for large n."""
import math
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _pairwise_distances(sequences, labels):
    """Compute JC-corrected pairwise distance matrix using numpy."""
    import numpy as np
    n = len(labels)
    # Build alignment matrix: n x L over {0,1,2,3}
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    # Use numpy for speed
    mat = np.zeros((n, L), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Convert string to array of uint8
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        # Map A->0, C->1, G->2, T->3
        # A=65, C=67, G=71, T=84
        # We need a mapping. Let's do it manually with numpy where
        mat[i] = np.where(arr == 65, 0, np.where(arr == 67, 1, np.where(arr == 71, 2, 3)))
    
    # Pairwise differences
    # For each pair, count mismatches
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            p = np.mean(mat[i] != mat[j])
            d = _jc_distance(p)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_tree(dist, labels):
    """Neighbour joining algorithm. Returns Newick string."""
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Work with a mutable distance matrix
    D = [list(row) for row in dist]
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(nodes) > 3:
        m = len(nodes)
        # Compute r_i = sum of distances from i to all other nodes
        r = [0.0] * m
        for idx, i in enumerate(nodes):
            for jdx, j in enumerate(nodes):
                if j != i:
                    r[idx] += D[i][j]
        
        # Find pair (a,b) minimizing Q = (m-2)*D[a][b] - r[a] - r[b]
        best_val = float('inf')
        best_a = best_b = -1
        for i in range(m):
            for j in range(i+1, m):
                a, b = nodes[i], nodes[j]
                q = (m - 2) * D[a][b] - r[i] - r[j]
                if q < best_val:
                    best_val = q
                    best_a, best_b = a, b
                    best_i, best_j = i, j
        
        # Join best_a and best_b
        # Compute branch lengths
        d_ab = D[best_a][best_b]
        delta = (r[best_i] - r[best_j]) / (2 * (m - 2))
        len_a = 0.5 * d_ab + delta
        len_b = 0.5 * d_ab - delta
        if len_a < 0: len_a = 0
        if len_b < 0: len_b = 0
        
        joined = f"({trees[best_a]},{trees[best_b]})"
        new_id = next_id
        next_id += 1
        trees[new_id] = joined
        
        # Compute distances from new node to remaining nodes
        remaining = [x for x in nodes if x not in (best_a, best_b)]
        for x in remaining:
            D[new_id][x] = D[x][new_id] = 0.5 * (D[best_a][x] + D[best_b][x] - d_ab)
        
        # Update nodes list
        nodes = remaining + [new_id]
    
    # Final 3-node tree
    return "(" + ",".join(trees[x] for x in nodes) + ")"


def _quartet_counting(sequences, labels):
    """
    For small n, use quartet counting to find the best topology.
    Enumerate all unrooted binary topologies for n <= 7, score each by
    the number of quartets it agrees with the data.
    """
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Precompute pairwise identity counts for all pairs
    # For quartet (a,b,c,d): the three splits are ab|cd, ac|bd, ad|bc
    # For split ab|cd: count sites where (a==b) XOR (c==d) vs (a==c) XOR (b==d) etc.
    # Actually, for quartet scoring: 
    #   For split ab|cd: score = count of sites where a,b agree and c,d agree but a,c disagree (or a,d disagree)
    #   More precisely, the likelihood ratio for the quartet topology.
    # 
    # Simpler: for each quartet, compute the three possible topologies and pick the one
    # with the highest sum of "supporting" sites.
    #
    # For a quartet with leaves a,b,c,d and topology ((a,b),(c,d)):
    #   A site supports this if the pattern is: a==b and c==d and a!=c (or a!=d, which is equivalent)
    #   Actually, the standard quartet score: for topology ((a,b),(c,d)):
    #     score = (# sites with pattern 0011 or 1100 where positions are a,b,c,d in order)
    #   where 0011 means a==b, c==d, a!=c.
    #
    # Let's use the standard approach: for each quartet, count the 6 possible patterns
    # and assign to the 3 topologies.
    
    # Precompute for each pair of leaves, the number of positions where they are equal
    # Actually, let's just compute per-quartet scores directly but efficiently.
    
    # For efficiency, precompute for each leaf a bitmask or array of its sequence values.
    # Convert sequences to arrays of 0,1,2,3
    import numpy as np
    seq_arrays = []
    for lab in labels:
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        # Map to 0,1,2,3
        mapped = np.empty(L, dtype=np.uint8)
        mapped[arr == 65] = 0
        mapped[arr == 67] = 1
        mapped[arr == 71] = 2
        mapped[arr == 84] = 3
        seq_arrays.append(mapped)
    
    # For each quartet, compute the three topology scores
    # Topology 1: ((a,b),(c,d)) -> patterns where a==b, c==d, a!=c
    # Topology 2: ((a,c),(b,d)) -> patterns where a==c, b==d, a!=b
    # Topology 3: ((a,d),(b,c)) -> patterns where a==d, b==c, a!=b
    
    # We'll compute a score for each of the 3 topologies per quartet, then sum over all quartets
    # to get a score for each candidate tree topology.
    
    # But enumerating all tree topologies for n=7: C(2n-5, n-1)/(n-1)! = C(9,6)/720 = 84/720... 
    # Actually, number of unrooted binary trees on n labeled leaves = (2n-5)!/(2^(n-1)*(n-1)!)
    # For n=5: 3, n=6: 15, n=7: 105.
    # For n=8: 945. n=9: 10395. n=10: 135135.
    # So for n<=8, we can enumerate. For n=8, 945 topologies, each requires checking all quartets.
    # Number of quartets for n=8: C(8,4) = 70. So 945*70 = 66150 quartet evaluations.
    # Each quartet evaluation involves L site checks. If L=2000, that's 132M operations. Might be slow.
    # Let's limit to n<=7 for exact enumeration, and use NJ for n>=8.
    
    if n > 7:
        return None  # Fall back to NJ
    
    # Enumerate all unrooted binary tree topologies
    # Represent a tree by its set of internal edges (splits)
    # For n leaves, a binary unrooted tree has n-3 internal edges, each splitting leaves into two groups.
    
    # Generate all possible topologies
    def generate_topologies(n):
        """Generate all unrooted binary tree topologies on n labeled leaves 0..n-1."""
        if n == 3:
            yield frozenset([])  # No internal edges
        elif n == 4:
            # 3 topologies: 12|34, 13|24, 14|23
            yield frozenset([frozenset({0,1}), frozenset({2,3})])
            yield frozenset([frozenset({0,2}), frozenset({1,3})])
            yield frozenset([frozenset({0,3}), frozenset({1,2})])
        else:
            # Recursive: pick a split of {0..n-1} into two parts of size k and n-k
            # where 2 <= k <= n-2, and the split contains leaf 0 in the first part.
            # Then recursively generate topologies for each part.
            # But we need to avoid duplicates. Standard approach:
            # Fix leaf 0. A split is determined by the subset containing 0, with size 2 to n-2.
            # To avoid duplicates, require that the subset containing 0 has size <= n/2 (or use canonical form).
            
            # Actually, let's use a different approach: generate all set partitions into n-3 internal edges.
            # This is complex. Let me use a known enumeration method.
            
            # For n=5: 3 topologies
            # For n=6: 15 topologies
            # For n=7: 105 topologies
            
            # I'll use a recursive approach based on "caterpillar" decomposition.
            # Alternative: use the fact that any binary tree can be represented by its Newick string.
            # Generate all Newick strings for n leaves.
            
            # Let me just hardcode for n=5,6,7 using known counts and generate via recursion.
            
            # Recursive generation:
            # A binary tree on n leaves can be formed by joining two subtrees of sizes i and n-i
            # at an internal node, where 2 <= i <= n-2.
            # To avoid duplicates, we fix that the subtree containing leaf 0 has size i where 2 <= i <= n//2.
            
            def gen(n_local, leaves):
                """Generate all unrooted binary trees on the given leaves."""
                n_local = len(leaves)
                if n_local == 3:
                    # A 3-leaf tree is just a star (no internal edges in the unrooted sense)
                    # Represent as empty set of splits
                    yield []
                elif n_local == 4:
                    # 3 topologies
                    a, b, c, d = leaves[0], leaves[1], leaves[2], leaves[3]
                    yield [[(a,b),(c,d)]]
                    yield [[(a,c),(b,d)]]
                    yield [[(a,d),(b,c)]]
                else:
                    # Split into two parts
                    for i in range(2, n_local // 2 + 1):
                        # Choose i leaves for the first part, must include leaves[0]
                        rest = leaves[1:]
                        for subset in itertools.combinations(rest, i - 1):
                            part1 = frozenset([leaves[0]] + list(subset))
                            part2 = frozenset(leaves) - part1
                            # Generate subtrees for part1 and part2
                            leaves1 = sorted(part1)
                            leaves2 = sorted(part2)
                            for sub1 in gen(len(leaves1), leaves1):
                                for sub2 in gen(len(leaves2), leaves2):
                                    # Combine: the new internal edge connects the two subtrees
                                    # The split is part1 | part2
                                    yield sub1 + sub2 + [[tuple(sorted(part1)), tuple(sorted(part2))]]
            
            yield from gen(n, list(range(n)))
    
    # For each topology, compute the total quartet score
    # A topology "supports" a quartet if the induced quartet topology matches.
    # For a given tree topology (set of splits), and a quartet (a,b,c,d),
    # the induced topology is determined by which pairs are separated by internal edges.
    
    # Precompute for each quartet the three possible scores
    quartet_scores = {}
    quartet_list = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Compute scores for the 3 topologies
        # Topology ((a,b),(c,d)): count sites where a==b and c==d and a!=c
        # Topology ((a,c),(b,d)): count sites where a==c and b==d and a!=b  
        # Topology ((a,d),(b,c)): count sites where a==d and b==c and a!=b
        
        sa = seq_arrays[a]
        sb = seq_arrays[b]
        sc = seq_arrays[c]
        sd = seq_arrays[d]
        
        # For efficiency, compute all three in one pass
        score_ab_cd = 0  # ((a,b),(c,d))
        score_ac_bd = 0  # ((a,c),(b,d))
        score_ad_bc = 0  # ((a,d),(b,c))
        
        for pos in range(L):
            va, vb, vc, vd = sa[pos], sb[pos], sc[pos], sd[pos]
            # ((a,b),(c,d)): a==b, c==d, a!=c
            if va == vb and vc == vd and va != vc:
                score_ab_cd += 1
            # ((a,c),(b,d)): a==c, b==d, a!=b
            if va == vc and vb == vd and va != vb:
                score_ac_bd += 1
            # ((a,d),(b,c)): a==d, b==c, a!=b
            if va == vd and vb == vc and va != vb:
                score_ad_bc += 1
        
        quartet_scores[quad] = (score_ab_cd, score_ac_bd, score_ad_bc)
        quartet_list.append(quad)
    
    # Now for each topology, sum up the quartet scores it supports
    best_score = -1
    best_topo = None
    
    for topo in generate_topologies(n):
        # topo is a list of splits, each split is a tuple of two tuples (part1, part2)
        # For each quartet, determine which of the 3 topologies is induced
        total = 0
        for quad in quartet_list:
            a, b, c, d = quad
            # Determine the induced quartet topology
            # For each pair, check if they are separated by an internal edge
            # Two leaves are "close" (same side of every internal edge) if no split separates them
            # The induced topology groups leaves that are not separated.
            
            # For the quartet, the three possible groupings are:
            # ab|cd: a and b are not separated, c and d are not separated, but a and c are separated
            # ac|bd: a and c not separated, b and d not separated
            # ad|bc: a and d not separated, b and c not separated
            
            # Check which grouping is consistent with the tree
            def separated(x, y, splits):
                """Check if leaves x and y are separated by any internal edge."""
                for split in splits:
                    part1, part2 = split
                    if (x in part1 and y in part2) or (x in part2 and y in part1):
                        return True
                return False
            
            # For ab|cd to be the induced topology:
            # a,b not separated; c,d not separated; a,c separated (and a,d separated)
            if not separated(a, b, topo) and not separated(c, d, topo) and separated(a, c, topo):
                total += quartet_scores[quad][0]
            elif not separated(a, c, topo) and not separated(b, d, topo) and separated(a, b, topo):
                total += quartet_scores[quad][1]
            elif not separated(a, d, topo) and not separated(b, c, topo) and separated(a, b, topo):
                total += quartet_scores[quad][2]
            # If none match (shouldn't happen for a valid binary tree), skip
        
        if total > best_score:
            best_score = total
            best_topo = topo
    
    if best_topo is None:
        return None
    
    # Convert the best topology to Newick
    # best_topo is a list of splits. We need to reconstruct the tree.
    # This is complex. Instead, let's directly build the Newick from the splits.
    
    # Actually, for the final answer, we just need to return a Newick string.
    # Let me reconstruct the tree from the splits.
    
    # For n=4, it's straightforward. For larger n, it's more complex.
    # Given time constraints, let me just handle n=4 explicitly and use NJ for n>=5.
    
    if n == 4:
        # best_topo has one split
        part1, part2 = best_topo[0]
        p1 = sorted(part1)
        p2 = sorted(part2)
        return f"({labels[p1[0]]},{labels[p1[1]]}),{labels[p2[0]]},{labels[p2[1]]})"
    
    # For n >= 5, the reconstruction from splits to Newick is non-trivial.
    # Let's fall back to NJ for n >= 5 to keep things simple and fast.
    return None


def reconstruct(sequences):
    """Reconstruct phylogenetic tree from DNA sequences."""
    start_time = time.time()
    DEADLINE = 260.0  # Leave 10s margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n < 2:
        return ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Try quartet counting for small n
    if n <= 4:
        result = _quartet_counting(sequences, labels)
        if result is not None:
            return result
    
    # Compute pairwise distances
    dist = _pairwise_distances(sequences, labels)
    
    # Use NJ
    result = _nj_tree(dist, labels)
    return result

# EVOLVE-BLOCK-END
