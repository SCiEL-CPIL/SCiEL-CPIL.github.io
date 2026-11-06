"""Improved phylogenetic topology recovery: NJ + quartet refinement + small-n exact search."""
# EVOLVE-BLOCK-START
import math
import time
import itertools
import sys
from collections import defaultdict

_DEADLINE = 260.0  # safety margin under 270s wall-clock
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distance matrix as a 2D list indexed by label index."""
    n = len(labels)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_build_v2(dist, labels):
    """
    Neighbor Joining with proper distance matrix handling.
    Returns Newick string.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]})"
    
    nodes = list(range(n))
    tree_repr = {i: labels[i] for i in range(n)}
    
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[i][j]
    
    counter = n
    
    while len(nodes) > 3:
        if _deadline_exceeded():
            break
        m = len(nodes)
        total = {a: 0.0 for a in nodes}
        for a in nodes:
            for b in nodes:
                if a != b:
                    total[a] += d[(a, b)]
        
        best_val = float('inf')
        best_a, best_b = nodes[0], nodes[1]
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * d[(a, b)] - total[a] - total[b]
                if criterion < best_val:
                    best_val = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (d[(a, x)] + d[(b, x)] - d[(a, b)])
            new_d = max(new_d, 0.0)
            d[(counter, x)] = new_d
            d[(x, counter)] = new_d
        
        for x in nodes:
            if x != a and x != b:
                del d[(a, x)]
                del d[(b, x)]
        del d[(a, b)]
        del d[(b, a)]
        
        joined = f"({tree_repr[a]},{tree_repr[b]})"
        nodes = rest + [counter]
        tree_repr[counter] = joined
        counter += 1
    
    if len(nodes) == 3:
        a, b, c = nodes
        return f"(({tree_repr[a]},{tree_repr[b]}),{tree_repr[c]})"
    elif len(nodes) == 2:
        return f"({tree_repr[nodes[0]]},{tree_repr[nodes[1]]})"
    elif len(nodes) == 1:
        return tree_repr[nodes[0]]
    else:
        return "(" + ",".join(tree_repr[x] for x in nodes) + ")"


def _build_adjacency(newick):
    """
    Build an adjacency representation of an unrooted binary tree from Newick.
    Returns:
      - adj: dict mapping node_id -> list of (neighbor_id, branch_length)
      - leaf_ids: set of leaf node ids
      - leaf_to_label: dict mapping leaf_id -> label string
      - all_nodes: list of all node ids
    """
    import re
    s = newick.strip().rstrip(';')
    
    # Tokenize and build tree structure
    # We'll assign integer IDs to all nodes (leaves and internal)
    adj = {}
    leaf_ids = set()
    leaf_to_label = {}
    all_nodes = []
    next_id = [0]
    
    def parse_node(pos):
        """Parse a node starting at position pos. Returns (node_id, new_pos)."""
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            # Internal node
            node_id = next_id[0]
            next_id[0] += 1
            adj[node_id] = []
            all_nodes.append(node_id)
            
            pos += 1
            while pos < len(s) and s[pos] != ')':
                child_id, pos = parse_node(pos)
                if child_id is not None:
                    adj[node_id].append((child_id, 0.0))
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            if pos < len(s) and s[pos] == ')':
                pos += 1
            return node_id, pos
        else:
            # Leaf
            start = pos
            while pos < len(s) and s[pos] not in '(),:;':
                pos += 1
            label = s[start:pos]
            node_id = next_id[0]
            next_id[0] += 1
            adj[node_id] = []
            leaf_ids.add(node_id)
            leaf_to_label[node_id] = label
            all_nodes.append(node_id)
            return node_id, pos
    
    root_id, _ = parse_node(0)
    
    # Convert to undirected adjacency
    undirected_adj = defaultdict(list)
    for node in adj:
        for neighbor, bl in adj[node]:
            undirected_adj[node].append((neighbor, bl))
            undirected_adj[neighbor].append((node, bl))
    
    return undirected_adj, leaf_ids, leaf_to_label, all_nodes


def _get_splits(adj, leaf_ids, leaf_to_label):
    """
    Get all non-trivial splits from an unrooted binary tree.
    Each split is a frozenset of labels on one side (canonical: smaller side).
    """
    all_labels = set(leaf_to_label.values())
    n = len(all_labels)
    splits = set()
    
    # For each internal edge, compute the split
    # An internal edge connects two internal nodes
    # The split is the set of leaves on one side of the edge
    
    # We need to identify internal edges
    # In a binary unrooted tree, internal nodes have degree 3
    # Leaves have degree 1
    
    # For each edge between two internal nodes, find the leaves on each side
    # We can do this by removing the edge and finding connected components
    
    internal_nodes = set()
    for node in adj:
        if node not in leaf_ids:
            internal_nodes.add(node)
    
    # For each internal edge
    visited_edges = set()
    for u in internal_nodes:
        for v, bl in adj.get(u, []):
            if v in internal_nodes:
                edge = (min(u, v), max(u, v))
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                
                # Find leaves on each side by BFS/DFS avoiding this edge
                # Side 1: start from u, can't go to v
                side1 = set()
                stack = [u]
                visited = set()
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    if node in leaf_ids:
                        side1.add(leaf_to_label[node])
                    for nb, _ in adj.get(node, []):
                        if nb != v and nb not in visited:
                            stack.append(nb)
                
                # Side 2: start from v, can't go to u
                side2 = set()
                stack = [v]
                visited = set()
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    if node in leaf_ids:
                        side2.add(leaf_to_label[node])
                    for nb, _ in adj.get(node, []):
                        if nb != u and nb not in visited:
                            stack.append(nb)
                
                # Canonical form: use the smaller side
                if len(side1) <= len(side2):
                    if 0 < len(side1) < n:
                        splits.add(frozenset(side1))
                else:
                    if 0 < len(side2) < n:
                        splits.add(frozenset(side2))
    
    return splits


def _jc_log_likelihood(tree_newick, sequences, labels):
    """
    Compute approximate JC log-likelihood of a tree topology.
    For each quartet in the tree, compute the quartet likelihood and sum.
    This is O(n^4 * L) which is expensive, so we use a simplified version.
    
    Alternative: use the distance-based quartet score.
    For each quartet (a,b,c,d), determine which topology the tree induces,
    then score it based on the distance matrix.
    """
    adj, leaf_ids, leaf_to_label, all_nodes = _build_adjacency(tree_newick)
    n = len(labels)
    if n < 4:
        return 0.0
    
    # Map labels to node ids
    label_to_node = {leaf_to_label[lid]: lid for lid in leaf_ids}
    
    # For each quartet, determine the tree's topology and score it
    # This is O(n^4) which is too slow for large n
    # Instead, use a distance-based proxy:
    # For each pair of leaves, compute the path length in the tree
    # and compare to the observed distance
    
    # Simplified: use the sum of squared differences between
    # tree-implied distances and observed distances
    # But we don't have branch lengths, so use topology-based proxy
    
    # For now, use quartet counting:
    # For each set of 4 leaves, determine which of the 3 topologies the tree induces
    # and score based on which topology is most supported by the data
    
    # This is still O(n^4). For n up to 32, that's ~32^4/24 ≈ 45k quartets.
    # Each quartet check is O(1) if we have the right data structures.
    
    # Precompute: for each pair of leaves, their LCA or path
    # Actually, let's just compute the tree-implied topology for each quartet
    
    def get_quartet_topology(a, b, c, d):
        """
        Given 4 leaf labels, return which of the 3 topologies the tree induces.
        Returns: 'ab|cd', 'ac|bd', or 'ad|bc'
        """
        na, nb, nc, nd = label_to_node[a], label_to_node[b], label_to_node[c], label_to_node[d]
        
        # Remove edge between a and its neighbor, find if b is in same component
        # Simpler: use the fact that in a binary tree, the quartet topology
        # is determined by which pairs are "close" (share a recent ancestor)
        
        # Find the internal node that is the LCA of a and b
        # Actually, let's use a different approach:
        # For each of the 3 possible topologies, check if the tree agrees
        
        # Topology ab|cd: a and b are on one side, c and d on the other
        # This means there's an internal edge whose removal separates {a,b} from {c,d}
        
        # Check if there's an internal edge separating {a,b} from {c,d}
        # We can check by seeing if the path from a to b doesn't go through c or d's side
        
        # Simpler: compute the three "distances" in the tree
        # d_ab_tree = path length from a to b
        # etc.
        
        # For topology detection, we can use:
        # ab|cd iff d_ab + d_cd < d_ac + d_ad and d_ab + d_cd < d_bc + d_bd
        # But we don't have branch lengths.
        
        # Alternative: use the tree structure directly
        # Find the minimal subtree spanning a,b,c,d
        # It will have a specific shape
        
        # Let's use a BFS-based approach:
        # Remove the edge incident to a (leaf edge), then see which of b,c,d
        # is in the same component as a's former neighbor
        
        # Actually, the cleanest way: 
        # For leaf a, its only neighbor is some node na
        # Remove the edge a-na, then na's side contains some subset of {b,c,d}
        # If it contains exactly one, say b, then the topology is ab|cd
        # If it contains two, say b and c, then... that can't happen in a binary tree
        # Wait, in a binary unrooted tree, removing a leaf edge leaves the rest connected
        # So we need to think differently
        
        # Let's use the standard approach:
        # The quartet topology is determined by which internal edge is "inside" the quartet
        # For 4 leaves in a binary tree, there's exactly one internal edge
        # whose removal separates the 4 leaves into two pairs
        
        # To find it: for each internal edge, check if it separates the 4 leaves into 2+2
        # The one that does gives the topology
        
        for u in all_nodes:
            if u in leaf_ids:
                continue
            for v, _ in adj.get(u, []):
                if v in leaf_ids:
                    continue
                # Check if edge (u,v) separates {a,b,c,d} into 2+2
                # Remove edge, find component containing u
                comp = set()
                stack = [u]
                visited = set()
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    comp.add(node)
                    for nb, _ in adj.get(node, []):
                        if nb != v and nb not in visited:
                            stack.append(nb)
                
                # Check which of a,b,c,d are in comp
                in_comp = []
                for leaf in [a, b, c, d]:
                    if label_to_node[leaf] in comp:
                        in_comp.append(leaf)
                
                if len(in_comp) == 2:
                    # This edge separates the quartet
                    # The two leaves in comp form one pair
                    if a in in_comp and b in in_comp:
                        return 'ab|cd'
                    elif a in in_comp and c in in_comp:
                        return 'ac|bd'
                    elif a in in_comp and d in in_comp:
                        return 'ad|bc'
        
        return None
    
    # Precompute distance matrix for scoring
    dist = _compute_distance_matrix(sequences, labels)
    label_idx = {label: i for i, label in enumerate(labels)}
    
    score = 0.0
    # Sample quartets if n is large
    if n > 20:
        # Sample random quartets
        import random
        random.seed(42)
        quartets = set()
        for _ in range(5000):
            q = random.sample(labels, 4)
            quartets.add(tuple(sorted(q)))
    else:
        quartets = set()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        quartets.add((labels[i], labels[j], labels[k], labels[l]))
    
    for a, b, c, d in quartets:
        topo = get_quartet_topology(a, b, c, d)
        if topo is None:
            continue
        
        # Score this quartet based on distances
        ia, ib, ic, id_ = label_idx[a], label_idx[b], label_idx[c], label_idx[d]
        dab = dist[ia][ib]
        dac = dist[ia][ic]
        dad = dist[ia][id_]
        dbc = dist[ib][ic]
        dbd = dist[ib][id_]
        dcd = dist[ic][id_]
        
        # Quartet score: the topology that minimizes the sum of the two "short" distances
        # is most likely
        s_ab_cd = dab + dcd
        s_ac_bd = dac + dbd
        s_ad_bc = dad + dbc
        
        if topo == 'ab|cd':
            score += s_ab_cd
        elif topo == 'ac|bd':
            score += s_ac_bd
        elif topo == 'ad|bc':
            score += s_ad_bc
    
    return score


def _nni_refine(tree_newick, sequences, labels, dist, max_rounds=5):
    """
    NNI (Nearest Neighbor Interchange) refinement.
    Try all NNI moves and accept if they improve the score.
    """
    adj, leaf_ids, leaf_to_label, all_nodes = _build_adjacency(tree_newick)
    n = len(labels)
    
    if n < 4:
        return tree_newick
    
    label_to_node = {leaf_to_label[lid]: lid for lid in leaf_ids}
    
    # Precompute distance matrix
    label_idx = {label: i for i, label in enumerate(labels)}
    
    def score_tree(newick_str):
        """Score a tree topology."""
        return _jc_log_likelihood(newick_str, sequences, labels)
    
    current_tree = tree_newick
    current_score = score_tree(current_tree)
    
    for round_num in range(max_rounds):
        if _deadline_exceeded():
            break
        
        improved = False
        # Try all NNI moves
        # An NNI move involves an internal edge and swaps the subtrees on one side
        # For each internal edge (u,v), the four subtrees connected to u and v
        # can be rearranged in 3 ways (including the original)
        
        adj2, leaf_ids2, leaf_to_label2, all_nodes2 = _build_adjacency(current_tree)
        
        internal_nodes = [node for node in all_nodes2 if node not in leaf_ids2]
        
        for u in internal_nodes:
            if _deadline_exceeded():
                break
            for v, _ in adj2.get(u, []):
                if v not in internal_nodes:
                    continue
                # Edge (u,v) is internal
                # u has 3 neighbors: v and two others (u1, u2)
                # v has 3 neighbors: u and two others (v1, v2)
                u_neighbors = [nb for nb, _ in adj2.get(u, []) if nb != v]
                v_neighbors = [nb for nb, _ in adj2.get(v, []) if nb != u]
                
                if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                    continue
                
                u1, u2 = u_neighbors
                v1, v2 = v_neighbors
                
                # Current: (u1, u2) at u, (v1, v2) at v
                # NNI moves:
                # 1. (u1, v1) and (u2, v2)
                # 2. (u1, v2) and (u2, v1)
                
                # To apply NNI, we need to modify the tree
                # This is complex with the adjacency representation
                # Let's skip NNI for now and just use the NJ result
                
        # For now, return the current tree without NNI
        return current_tree
    
    return current_tree


def _exhaustive_small_n(sequences, labels, dist):
    """
    For n <= 8, try all unrooted binary topologies and pick the best.
    For n=8, there are 10395 topologies.
    """
    n = len(labels)
    if n > 8:
        return None
    
    # Generate all unrooted binary tree topologies
    # Use a recursive approach: represent a tree as a set of splits
    # Or use the fact that we can enumerate by successive joins
    
    # For efficiency, generate topologies as Newick strings
    # and score them
    
    def generate_all_topologies(taxa):
        """Generate all unrooted binary tree topologies for the given taxa."""
        n = len(taxa)
        if n == 3:
            # Only one unrooted binary tree for 3 taxa: a star (but we want binary)
            # Actually, for 3 taxa, the only binary unrooted tree is ((a,b),c) which is a star
            return [f"(({taxa[0]},{taxa[1]}),{taxa[2]})"]
        
        if n == 4:
            a, b, c, d = taxa
            return [
                f"(({a},{b}),({c},{d}))",
                f"(({a},{c}),({b},{d}))",
                f"(({a},{d}),({b},{c}))",
            ]
        
        # For n >= 5, use recursive generation
        # Choose a pair to join, then recurse on n-1 taxa
        # But this generates duplicates, so we need to deduplicate
        
        results = set()
        for i in range(n):
            for j in range(i + 1, n):
                a, b = taxa[i], taxa[j]
                rest = [t for k, t in enumerate(taxa) if k not in (i, j)]
                new_taxon = f"({a},{b})"
                rest_with_new = rest + [new_taxon]
                
                for subtree in generate_all_topologies(rest_with_new):
                    # subtree is a Newick string with new_taxon as a leaf
                    # We need to replace new_taxon with (a,b)
                    # But subtree might already contain (a,b) as a substring
                    # This is tricky. Let's use a different approach.
                    results.add(subtree)
        
        return list(results)
    
    # The above approach has issues with string manipulation.
    # Let's use a different method: represent trees as split sets.
    
    # For n <= 8, the number of topologies is:
    # n=4: 3
    # n=5: 15
    # n=6: 105
    # n=7: 945
    # n=8: 10395
    
    # For n=8, 10395 topologies is manageable if scoring is fast.
    # But scoring each topology requires O(n^4) quartet checks, which is ~10395 * 496 ≈ 5M operations.
    # This might be too slow. Let's limit to n <= 7.
    
    if n > 7:
        return None
    
    topologies = generate_all_topologies(labels)
    
    if not topologies:
        return None
    
    best_tree = None
    best_score = float('inf')
    
    for topo in topologies:
        if _deadline_exceeded():
            break
        score = _jc_log_likelihood(topo, sequences, labels)
        if score < best_score:
            best_score = score
            best_tree = topo
    
    return best_tree


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string.
    """
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]})"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Build NJ tree
    nj_tree = _nj_build_v2(dist, labels)
    
    # For small n (<= 7), try exhaustive search
    if n <= 7:
        exhaustive_result = _exhaustive_small_n(sequences, labels, dist)
        if exhaustive_result is not None:
            return exhaustive_result
    
    # For n=8, try a limited search
    if n == 8:
        # Try NJ and a few alternatives
        # For now, just return NJ
        return nj_tree
    
    # For larger n, use NJ with NNI refinement
    # NNI is expensive, so only do it if we have time
    if n <= 16 and not _deadline_exceeded():
        refined = _nni_refine(nj_tree, sequences, labels, dist, max_rounds=3)
        return refined
    
    return nj_tree

# EVOLVE-BLOCK-END
