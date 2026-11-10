import math
import time
import numpy as np
from itertools import combinations

def _quartet_count(sequences, labels, i, j, k, l):
    """
    Count the frequency of each of the three possible quartet topologies
    for taxa i, j, k, l using site-by-site analysis.
    
    Returns a dict mapping topology string to count.
    Topologies: "ij|kl", "ik|jl", "il|jk"
    """
    seq_i = sequences[labels[i]]
    seq_j = sequences[labels[j]]
    seq_k = sequences[labels[k]]
    seq_l = sequences[labels[l]]
    L = len(seq_i)
    
    counts = {"ij|kl": 0, "ik|jl": 0, "il|jk": 0}
    
    for pos in range(L):
        a = seq_i[pos]
        b = seq_j[pos]
        c = seq_k[pos]
        d = seq_l[pos]
        
        # Determine which quartet topology is supported at this site
        # A site supports a topology if the pattern of mutations is consistent
        # with that topology under the JC model.
        
        # For simplicity, use a heuristic: count the number of pairwise differences
        # and assign to the topology with the smallest sum of two "cross" distances.
        # Actually, let's use a more principled approach:
        # Under JC, the probability of observing a pattern depends on the topology.
        # For a given site, we can compute the likelihood under each topology
        # using the JC model, but that's complex.
        
        # Simpler approach: Use the "quartet score" based on distances.
        # For a site, define:
        # d_ij = 1 if seq_i != seq_j else 0
        # etc.
        
        dij = 1 if a != b else 0
        dkl = 1 if c != d else 0
        dik = 1 if a != c else 0
        djl = 1 if b != d else 0
        dil = 1 if a != d else 0
        djk = 1 if b != c else 0
        
        # The topology "ij|kl" is supported if dij + dkl < dik + djl and dij + dkl < dil + djk
        # But this is not quite right for a single site.
        
        # Better: For a single site, the "signal" is weak. Let's use a different approach.
        # Count the number of sites where the pattern matches each topology.
        
        # Pattern analysis:
        # If all four are the same: no signal
        # If three are the same and one differs: the differing taxon is likely on a separate branch
        # If two pairs are the same: supports the topology where the pairs are on separate branches
        # If all four are different: complex
        
        if a == b == c == d:
            continue  # No signal
        
        # Count how many are in each "group"
        from collections import Counter
        chars = [a, b, c, d]
        char_counts = Counter(chars)
        
        if len(char_counts) == 2:
            # Two characters, each appearing twice
            # Find which two taxa share each character
            if a == b and c == d and a != c:
                counts["ij|kl"] += 1
            elif a == c and b == d and a != b:
                counts["ik|jl"] += 1
            elif a == d and b == c and a != b:
                counts["il|jk"] += 1
        elif len(char_counts) == 3:
            # Three characters, one appearing twice
            # The taxon that is different from the majority is likely on a separate branch
            # Find which character appears twice
            for ch, cnt in char_counts.items():
                if cnt == 2:
                    # The two taxa with this character form a clade
                    # The other two are on separate branches
                    if a == b and a == ch:
                        # a and b share the majority character
                        # c and d are different
                        # This supports ij|kl if c and d are on the same side
                        # But c != d, so they're on different sides
                        # This is ambiguous. Let's skip for now or assign to the topology
                        # that groups the two with the same character.
                        counts["ij|kl"] += 1
                    elif a == c and a == ch:
                        counts["ik|jl"] += 1
                    elif a == d and a == ch:
                        counts["il|jk"] += 1
                    elif b == c and b == ch:
                        # b and c share the majority character
                        # a and d are different
                        # This supports the topology where b and c are together
                        # Which is ik|jl if i and k are together? No.
                        # Let's think: if b and c share the majority, then the split is
                        # {b,c} vs {a,d}
                        # This corresponds to "il|jk" if i and l are on one side and j and k on the other?
                        # No, let's just assign based on which pair shares the majority.
                        # If b and c share, the split is bc|ad, which is "il|jk"
                        counts["il|jk"] += 1
                    elif b == d and b == ch:
                        # b and d share the majority
                        # Split is bd|ac, which is "ik|jl"
                        counts["ik|jl"] += 1
                    elif c == d and c == ch:
                        # c and d share the majority
                        # Split is cd|ab, which is "ij|kl"
                        counts["ij|kl"] += 1
                    break
        elif len(char_counts) == 4:
            # All four different
            # No clear signal, skip or assign randomly
            continue
    
    return counts

def _build_tree_from_quartets(quartet_scores, n, labels):
    """
    Build a tree from quartet scores using a greedy approach.
    This is a simplified version of quartet fitting.
    """
    # Start with a star topology
    # Then iteratively add quartets that are well-supported
    
    # For simplicity, let's use a different approach:
    # Compute the distance matrix from the quartet scores
    # and use NJ.
    
    # Actually, let's just return a star topology for now if quartet fitting is too complex.
    # Or use a simple heuristic.
    
    # For the time being, let's compute a distance matrix from the sequences
    # and use NJ, but with the quartet information to break ties.
    
    # This is getting complex. Let's implement a simpler version:
    # Use the quartet scores to build a consensus tree.
    
    # For now, return a star topology as a placeholder.
    # This is not ideal, but it's a valid tree.
    
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    elif n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Star topology
    star = f"({','.join(labels)})"
    return star

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    idx_map = {label: i for i, label in enumerate(labels)}
    seq_arrays = []
    for label in labels:
        seq = sequences[label]
        arr = np.zeros(len(seq), dtype=np.int8)
        for i, c in enumerate(seq):
            if c == 'A': arr[i] = 0
            elif c == 'C': arr[i] = 1
            elif c == 'G': arr[i] = 2
            elif c == 'T': arr[i] = 3
        seq_arrays.append(arr)
    
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seq_arrays[i] != seq_arrays[j])
            p = diff / len(seq_arrays[0])
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat

def _nj_tree(dist_mat, labels):
    n = len(labels)
    active = list(range(n))
    internal_nodes = {}
    next_id = n
    max_nodes = 2 * n
    dist = [[0.0 for _ in range(max_nodes)] for _ in range(max_nodes)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = dist_mat[i, j]
    
    while len(active) > 3:
        k = len(active)
        r = [0.0] * k
        for i, node in enumerate(active):
            for j, other in enumerate(active):
                if i != j:
                    r[i] += dist[node][other]
        
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(k):
            for j in range(i + 1, k):
                node_i = active[i]
                node_j = active[j]
                q = (k - 2) * dist[node_i][node_j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        node_a = active[best_i]
        node_b = active[best_j]
        
        delta = (r[best_i] - r[best_j]) / (k - 2)
        len_a = 0.5 * (dist[node_a][node_b] + delta)
        len_b = 0.5 * (dist[node_a][node_b] - delta)
        
        new_node = next_id
        next_id += 1
        
        internal_nodes[new_node] = (node_a, node_b, len_a, len_b)
        
        for x in active:
            if x != node_a and x != node_b:
                dist[new_node][x] = 0.5 * (dist[node_a][x] + dist[node_b][x] - dist[node_a][node_b])
                dist[x][new_node] = dist[new_node][x]
        
        active = [x for x in active if x != node_a and x != node_b]
        active.append(new_node)
    
    def get_newick(node):
        if node < n:
            return labels[node]
        else:
            child_a, child_b, len_a, len_b = internal_nodes[node]
            return f"({get_newick(child_a)}:{len_a:.4f},{get_newick(child_b)}:{len_b:.4f})"
    
    if len(active) == 3:
        u, v, w = active
        len_u = 0.5 * (dist[u][v] + dist[u][w] - dist[v][w])
        len_v = 0.5 * (dist[u][v] + dist[v][w] - dist[u][w])
        len_w = 0.5 * (dist[u][w] + dist[v][w] - dist[u][v])
        
        new_node_1 = next_id
        next_id += 1
        internal_nodes[new_node_1] = (u, v, len_u, len_v)
        
        new_node_2 = next_id
        next_id += 1
        internal_nodes[new_node_2] = (new_node_1, w, 0.0, len_w)
        
        return get_newick(new_node_2)
    else:
        return ",".join(labels)

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # Use NJ for the initial tree
    newick = _nj_tree(dist_mat, labels)
    
    # Time check
    if time.time() - start_time > time_limit:
        return newick
    
    return newick
