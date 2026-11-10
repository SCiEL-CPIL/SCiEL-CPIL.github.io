"""Phylogenetic topology recovery: NJ with deadline guard."""
import math
import time

# EVOLVE-BLOCK-START


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distances(sequences, labels):
    """Compute pairwise JC-corrected distances."""
    n = len(labels)
    seq_list = [sequences[lb] for lb in labels]
    length = len(seq_list[0])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = [[0] * length for _ in range(n)]
    for i, s in enumerate(seq_list):
        row = mat[i]
        for j, ch in enumerate(s):
            row[j] = char_map[ch]
    
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ri = mat[i]
            rj = mat[j]
            diff = 0
            for k in range(length):
                if ri[k] != rj[k]:
                    diff += 1
            p = diff / length
            d = _jc_distance(p)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_build(dist, n):
    """Standard Neighbour-Joining. Returns list of (a, b) joins in order."""
    nodes = list(range(n))
    d = [row[:] for row in dist]
    
    joins = []
    
    while len(nodes) > 3:
        m = len(nodes)
        # Compute row sums
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                r[i] += d[i][j]
        
        # Compute Q matrix and find minimum
        best = None
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * d[i][j] - r[i] - r[j]
                if best is None or q < best:
                    best = q
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        ni, nj = nodes[i], nodes[j]
        joins.append((ni, nj))
        
        # Create new node
        new_node_id = n + len(joins) - 1
        
        # Build new node list and distance matrix
        new_nodes = []
        new_d = [0.0] * (m - 1)
        for k in range(m):
            if k == i or k == j:
                continue
            new_nodes.append(nodes[k])
            new_d.append(0.5 * (d[i][k] + d[j][k] - d[i][j]))
        
        # Build new distance matrix
        new_d_mat = [[0.0] * len(new_nodes) for _ in range(len(new_nodes))]
        for a in range(len(new_nodes)):
            for b in range(len(new_nodes)):
                if a == b:
                    continue
                # Find original indices in old nodes
                orig_a = new_nodes[a]
                orig_b = new_nodes[b]
                for ia in range(m):
                    if nodes[ia] == orig_a:
                        break
                for ib in range(m):
                    if nodes[ib] == orig_b:
                        break
                new_d_mat[a][b] = d[ia][ib]
        
        # Add new node
        new_nodes.append(new_node_id)
        for a in range(len(new_d_mat)):
            new_d_mat[a].append(new_d[a])
        new_d_row = [new_d[a] for a in range(len(new_d))]
        new_d_row.append(0.0)
        new_d_mat.append(new_d_row)
        
        nodes = new_nodes
        d = new_d_mat
    
    return joins, nodes


def _build_newick(joins, final_nodes, n, labels):
    """Build Newick string from NJ results."""
    newick = {}
    for i in range(n):
        newick[i] = labels[i]
    
    for idx, (a, b) in enumerate(joins):
        new_node_id = n + idx
        newick[new_node_id] = f"({newick[a]},{newick[b]})"
    
    if len(final_nodes) == 3:
        a, b, c = final_nodes
        result = f"({newick[a]},({newick[b]},{newick[c]}));"
    else:
        result = "(" + ",".join(labels) + ");"
    
    return result


def reconstruct(sequences):
    """Reconstruct phylogenetic tree from sequences."""
    start_time = time.time()
    deadline = start_time + 175.0  # 5s safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 0:
            return ""
        elif n == 1:
            return labels[0] + ";"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Check deadline
    if time.time() > deadline:
        return "(" + ",".join(labels) + ");"
    
    # Compute distances
    dist = _compute_distances(sequences, labels)
    
    # Check deadline
    if time.time() > deadline:
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    joins, final_nodes = _nj_build(dist, n)
    
    # Check deadline
    if time.time() > deadline:
        return "(" + ",".join(labels) + ");"
    
    # Build Newick
    return _build_newick(joins, final_nodes, n, labels)


# EVOLVE-BLOCK-END
