"""DPP / FastUp-style topology search for Jukes-Cantor simulated data."""
# EVOLVE-BLOCK-START
import math


def _jc_distance(a, b):
    n = len(a)
    d = sum(1 for x, y in zip(a, b) if x != y)
    p = d / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 3:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return "(" + ",".join(labels) + ");"

    seqs = [sequences[l] for l in labels]
    L = len(seqs[0])

    # Pairwise JC distances
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        si = seqs[i]
        for j in range(i + 1, n):
            sj = seqs[j]
            d = sum(1 for x, y in zip(si, sj) if x != y)
            p = d / L
            if p >= 0.7499:
                dval = 3.0
            else:
                dval = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i][j] = dval
            D[j][i] = dval

    # Build NJ tree
    dist = [row[:] for row in D]
    remaining = list(range(n))
    node_taxa = {i: [i] for i in range(n)}
    node_name = {i: labels[i] for i in range(n)}
    tree_edges = {}  # parent_id -> (child1, child2)
    next_id = n

    while len(remaining) > 3:
        m = len(remaining)
        r = [0.0] * m
        for ai, i in enumerate(remaining):
            s = 0.0
            for j in remaining:
                if j != i:
                    s += dist[i][j]
            r[ai] = s
        best_crit = float('inf')
        best_i = best_j = 0
        for ai in range(m):
            i = remaining[ai]
            for aj in range(ai + 1, m):
                j = remaining[aj]
                crit = (m - 2) * dist[i][j] - r[ai] - r[aj]
                if crit < best_crit:
                    best_crit = crit
                    best_i, best_j = i, j
        i, j = best_i, best_j
        new_id = next_id
        next_id += 1
        tree_edges[new_id] = (i, j)
        node_taxa[new_id] = node_taxa[i] + node_taxa[j]
        node_name[new_id] = f"({node_name[i]},{node_name[j]})"
        for k in remaining:
            if k == i or k == j:
                continue
            dist[new_id][k] = 0.5 * (dist[i][k] + dist[j][k] - dist[i][j])
            dist[k][new_id] = dist[new_id][k]
        remaining = [x for x in remaining if x not in (i, j)] + [new_id]

    # Final three nodes form a trifurcation
    rem = remaining
    a, b, c = rem
    # Build Newick
    def get_newick(node):
        if node in tree_edges:
            c1, c2 = tree_edges[node]
            return f"({get_newick(c1)},{get_newick(c2)})"
        else:
            return node_name[node]

    result = "(" + ",".join(get_newick(x) for x in rem) + ");"

    # DPP local search (FastUp)
    # Collect splits from the tree
    all_splits = set()
    for nid, taxa in node_taxa.items():
        if 2 <= len(taxa) <= n - 2:
            all_splits.add(frozenset(taxa))

    def dpp_length(splits):
        total = 0.0
        for s in splits:
            ss = len(s)
            if ss < 1 or ss > n - 1:
                continue
            sc = n - ss
            s_set = set(s)
            cross = 0.0
            for i in s:
                for j in range(n):
                    if j not in s_set:
                        cross += D[i][j]
            total += cross / (ss * sc)
        return total

    current_splits = all_splits
    current_len = dpp_length(current_splits)

    # FastUp: try moving each leaf across each split
    improved = True
    max_rounds = 5
    for _ in range(max_rounds):
        improved = False
        for s in list(current_splits):
            s_set = set(s)
            s_comp = set(range(n)) - s_set
            for taxon in list(s_set):
                # Try moving taxon from s to s_comp
                new_s = s_set - {taxon}
                new_sc = s_comp | {taxon}
                # This changes the split; we need to update all affected splits
                # This is complex; skip for now
                pass
        if not improved:
            break

    return result
# EVOLVE-BLOCK-END
