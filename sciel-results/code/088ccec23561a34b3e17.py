"""Phylogenetic topology recovery: combinatorial split-based quartet consensus."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # One-hot encode
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    codes = np.zeros((n, L, 4), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, ch in enumerate(s):
            codes[i, j, base_map[ch]] = 1
    # Hamming via matrix multiply per base
    D = np.zeros((n, n), dtype=np.float64)
    for b in range(4):
        M = codes[:, :, b]
        D += M @ M.T
    D = 4.0 * L - D
    np.fill_diagonal(D, 0.0)
    p = np.clip(D / float(L), 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D, labels):
    """Vectorized Neighbor Joining. Returns Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + ",(" + labels[1] + "," + labels[2] + "));"

    # Use arrays for active nodes
    active_nodes = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    dist = D.copy()
    next_id = n

    while len(active_nodes) > 3:
        m = len(active_nodes)
        # Row sums
        row_sums = np.array([sum(dist[active_nodes[i], active_nodes[j]] for j in range(m) if j != i) for i in range(m)])
        # Find min criterion
        best_crit = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                crit = (m - 2) * dist[active_nodes[i], active_nodes[j]] - row_sums[i] - row_sums[j]
                if crit < best_crit:
                    best_crit = crit
                    best_i, best_j = i, j

        a = active_nodes[best_i]
        b = active_nodes[best_j]
        new_id = next_id
        next_id += 1

        # Compute new distances
        rest = [active_nodes[k] for k in range(m) if k != best_i and k != best_j]
        new_dists = {}
        for x in rest:
            val = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
            new_dists[x] = val

        # Update dist matrix (extend)
        new_dist = np.zeros((new_id + 1, new_id + 1), dtype=np.float64)
        for x in rest:
            new_dist[x, new_id] = new_dists[x]
            new_dist[new_id, x] = new_dists[x]
        # Copy old distances for remaining pairs
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                if active_nodes[i] != a and active_nodes[i] != b and active_nodes[j] != a and active_nodes[j] != b:
                    new_dist[active_nodes[i], active_nodes[j]] = dist[active_nodes[i], active_nodes[j]]
                    new_dist[active_nodes[j], active_nodes[i]] = dist[active_nodes[i], active_nodes[j]]

        dist = new_dist
        tree_str[new_id] = "(" + tree_str[a] + "," + tree_str[b] + ")"
        active_nodes = rest + [new_id]

    rem = active_nodes
    return "(" + ",".join(tree_str[x] for x in rem) + ");"


def _quartet_split_scores(sequences, labels, deadline):
    """
    Score all candidate splits by counting how many quartets support them.
    Returns dict: frozenset(smaller_side) -> (vote_count, margin)
    Uses direct sequence comparison (no distance matrix needed for topology).
    """
    n = len(labels)
    L = len(sequences[labels[0]])
    allset = frozenset(range(n))
    t_start = time.time()

    # Pre-encode sequences as int arrays for fast comparison
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = []
    for lab in labels:
        s = sequences[lab]
        arr = np.array([base_map[ch] for ch in s], dtype=np.int8)
        seq_arrays.append(arr)

    # For each quartet, determine the best split by comparing pairwise agreement
    # Under JC, the true split minimizes the sum of within-group distances.
    # We use a fast heuristic: for quartet {a,b,c,d}, compute:
    #   score_ab = sum over sites of (1 if seq_a[site]==seq_b[site] else 0)
    #   Similarly for ac, ad, bc, bd, cd
    #   Then s_ab_cd = (L - agree_ab) + (L - agree_cd) = 2L - agree_ab - agree_cd
    #   The split with minimum sum wins.

    votes = {}

    for a in range(n):
        if time.time() - t_start > deadline * 0.4:
            break
        sa = seq_arrays[a]
        for b in range(a + 1, n):
            if time.time() - t_start > deadline * 0.4:
                break
            sb = seq_arrays[b]
            agree_ab = np.sum(sa == sb)
            for c in range(b + 1, n):
                if time.time() - t_start > deadline * 0.4:
                    break
                sc = seq_arrays[c]
                agree_ac = np.sum(sa == sc)
                agree_bc = np.sum(sb == sc)
                for d in range(c + 1, n):
                    sd = seq_arrays[d]
                    agree_ad = np.sum(sa == sd)
                    agree_bd = np.sum(sb == sd)
                    agree_cd = np.sum(sc == sd)

                    # Distance-like scores (higher agreement = lower distance)
                    # s_ab_cd: a,b together; c,d together
                    s_ab_cd = (L - agree_ab) + (L - agree_cd)
                    s_ac_bd = (L - agree_ac) + (L - agree_bd)
                    s_ad_bc = (L - agree_ad) + (L - agree_bc)

                    m = min(s_ab_cd, s_ac_bd, s_ad_bc)
                    if m == s_ab_cd:
                        split = frozenset((a, b))
                        margin = (min(s_ac_bd, s_ad_bc) - m)
                    elif m == s_ac_bd:
                        split = frozenset((a, c))
                        margin = (min(s_ab_cd, s_ad_bc) - m)
                    else:
                        split = frozenset((a, d))
                        margin = (min(s_ab_cd, s_ac_bd) - m)

                    # Normalize to smaller side
                    comp = allset - split
                    if len(comp) < len(split):
                        split = comp

                    if split in votes:
                        votes[split] = (votes[split][0] + 1, votes[split][1] + margin)
                    else:
                        votes[split] = (1, margin)

    return votes


def _select_compatible_splits(votes, n, deadline):
    """
    Greedy maximum-weight independent set on split conflict graph.
    Weight = vote_count * (1 + margin/n_quartets) to favor high-confidence splits.
    Returns list of chosen splits (frozensets).
    """
    allset = frozenset(range(n))
    t_start = time.time()

    # Total number of quartets
    n_quartets = 1
    for i in range(4, n + 1):
        n_quartets *= i
    n_quartets //= 24  # 4!

    # Rank by (votes, margin) descending
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))

    chosen = []
    chosen_comps = []  # complements for fast check

    for split, (cnt, margin) in ranked:
        if time.time() - t_start > deadline * 0.3:
            break
        if len(chosen) >= n - 3:
            break

        comp = allset - split
        ok = True
        for existing, ecomp in zip(chosen, chosen_comps):
            # Incompatible if all four intersections are non-empty
            if (split & existing) and (split & ecomp) and (comp & existing) and (comp & ecomp):
                ok = False
                break

        if ok:
            chosen.append(split)
            chosen_comps.append(comp)

    return chosen


def _build_tree_from_splits(chosen_splits, labels, D, deadline):
    """
    Build a tree from a set of compatible splits using recursive decomposition.
    Falls back to NJ on subtrees where no split partitions.
    """
    n = len(labels)
    allset = frozenset(range(n))
    t_start = time.time()

    def build(taxa_fset):
        if time.time() - t_start > deadline * 0.3:
            # Fallback: NJ on this subset
            idx_list = sorted(taxa_fset)
            sub = D[np.ix_(idx_list, idx_list)]
            sub_labels = [labels[i] for i in idx_list]
            return _neighbor_joining(sub, sub_labels)

        if len(taxa_fset) == 1:
            return labels[list(taxa_fset)[0]]
        if len(taxa_fset) == 2:
            a, b = sorted(taxa_fset)
            return "(" + labels[a] + "," + labels[b] + ")"

        # Find a split that partitions this set
        best_part = None
        for sp in chosen_splits:
            part = sp & taxa_fset
            if 0 < len(part) < len(taxa_fset):
                best_part = part
                break

        if best_part is None:
            # No split partitions: use NJ
            idx_list = sorted(taxa_fset)
            sub = D[np.ix_(idx_list, idx_list)]
            sub_labels = [labels[i] for i in idx_list]
            return _neighbor_joining(sub, sub_labels)

        other = taxa_fset - best_part
        left = build(best_part)
        right = build(other)
        return "(" + left + "," + right + ")"

    return build(allset)


def reconstruct(sequences):
    t0 = time.time()
    deadline = 240.0  # safety margin under 270s

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        return "(" + ",".join(labels) + ");"

    # Compute distance matrix (needed for NJ fallback)
    D = _jc_distance_matrix(sequences, labels)

    # Primary: combinatorial split-based quartet consensus
    try:
        remaining = deadline - (time.time() - t0)
        if remaining > 5.0:
            votes = _quartet_split_scores(sequences, labels, remaining * 0.6)
            if votes:
                chosen = _select_compatible_splits(votes, n, remaining * 0.25)
                if chosen:
                    newick = _build_tree_from_splits(chosen, labels, D, remaining * 0.15)
                    # Validate
                    from collections import Counter
                    import re
                    found = re.findall(r't\d+', newick)
                    cnt = Counter(found)
                    if len(cnt) == n and all(cnt[l] == 1 for l in labels):
                        return newick
    except Exception:
        pass

    # Fallback: neighbor joining
    try:
        if time.time() - t0 < deadline:
            newick = _neighbor_joining(D, labels)
            from collections import Counter
            import re
            found = re.findall(r't\d+', newick)
            cnt = Counter(found)
            if len(cnt) == n and all(cnt[l] == 1 for l in labels):
                return newick
    except Exception:
        pass

    # Last resort: star tree
    return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
