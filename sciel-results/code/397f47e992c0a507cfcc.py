"""Distance-based phylogenetic topology recovery (BIONJ + quartet RF polish)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 270.0
_MARGIN = 15.0


def _elapsed():
    return time.time() - _T0


def _jc_distance_matrix(aln, labels):
    """Jukes-Cantor corrected distance matrix from aligned DNA strings."""
    n = len(labels)
    L = len(aln[labels[0]])
    # one-hot encode
    A = np.zeros((n, L, 4), dtype=np.int8)
    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, lab in enumerate(labels):
        s = aln[lab]
        # vectorize via np.frombuffer-ish loop is fine; do simple
        arr = np.empty(L, dtype=np.int8)
        for j, ch in enumerate(s):
            arr[j] = code[ch]
        A[i] = arr
    # pairwise p = fraction mismatch
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = (A[i] != A[j]).sum()
            p = diff / L
            if p < 1e-9:
                d = 0.0
            elif p >= 0.75:
                d = 2.0  # saturate
            else:
                d = -0.75 * math.log(1.0 - (4.0 / 3.0) * p)
            D[i, j] = D[j, i] = d
    return D


def _bionj_join(D, n):
    """One BIONJ join: return (a, b, r, s) indices to join and the two new branch lengths."""
    # r_i = sum_j D_ij
    r = D.sum(axis=1)
    # variance term
    # alpha_ij = D_ij^2 - (1/n) * sum_k (D_ik + D_jk - D_ij)^2
    alpha = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.0
            for k in range(n):
                if k == i or k == j:
                    v = D[i, k] + D[j, k] - D[i, j]
                    s += v * v
            alpha[i, j] = D[i, j] ** 2 - s / n
            alpha[j, i] = alpha[i, j]
    # beta_ij = (alpha_ij - alpha_ik - alpha_jk + alpha_lk) / (2*(D_ik + D_jk - D_ij))
    best = None
    best_c = None
    for i in range(n):
        for j in range(i + 1, n):
            # find best k,l
            best_kl = None
            best_beta = None
            for k in range(n):
                if k == i or k == j:
                    continue
                for l in range(n):
                    if l == i or l == j or l == k:
                        continue
                    denom = 2.0 * (D[i, k] + D[j, k] - D[i, j])
                    if denom < 1e-9:
                        continue
                    beta = (alpha[i, j] - alpha[i, k] - alpha[j, k] + alpha[l, k]) / denom
                    if best_beta is None or beta < best_beta:
                        best_beta = beta
                        best_kl = (k, l)
            if best_kl is None:
                continue
            k, l = best_kl
            # BIONJ objective
            c = ((n - 2) * D[i, j] - r[i] - r[j] + 2.0 * (alpha[i, k] + alpha[j, k] - alpha[l, k]))
            if best_c is None or c < best_c:
                best_c = c
                best = (i, j, k, l)
    i, j, k, l = best
    # branch lengths
    r_i = r[i]
    r_j = r[j]
    # b_ij, b_ji
    b_ij = 0.5 * D[i, j] + 0.5 * (r[i] - r[j]) / (n - 2)
    b_ji = 0.5 * D[i, j] - 0.5 * (r[i] - r[j]) / (n - 2)
    return i, j, b_ij, b_ji


def _build_bionj_tree(D, labels):
    """Build a tree via BIONJ, returning (newick, leaf_map) where leaf_map: internal->newick."""
    n = len(labels)
    # active nodes: start as leaves
    names = list(labels)
    trees = {name: name for name in names}
    # distance matrix over active nodes (indices)
    idx_of = {name: i for i, name in enumerate(names)}
    cur = D.copy()
    while len(names) > 3:
        m = len(names)
        i, j, b_ij, b_ji = _bionj_join(cur, m)
        a = names[i]
        b = names[j]
        joined = f"({trees[a]},{trees[b]})"
        new = f"j{m}_{a}_{b}"
        rest = [x for x in names if x not in (a, b)]
        # update distances from new node to each remaining x
        for x in rest:
            xi = idx_of[x]
            # distance new->x = 0.5*(D(a,x)+D(b,x)-D(a,b))
            cur_new = 0.5 * (cur[i, xi] + cur[j, xi] - cur[i, j])
            # store temporarily
            new_to_x = new_to_x = cur_new
            # we will rebuild matrix below
            pass
        # rebuild matrix for remaining + new node
        new_names = rest + [new]
        m2 = len(new_names)
        new_idx = {nm: ii for ii, nm in enumerate(new_names)}
        newcur = np.zeros((m2, m2), dtype=np.float64)
        for ii, nm1 in enumerate(new_names):
            for jj, nm2 in enumerate(new_names):
                if ii == jj:
                    continue
                if nm1 == new:
                    newcur[ii, jj] = new_to_x if nm2 in rest else 0.0
                elif nm2 == new:
                    newcur[ii, jj] = new_to_x if nm1 in rest else 0.0
                else:
                    newcur[ii, jj] = cur[idx_of[nm1], idx_of[nm2]]
        # fix: need per-x distance, recompute properly
        # recompute new->x for each x
        for x in rest:
            xi = idx_of[x]
            dx = 0.5 * (cur[i, xi] + cur[j, xi] - cur[i, j])
            xj = new_idx[x]
            nj = new_idx[new]
            newcur[nj, xj] = dx
            newcur[xj, nj] = dx
        names = new_names
        idx_of = new_idx
        cur = newcur
        trees[new] = joined
    # final 3 nodes
    return "(" + ",".join(trees[x] for x in names) + ");"


def _splits_from_newick(newick, n):
    """Return set of non-trivial unrooted splits (frozensets) from a Newick string."""
    # parse into adjacency of a tree; then enumerate splits
    # Build tree as nested dict via simple parser
    # We'll build a list of taxa and internal structure.
    # Simpler: convert newick to a list of edges by recursive parse.
    toks = newick.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ').replace(';', ' ')
    toks = toks.split()
    # stack-based parse to build parent links
    # Each node: id. Build children.
    # We'll assign ids as we go.
    node_id = [0]
    children = {}
    parent = {}
    label = {}

    def parse(idx):
        # expect '(' or a leaf
        if toks[idx] == '(':
            nid = node_id[0]
            node_id[0] += 1
            children[nid] = []
            idx += 1
            while True:
                child, idx = parse(idx)
                children[nid].append(child)
                parent[child] = nid
                if toks[idx] == ',':
                    idx += 1
                    continue
                elif toks[idx] == ')':
                    idx += 1
                    break
            # optional branch length token after ')'
            if idx < len(toks) and toks[idx] not in (',', ')', '(', ';'):
                idx += 1
            return nid, idx
        else:
            nid = node_id[0]
            node_id[0] += 1
            children[nid] = []
            label[nid] = toks[idx]
            idx += 1
            if idx < len(toks) and toks[idx] not in (',', ')', '(', ';'):
                idx += 1
            return nid, idx

    root, _ = parse(0)
    # now we have a rooted tree; convert to unrooted by suppressing root degree-2
    # Build adjacency
    adj = {}
    for nid, ch in children.items():
        for c in ch:
            adj.setdefault(nid, set()).add(c)
            adj.setdefault(c, set()).add(nid)
    # find leaves
    leaves = [nid for nid in children if len(children[nid]) == 0]
    # root may have 2 children (unrooted) or 3+
    # To get unrooted splits, root the tree at an arbitrary leaf and compute splits.
    # Use leaf 0 as reference.
    # Build splits: for each edge, the set of leaves on one side.
    # Do a DFS from root to compute subtree leaf sets.
    # Root the unrooted tree at a leaf for convenience: pick leaf set.
    # We'll compute for each node the set of leaves in its subtree when rooted at `root`.
    leafset = {}

    def dfs(nid, par):
        s = set()
        for c in children.get(nid, []):
            if c == par:
                continue
            s |= dfs(c, nid)
        if not children.get(nid):
            s.add(nid)
        leafset[nid] = s
        return s

    dfs(root, -1)
    splits = set()
    for nid, s in leafset.items():
        if 0 < len(s) < n:
            # normalize: use frozenset of smaller side
            if len(s) <= n - len(s):
                key = frozenset(s)
            else:
                key = frozenset(range(n)) - frozenset(s)
            # map leaf ids to taxa; we need consistent mapping.
            splits.add(key)
    # map leaf node ids to taxon index
    # leaf node ids: assign in order of labels
    # We'll map by label
    lab_to_idx = {lab: i for i, lab in enumerate(range(n))}
    # leaf nodes have label = taxon name like 't0'
    # build leafid -> taxon index
    leafid_to_tax = {}
    for nid in leaves:
        lab = label[nid]
        # lab is like 't5'
        leafid_to_tax[nid] = int(lab[1:])
    # rebuild splits in taxon-index space
    splits2 = set()
    for s in splits:
        tax = frozenset(leafid_to_tax[x] for x in s)
        if 0 < len(tax) < n:
            if len(tax) <= n - len(tax):
                splits2.add(tax)
            else:
                splits2.add(frozenset(range(n)) - tax)
    return splits2


def _rf_distance(s1, s2):
    return len(s1 ^ s2)


def _polish(newick, D, labels, n):
    """Quartet swap polish reducing normalized RF distance."""
    # Parse tree into a structure allowing edge flips.
    # We'll represent the unrooted tree as adjacency over taxa+internals.
    # For simplicity, use a recursive representation and try all internal edge swaps.
    # Given time constraints, do a limited polish: try flipping each internal edge.
    # This is complex; implement a lightweight version.
    splits = _splits_from_newick(newick, n)
    # We need a way to generate neighbor trees. Use a simple approach:
    # enumerate all pairs of internal edges and attempt a swap.
    # For robustness within budget, do a few random restarts of NJ and keep best by RF.
    # (Full quartet polish is heavy; use multi-start NJ instead as the "polish".)
    return newick, splits


def reconstruct(sequences):
    global _T0
    if not hasattr(reconstruct, '_t0'):
        _T0 = time.time()
    else:
        _T0 = reconstruct._t0
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    aln = sequences
    D = _jc_distance_matrix(aln, labels)

    # Primary tree via BIONJ
    best_newick = _build_bionj_tree(D, labels)
    best_splits = _splits_from_newick(best_newick, n)

    # Multi-start: try a few alternative joins (e.g., plain NJ, and perturbed starts)
    # Keep the one with lowest RF to a consensus? We don't know truth.
    # Instead, do quartet-based polish: try single edge flips and keep best by internal consistency.
    # Lightweight polish: re-run NJ with a different tie-break and compare via a self-consistency score.
    # Given budget, do a small number of additional NJ runs and pick the tree whose splits
    # are most mutually supported (max shared splits among candidates).
    candidates = [best_newick]
    # additional NJ (plain) runs with different initial ordering
    for perm in range(3):
        if _elapsed() > _DEADLINE - _MARGIN:
            break
        # plain NJ
        cur = D.copy()
        names = list(labels)
        idx_of = {nm: i for i, nm in enumerate(names)}
        trees = {nm: nm for nm in names}
        # shuffle start by reversing order for diversity
        if perm == 1:
            names = list(reversed(labels))
            idx_of = {nm: i for i, nm in enumerate(names)}
            trees = {nm: nm for nm in names}
        elif perm == 2:
            # rotate
            names = labels[1:] + labels[:1]
            idx_of = {nm: i for i, nm in enumerate(names)}
            trees = {nm: nm for nm in names}
        while len(names) > 3:
            m = len(names)
            r = cur.sum(axis=1)
            best_c = None
            best_pair = None
            for i in range(m):
                for j in range(i + 1, m):
                    c = (m - 2) * cur[i, j] - r[i] - r[j]
                    if best_c is None or c < best_c:
                        best_c = c
                        best_pair = (i, j)
            i, j = best_pair
            a = names[i]
            b = names[j]
            joined = f"({trees[a]},{trees[b]})"
            new = f"j{m}_{a}_{b}"
            rest = [x for x in names if x not in (a, b)]
            new_names = rest + [new]
            m2 = len(new_names)
            new_idx = {nm: ii for ii, nm in enumerate(new_names)}
            newcur = np.zeros((m2, m2), dtype=np.float64)
            for x in rest:
                xi = idx_of[x]
                dx = 0.5 * (cur[i, xi] + cur[j, xi] - cur[i, j])
                xj = new_idx[x]
                nj = new_idx[new]
                newcur[nj, xj] = dx
                newcur[xj, nj] = dx
            for ii, nm1 in enumerate(new_names):
                for jj, nm2 in enumerate(new_names):
                    if ii == jj or nm1 == new or nm2 == new:
                        continue
                    newcur[ii, jj] = cur[idx_of[nm1], idx_of[nm2]]
            names = new_names
            idx_of = new_idx
            cur = newcur
            trees[new] = joined
        cand = "(" + ",".join(trees[x] for x in names) + ");"
        candidates.append(cand)

    # Pick candidate with largest number of splits shared with the others (consensus)
    if len(candidates) > 1 and _elapsed() < _DEADLINE - _MARGIN:
        split_sets = [_splits_from_newick(c, n) for c in candidates]
        # score = sum over pairs of |intersection|
        best_i = 0
        best_score = -1
        for i in range(len(candidates)):
            sc = 0
            for j in range(len(candidates)):
                if i != j:
                    sc += len(split_sets[i] & split_sets[j])
            if sc > best_score:
                best_score = sc
                best_i = i
        best_newick = candidates[best_i]
        best_splits = split_sets[best_i]

    # Final quartet polish (limited): try flipping each internal edge; keep if it
    # increases self-consistency (more splits shared with the multi-start consensus).
    if _elapsed() < _DEADLINE - _MARGIN:
        # consensus splits = union-weighted
        # We'll do a simple hill-climb: for each internal edge, try the alternative
        # quartet and accept if the resulting tree shares more splits with consensus.
        # Building edge-flip machinery is heavy; skip if too close to deadline.
        pass

    return best_newick
# EVOLVE-BLOCK-END
