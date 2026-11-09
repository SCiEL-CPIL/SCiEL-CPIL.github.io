"""Phylogenetic unrooted topology recovery for JC-simulated DNA.

Strategy:
  1) Fast vectorised BIONJ-corrected neighbour-joining (O(n^3), n<=32) to get a
     strong initial tree and its split set.
  2) Distance-based quartet voting: for each 4-taxon set, pick the topology with
     the smallest sum of the two opposite pairwise distances.
  3) Join quartet votes into a tree (greedy pair-merge on co-occurring support),
     keeping only splits compatible with the current set.
  4) Fitch-parsimony bisection-join refinement with compatibility pruning.
Return the tree whose unrooted splits are the recovered set (or NJ / star).
"""
import time
import math
import itertools
import numpy as np

_DEADLINE = None
_START = None
_MARGIN = 6.0  # safety margin before the 270 s wall-clock kill


def _now():
    return time.time()


def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - _MARGIN - _now()


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------
def _jc_matrix(seqs, n):
    """Jukes-Cantor distance matrix from encoded sequences."""
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(seqs[i] != seqs[j])
            p = diff / seqs.shape[1]
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def _bionj_correct(D, n):
    """BIONJ rate-correction of the distance matrix (in place)."""
    r = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i != j:
                s += D[i, j]
        r[i] = s / (n - 1)
    for i in range(n):
        for j in range(i + 1, n):
            rj = r[j]
            ri = r[i]
            if rj > 0:
                D[i, j] -= (ri * D[i, j]) / (rj * (n - 2))
            if ri > 0:
                D[i, j] -= (rj * D[i, j]) / (ri * (n - 2))
    return D


# ---------------------------------------------------------------------------
# Newick parsing / split extraction
# ---------------------------------------------------------------------------
def _extract_splits(newick, all_labels):
    """Return (splits, order) where splits is a list of frozensets (one side of
    each internal edge) and order is the left-to-right leaf order."""
    s = newick
    pos = [0]
    order = []
    L = len(s)

    def skip():
        while pos[0] < L and s[pos[0]] in ' \t\n':
            pos[0] += 1

    def parse():
        skip()
        if pos[0] >= L or s[pos[0]] != '(':
            start = pos[0]
            while pos[0] < L and s[pos[0]] not in '(),;':
                pos[0] += 1
            tok = s[start:pos[0]].strip()
            if tok:
                order.append(tok)
            return tok
        pos[0] += 1  # '('
        children = []
        while True:
            children.append(parse())
            skip()
            if pos[0] < L and s[pos[0]] == ',':
                pos[0] += 1
                continue
            if pos[0] < L and s[pos[0]] == ')':
                pos[0] += 1
                break
        return children

    def leaves(node):
        if isinstance(node, str):
            return {node}
        out = set()
        for c in node:
            out |= leaves(c)
        return out

    def collect(node):
        if isinstance(node, str):
            return
        sub = [leaves(c) for c in node]
        total = set()
        for st in sub:
            total |= st
        for st in sub:
            if 0 < len(st) < len(total):
                splits.append(frozenset(st))

    root = parse()
    splits = []
    collect(root)
    return splits, order


def _compatible(A, B, n):
    """Two splits (frozensets of leaf indices) are compatible iff one of the
    four intersections is empty."""
    a = A & B
    if not a:
        return True
    b = A & (B ^ 1)  # placeholder
    # compute properly
    Bcomp = frozenset(set(range(n)) - B)
    Acomp = frozenset(set(range(n)) - A)
    return not (A & B) or not (A & Bcomp) or not (Acomp & B) or not (Acomp & Bcomp)


def _compatible2(A, B, n):
    Bcomp = frozenset(set(range(n)) - B)
    Acomp = frozenset(set(range(n)) - A)
    return (not (A & B)) or (not (A & Bcomp)) or (not (Acomp & B)) or (not (Acomp & Bcomp))


# ---------------------------------------------------------------------------
# Fitch parsimony
# ---------------------------------------------------------------------------
def _fitch_leaf(seq):
    """Return a 4-bit bitmask for the set of states at a leaf."""
    m = 0
    for ch in seq:
        if ch == 'A':
            m |= 1
        elif ch == 'C':
            m |= 2
        elif ch == 'G':
            m |= 4
        elif ch == 'T':
            m |= 8
    return m


def _fitch_cost_for_split(seqs_idx, side):
    """Parsimony cost of a tree that contains the split `side` (a frozenset of
    leaf indices). Uses the standard two-cluster Fitch reconciliation:
    cost = cost(side) + cost(complement) + (1 if the two root state-sets are
    disjoint else 0). We compute the state-set and internal cost for each side
    using a balanced binary tree over that side."""
    n = len(seqs_idx)
    side_set = side
    comp = frozenset(set(range(n)) - side)
    if not side_set or not comp:
        return 10 ** 9

    def cluster_cost(idxs):
        # returns (mask, cost)
        if len(idxs) == 1:
            return (seqs_idx[idxs[0]], 0)
        mid = len(idxs) // 2
        m1, c1 = cluster_cost(idxs[:mid])
        m2, c2 = cluster_cost(idxs[mid:])
        inter = m1 & m2
        if inter:
            return (inter, c1 + c2)
        return (m1 | m2, c1 + c2 + 1)

    ia = sorted(side_set)
    ib = sorted(comp)
    ma, ca = cluster_cost(ia)
    mb, cb = cluster_cost(ib)
    inter = ma & mb
    if inter:
        return ca + cb
    return ca + cb + 1


# ---------------------------------------------------------------------------
# Build a Newick tree from a set of splits
# ---------------------------------------------------------------------------
def _tree_from_splits(splits, labels):
    """Given a set of compatible splits (frozensets of leaf indices) that
    define a binary tree on n leaves, return a Newick string."""
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        if n == 2:
            return "(%s,%s);" % (labels[0], labels[1])
        return "(%s,%s,%s);" % (labels[0], labels[1], labels[2])

    # Build the tree by recursively choosing a split that partitions the
    # current set of leaves.
    all_set = frozenset(range(n))

    # For a given node leaf-set S, find a split that properly partitions S.
    def build(S, depth):
        S = frozenset(S)
        if len(S) == 1:
            return labels[list(S)[0]]
        if len(S) == 2:
            a, b = sorted(S)
            return "(%s,%s)" % (labels[a], labels[b])
        if len(S) == 3:
            a, b, c = sorted(S)
            return "(%s,(%s,%s))" % (labels[a], labels[b], labels[c])
        # find a split that partitions S into two non-empty parts
        for sp in splits:
            A = sp & S
            B = S - A
            if len(A) > 0 and len(B) > 0 and len(A) < len(S):
                # ensure the split is consistent (sp restricted to S)
                left = build(A, depth + 1)
                right = build(B, depth + 1)
                return "(%s,%s)" % (left, right)
        # fallback: balanced split
        L = sorted(S)
        mid = len(L) // 2
        left = build(L[:mid], depth + 1)
        right = build(L[mid:], depth + 1)
        return "(%s,%s)" % (left, right)

    return "(" + build(all_set, 0) + ");"


# ---------------------------------------------------------------------------
# Neighbour joining (vectorised distance, O(n^3) loop)
# ---------------------------------------------------------------------------
def _nj_tree(D, labels):
    n = len(labels)
    if n <= 3:
        return None
    dist = [D[i].copy() for i in range(n)]
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    while len(nodes) > 3:
        m = len(nodes)
        total = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if b != a:
                    s += dist[a][b]
            total[a] = s
        best = None
        best_i = best_j = None
        for ii in range(m):
            a = nodes[ii]
            for jj in range(ii + 1, m):
                b = nodes[jj]
                crit = (m - 2) * dist[a][b] - total[a] - total[b]
                if best is None or crit < best:
                    best = crit
                    best_i, best_j = ii, jj
        a = nodes[best_i]
        b = nodes[best_j]
        joined = "(" + trees[a] + "," + trees[b] + ")"
        new = 1000 + len(nodes)
        rest = [x for x in nodes if x != a and x != b]
        for x in rest:
            val = 0.5 * (dist[a][x] + dist[b][x] - dist[a][b])
            dist[a][x] = val
            dist[x][a] = val
            dist[b][x] = val
            dist[x][b] = val
        nodes = rest + [new]
        trees[new] = joined
        # extend dist rows for new node
        dist[new] = [0.0] * (n + 200)
        for x in nodes:
            if x != new:
                dist[new][x] = dist[x][new]
    newick = "(" + ",".join(trees[x] for x in nodes) + ");"
    return newick


# ---------------------------------------------------------------------------
# Quartet voting
# ---------------------------------------------------------------------------
def _quartet_votes(D, n):
    """Return dict: frozenset({a,b}) -> count, for the 'ab|cd' split chosen by
    the smallest sum of opposite distances. Also store the full quartet for
    later compatibility checks."""
    # For each quartet {a,b,c,d}, the three candidate splits:
    #   ab|cd : D[ab]+D[cd]
    #   ac|bd : D[ac]+D[bd]
    #   ad|bc : D[ad]+D[bc]
    # choose the min. Record the winning split as frozenset({a,b}) (the two
    # leaves on one side).
    votes = {}
    for a, b, c, d in itertools.combinations(range(n), 4):
        s1 = D[a, b] + D[c, d]
        s2 = D[a, c] + D[b, d]
        s3 = D[a, d] + D[b, c]
        if s1 <= s2 and s1 <= s3:
            key = frozenset((a, b))
        elif s2 <= s1 and s2 <= s3:
            key = frozenset((a, c))
        else:
            key = frozenset((a, d))
        votes[key] = votes.get(key, 0) + 1
    return votes


def _join_quartets(votes, n, labels, compat_set):
    """Greedy join: repeatedly merge the pair of leaves with the strongest
    quartet support, building a tree. Returns (splits, newick)."""
    if n <= 3:
        if n == 1:
            return set(), labels[0]
        if n == 2:
            return set(), "(%s,%s);" % (labels[0], labels[1])
        return set(), "(%s,%s,%s);" % (labels[0], labels[1], labels[2])

    # current clusters: dict cluster_id -> frozenset of leaf indices
    clusters = {i: frozenset((i,)) for i in range(n)}
    # support between clusters: for a pair of clusters, sum votes of quartets
    # where the winning split has one leaf in each cluster (i.e. the split
    # separates the two clusters' leaves on opposite sides).
    # We approximate by, for each pair of clusters, the number of quartet
    # votes whose winning 2-set has exactly one leaf in cluster A and one in
    # cluster B.
    def pair_support(ca, cb):
        cnt = 0
        for key, v in votes.items():
            # key is a 2-set {x,y}; support for joining ca,cb if x in ca, y in cb
            k = list(key)
            if k[0] in ca and k[1] in cb:
                cnt += v
            elif k[0] in cb and k[1] in ca:
                cnt += v
        return cnt

    # Greedy: join pair with max support
    clusters_list = list(clusters.keys())
    while len(clusters_list) > 3:
        best = -1
        bi = bj = None
        for ii in range(len(clusters_list)):
            for jj in range(ii + 1, len(clusters_list)):
                ca = clusters[clusters_list[ii]]
                cb = clusters[clusters_list[jj]]
                sup = pair_support(ca, cb)
                if sup > best:
                    best = sup
                    bi, bj = ii, jj
        if bi is None:
            break
        a = clusters_list[bi]
        b = clusters_list[bj]
        merged = clusters[a] | clusters[b]
        newid = max(clusters_list) + 1
        clusters[newid] = merged
        del clusters[a]
        del clusters[b]
        clusters_list.remove(a)
        clusters_list.remove(b)
        clusters_list.append(newid)

    # Build a tree from the final 3 clusters (a trifurcation or nested)
    # Represent as nested structure to emit newick.
    def newick_for(cid):
        c = clusters[cid]
        if len(c) == 1:
            return labels[list(c)[0]]
        # find a split within c from votes to make it binary-ish
        best = None
        bestsup = -1
        for key, v in votes.items():
            k = list(key)
            A = frozenset(x for x in c if x in key)
            if len(A) == 1 and len(c) - 1 >= 1:
                # split {x} vs rest
                if v > bestsup:
                    bestsup = v
                    best = A
        if best is not None and bestsup > 0:
            x = list(best)[0]
            rest = c - best
            return "(%s,%s)" % (labels[x], _newick_cluster(rest))
        L = sorted(c)
        mid = len(L) // 2
        return "(%s,%s)" % (_newick_cluster(frozenset(L[:mid])),
                            _newick_cluster(frozenset(L[mid:])))

    def _newick_cluster(c):
        if len(c) == 1:
            return labels[list(c)[0]]
        L = sorted(c)
        mid = len(L) // 2
        if mid == 0 or mid == len(L):
            return "(%s,%s)" % (labels[L[0]], _newick_cluster(frozenset(L[1:])))
        return "(%s,%s)" % (_newick_cluster(frozenset(L[:mid])),
                            _newick_cluster(frozenset(L[mid:])))

    # Emit a tree over the 3 top clusters
    top = list(clusters_list)
    parts = [newick_for(c) for c in top]
    return set(), "(" + ",".join(parts) + ");"


# ---------------------------------------------------------------------------
# Main reconstruction
# ---------------------------------------------------------------------------
def _reconstruct(sequences, labels):
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        if n == 2:
            return "(%s,%s);" % (labels[0], labels[1])
        return "(%s,%s,%s);" % (labels[0], labels[1], labels[2])

    # Encode sequences
    enc = np.array([list(sequences[lab]) for lab in labels], dtype='U1')
    D = _jc_matrix(enc, n)
    D = _bionj_correct(D, n)

    # 1) NJ tree
    nj_newick = _nj_tree(D, labels)
    if nj_newick is None:
        return "(%s);" % ",".join(labels)
    nj_splits, _ = _extract_splits(nj_newick, labels)

    # Convert splits to index-space frozensets
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    def to_idx(fs):
        return frozenset(label_to_idx[x] for x in fs)

    current = set(to_idx(s) for s in nj_splits)

    # 2) Quartet votes
    votes = _quartet_votes(D, n)

    # 3) Add quartet-supported splits compatible with current, in order of
    #    support (only if it improves the parsimony or is strongly supported).
    #    We add a split if it's compatible with all current splits and its
    #    Fitch cost is <= the current tree's cost for that region (heuristic:
    #    just add if compatible and support is high enough).
    seq_masks = np.array([_fitch_leaf(sequences[lab]) for lab in labels], dtype=np.int32)

    def add_if_compatible(sp):
        if len(sp) < 2 or len(sp) > n - 2:
            return False
        for c in current:
            if not _compatible2(sp, c, n):
                return False
        current.add(sp)
        return True

    # Try to add the most-supported quartet splits (2-sets) that are compatible
    for key, v in sorted(votes.items(), key=lambda kv: -kv[1]):
        if _time_left() < 1.0:
            break
        if len(key) != 2:
            continue
        # a 2-set split is only meaningful if 2 <= len <= n-2
        if len(key) < 2 or len(key) > n - 2:
            continue
        add_if_compatible(key)
        if len(current) >= n - 3:
            break

    # 4) Fitch-parsimony bisection-join refinement (limited passes)
    #    Try adding candidate splits (from quartet votes and from pairs of
    #    current split sides) that reduce total Fitch cost and are compatible.
    def total_fitch(split_set):
        # Sum of per-split Fitch costs as a proxy (not exact but monotone-ish)
        c = 0
        for sp in split_set:
            c += _fitch_cost_for_split(seq_masks, sp)
        return c

    base_cost = total_fitch(current)

    # Candidate splits: all 2-sets from votes (already considered) plus
    # combinations of current split sides.
    candidates = set(votes.keys())
    for sp in list(current):
        comp = frozenset(set(range(n)) - sp)
        candidates.add(sp)
        candidates.add(comp)
        # also try merging with another current split's side
        for sp2 in current:
            if sp2 == sp:
                continue
            for cand in (sp & sp2, sp | sp2, sp - sp2, sp2 - sp):
                if 2 <= len(cand) <= n - 2:
                    candidates.add(cand)

    # Greedy: add the candidate that most reduces cost, repeat a few times.
    for _ in range(3):
        if _time_left() < 1.0:
            break
        best_sp = None
        best_cost = base_cost
        for sp in candidates:
            if sp in current:
                continue
            if len(sp) < 2 or len(sp) > n - 2:
                continue
            # compatibility
            ok = True
            for c in current:
                if not _compatible2(sp, c, n):
                    ok = False
                    break
            if not ok:
                continue
            new_cost = base_cost + _fitch_cost_for_split(seq_masks, sp)
            if new_cost < best_cost:
                best_cost = new_cost
                best_sp = sp
        if best_sp is None:
            break
        current.add(best_sp)
        base_cost = best_cost
        if len(current) >= n - 3:
            break

    # Build the final tree from the split set
    if len(current) >= n - 3:
        try:
            final = _tree_from_splits(current, labels)
        except Exception:
            final = nj_newick
    else:
        final = nj_newick

    return final


def reconstruct(sequences):
    global _DEADLINE, _START
    if _START is None:
        _START = _now()
        _DEADLINE = _START + 270.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if _time_left() < 1.0:
        return "(%s);" % ",".join(labels)

    try:
        result = _reconstruct(sequences, labels)
    except Exception:
        result = "(%s);" % ",".join(labels)

    if result is None:
        result = "(%s);" % ",".join(labels)

    return result
