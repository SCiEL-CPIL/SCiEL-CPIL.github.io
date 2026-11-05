"""Phylogenetic topology recovery: maximum-parsimony quartet + greedy split assembly."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

_DEADLINE = None
_TIME_LIMIT = 260.0  # safety margin under 270s wall-clock kill


def _elapsed():
    if _DEADLINE is None:
        return 0.0
    return time.time() - _DEADLINE[0]


def _quartet_parsimony_count(a, b, c, d, seqs, cols):
    """Return (n_ab_cd, n_ac_bd, n_ad_bc): parsimony steps for each of the 3 topologies."""
    n_ab_cd = 0
    n_ac_bd = 0
    n_ad_bc = 0
    for col in cols:
        sa = seqs[a][col]
        sb = seqs[b][col]
        sc = seqs[c][col]
        sd = seqs[d][col]
        # (a,b)|(c,d)
        if sa == sb and sc == sd:
            n_ab_cd += 1
        elif sa == sc and sb == sd:
            n_ab_cd += 2
        elif sa == sd and sb == sc:
            n_ab_cd += 2
        else:
            n_ab_cd += 2
        # (a,c)|(b,d)
        if sa == sc and sb == sd:
            n_ac_bd += 1
        elif sa == sb and sc == sd:
            n_ac_bd += 2
        elif sa == sd and sb == sc:
            n_ac_bd += 2
        else:
            n_ac_bd += 2
        # (a,d)|(b,c)
        if sa == sd and sb == sc:
            n_ad_bc += 1
        elif sa == sb and sc == sd:
            n_ad_bc += 2
        elif sa == sc and sb == sd:
            n_ad_bc += 2
        else:
            n_ad_bc += 2
    return n_ab_cd, n_ac_bd, n_ad_bc


def _build_tree_from_splits(splits, n_taxa, labels, rng):
    """Greedy assembly: start with star, repeatedly add the highest-weight compatible split.
    Returns a Newick string."""
    # A split is a frozenset of the smaller side (canonical: size <= n/2).
    # Compatibility: two splits A, B (canonical small sides) are compatible iff
    # A&B == empty, A&B == A, or A&B == B (one contains the other).

    def canonical_split(S):
        S = frozenset(S)
        if len(S) > n_taxa // 2:
            S = frozenset(set(labels) - S)
        if len(S) == 0 or len(S) == n_taxa:
            return None
        return S

    # Initialize: no splits -> star tree. We'll build the tree by recursively
    # partitioning using the selected splits.
    selected = []
    remaining = list(splits)  # list of (weight, frozenset_small_side)
    # Sort by weight descending
    remaining.sort(key=lambda x: -x[0])

    # We maintain the set of currently "available" regions as the full set of taxa.
    # A split s (small side S) is applicable if S is a proper subset of the current
    # full set AND S is compatible with all already-selected splits.
    # Simpler: greedy compatibility check against all selected splits.

    def compatible(s1, s2):
        # s1, s2 are canonical small sides
        inter = s1 & s2
        if len(inter) == 0:
            return True
        if inter == s1:
            return True
        if inter == s2:
            return True
        return False

    for weight, S in remaining:
        if S is None:
            continue
        ok = True
        for _, S2 in selected:
            if not compatible(S, S2):
                ok = False
                break
        if ok:
            selected.append((weight, S))
        if _elapsed() > _TIME_LIMIT - 30:
            break

    # Now build the tree from the selected splits.
    # We'll use a recursive partitioning: a node corresponds to a set of taxa.
    # The splits that are "active" at that node are those whose small side is a
    # proper non-empty subset of the node's taxa.
    # To make recursion tractable, we pick the split that best partitions the node.

    # Build a mapping from split -> weight for quick lookup.
    split_weight = {}
    for w, S in selected:
        # store both orientations
        split_weight[S] = w
        other = frozenset(set(labels) - S)
        split_weight[other] = w

    all_labels = set(labels)

    def build_node(taxa_set, depth=0):
        if len(taxa_set) <= 1:
            return list(taxa_set)[0]
        if len(taxa_set) == 2:
            a, b = list(taxa_set)
            return f"({a},{b})"
        # Find the best split that partitions taxa_set into two non-empty proper parts.
        best_split = None
        best_weight = -1
        for S in split_weight:
            inter = S & taxa_set
            if len(inter) == 0 or len(inter) == len(taxa_set):
                continue
            # inter is a proper non-empty subset of taxa_set
            # canonical within this node: pick the smaller side
            small = inter if len(inter) <= len(taxa_set) - len(inter) else taxa_set - inter
            # This split is usable at this node if small is a subset of taxa_set
            w = split_weight[S]
            if w > best_weight:
                best_weight = w
                best_split = small
        if best_split is None:
            # star: return a balanced-ish star
            items = list(taxa_set)
            return "(" + ",".join(items) + ")"
        left = best_split
        right = taxa_set - best_split
        left_str = build_node(left, depth + 1)
        right_str = build_node(right, depth + 1)
        return f"({left_str},{right_str})"

    return build_node(all_labels)


def _nj_fallback(sequences, labels):
    """Neighbour-Joining fallback if quartet assembly fails or times out."""
    import math
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ")"

    def jc_dist(a, b):
        diffs = sum(1 for x, y in zip(sequences[a], sequences[b]) if x != y)
        p = diffs / len(sequences[a])
        if p >= 0.7499:
            return 3.0
        return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

    matrix = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            d = jc_dist(a, b)
            matrix[(a, b)] = d
            matrix[(b, a)] = d
            matrix[(a, a)] = 0.0

    nodes = list(labels)
    trees = {name: name for name in labels}
    while len(nodes) > 3:
        n = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                crit = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or crit < best[0]:
                    best = (crit, a, b)
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            nd = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
            matrix[(new, x)] = nd
            matrix[(x, new)] = nd
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def reconstruct(sequences):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = [time.time()]
    # Reset per-call deadline tracking is not possible since _DEADLINE is global
    # across calls; but the budget is per all 20 instances, so we keep a global clock.

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"

    # For very small n, just do NJ (fast and reliable).
    if n <= 5:
        return _nj_fallback(sequences, labels)

    seqs = {lab: sequences[lab] for lab in labels}
    m = len(sequences[labels[0]])

    # Determine which columns to use for quartet parsimony.
    # For large m, subsample columns to keep quartet computation tractable.
    # Number of quartets is C(n,4). For n=32, that's 35960 quartets.
    # For each quartet, we scan `cols` columns. Total work ~ C(n,4) * |cols|.
    # Budget: we have ~260s total for 20 instances, so ~13s per instance on average.
    # Let's cap |cols| so that C(n,4)*|cols| stays reasonable.
    rng = random.Random(42)
    n_quartets = len(list(combinations(range(n), 4)))
    # Target: at most ~2e7 column-quartet operations per instance.
    max_ops = 2_000_000
    if n_quartets > 0:
        cols_count = max(1, min(m, max_ops // n_quartets))
    else:
        cols_count = m
    if cols_count >= m:
        cols = list(range(m))
    else:
        cols = rng.sample(range(m), cols_count)

    # Compute quartet scores.
    # For each 4-taxon subset {a,b,c,d}, compute parsimony counts for the 3 topologies.
    # The "support" for topology (a,b)|(c,d) is (min of the other two) - (that one),
    # i.e., how much better it is than the alternatives. We'll accumulate split support.
    # split_support: dict from frozenset (small side) -> total support weight.

    split_support = defaultdict(float)

    idx_map = {lab: i for i, lab in enumerate(labels)}

    quartet_iter = combinations(range(n), 4)
    for i, j, k, l in quartet_iter:
        if _elapsed() > _TIME_LIMIT - 40:
            break
        a = labels[i]
        b = labels[j]
        c = labels[k]
        d = labels[l]
        n_ab_cd, n_ac_bd, n_ad_bc = _quartet_parsimony_count(a, b, c, d, seqs, cols)
        # The best topology is the one with the minimum count.
        # Support for (a,b)|(c,d) = min(n_ac_bd, n_ad_bc) - n_ab_cd (if positive, it's the unique min)
        # For a balanced contribution, use: (n_ac_bd + n_ad_bc) / 2 - n_ab_cd, but only count if it's the argmin.
        # Simpler: assign support = (second_best - best) to the best topology's split.
        counts = [n_ab_cd, n_ac_bd, n_ad_bc]
        mn = min(counts)
        idx_min = counts.index(mn)
        # second best
        second = sorted(counts)[1]
        support = second - mn
        if support <= 0:
            continue
        if idx_min == 0:
            # (a,b)|(c,d)
            S = frozenset({a, b})
        elif idx_min == 1:
            # (a,c)|(b,d)
            S = frozenset({a, c})
        else:
            # (a,d)|(b,c)
            S = frozenset({a, d})
        # Canonicalize: use the smaller side
        if len(S) > n // 2:
            S = frozenset(set(labels) - S)
        if len(S) > 1 and len(S) < n - 1:
            split_support[S] += support

    if not split_support:
        return _nj_fallback(sequences, labels)

    # Convert to list of (weight, S) for greedy assembly.
    splits_list = [(w, S) for S, w in split_support.items()]

    # Greedy assembly.
    try:
        newick = _build_tree_from_splits(splits_list, n, labels, rng)
    except Exception:
        newick = _nj_fallback(sequences, labels)

    # Validate: ensure every label appears exactly once.
    # Strip trailing semicolon for check, then re-add.
    check = newick.rstrip(';').strip()
    found = set()
    for lab in labels:
        if check.count(lab) != 1:
            # Fallback if something went wrong.
            return _nj_fallback(sequences, labels)
    if not check.startswith('('):
        return _nj_fallback(sequences, labels)
    return newick if newick.endswith(';') else newick + ';'
# EVOLVE-BLOCK-END
