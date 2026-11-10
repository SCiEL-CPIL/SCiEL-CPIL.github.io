"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Vectorized Neighbour Joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import time
import numpy as np

_DEADLINE_START = None
_DEADLINE_LIMIT = 265.0  # wall-clock seconds; evaluator kills at 270s


def _elapsed():
    return time.time() - _DEADLINE_START


def _jc_distances(mat):
    """mat: (n, L) int array with 0..3 encoding. Returns (n, n) JC distances."""
    # per-site, per-taxon differences vs each other taxon
    # one-hot then count mismatches
    onehot = np.zeros((mat.shape[0], mat.shape[1], 4), dtype=np.int8)
    onehot[np.arange(mat.shape[0])[:, None], np.arange(mat.shape[1])[None, :], mat] = 1
    # mismatches[i,j] = L - sum over sites of (onehot[i,s] == onehot[j,s])
    # use dot product of one-hot blocks
    # dot: (n, L, 4) @ (4, L, n) -> (n, L, n); sum over L
    match = np.einsum('ilc,jlc->ij', onehot.astype(np.int16), onehot.astype(np.int16))
    L = mat.shape[1]
    mismatches = L - match
    p = mismatches.astype(np.float64) / L
    np.fill_diagonal(p, 0.0)
    # JC correction
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    # cap where p >= 0.75
    d = np.where(p >= 0.75, 3.0, d)
    d = np.where(np.isnan(d) | np.isinf(d), 3.0, d)
    return d



def _nj_with_structure(dist):
    n0 = dist.shape[0]
    D = dist.astype(np.float64).copy()
    nodes = list(range(n0))
    next_id = n0
    tree = {i: str(i) for i in range(n0)}
    while len(nodes) > 3:
        if _elapsed() > 260.0:
            break
        k = len(nodes)
        idx = np.array(nodes, dtype=np.int64)
        sub = D[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub, 0.0)
        r = sub.sum(axis=1)
        Q = (k - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i == j:
            break
        a = nodes[i]
        b = nodes[j]
        new = next_id
        next_id += 1
        tree[new] = "(" + tree[a] + "," + tree[b] + ")"
        others = [x for x in nodes if x not in (a, b)]
        if others:
            ox = np.array(others, dtype=np.int64)
            vals = 0.5 * (D[a, ox] + D[b, ox] - D[a, b])
            D[new, ox] = vals
            D[ox, new] = vals
        nodes = [x for x in nodes if x not in (a, b)] + [new]
        if new >= D.shape[0]:
            newsize = new + 1
            D2 = np.full((newsize, newsize), np.nan, dtype=np.float64)
            D2[:D.shape[0], :D.shape[0]] = D
            D = D2
    if len(nodes) == 3:
        a, b, c = nodes
        return "(" + tree[a] + "," + tree[b] + "," + tree[c] + ");"
    return "(" + ",".join(tree[x] for x in nodes) + ");"


def _reconstruct_single(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 265.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 3:
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
    # encode sequences
    enc = np.empty((n, len(sequences[labels[0]])), dtype=np.int8)
    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, lab in enumerate(labels):
        s = sequences[lab]
        enc[i] = np.fromiter((code[c] for c in s), dtype=np.int8, count=len(s))
    D = _jc_distances(enc)
    newick = _nj_with_structure(D)
    # relabel leaves: we used 0..n-1 as labels; need t0..t(n-1)
    # Since labels are sorted t0..t(n-1), leaf "i" corresponds to labels[i]
    # Replace leaf tokens
    for i in range(n - 1, -1, -1):
        # replace leaf label str(i) with labels[i]
        # careful: str(i) could be substring of internal node names; but we only use digits for leaves
        # and internal nodes are built as "(...)" so leaf tokens are bare integers
        # Use a safe replacement: replace occurrences that are leaf tokens
        # Since internal nodes contain parentheses, bare integers only appear as leaves
        # But an internal node string like "(0,1)" contains "0" and "1" as leaves - that's fine
        # However "10" would contain "1" and "0" - but we replace from high to low and use exact tokens
        # Better: do a single pass replacement using a placeholder
        pass
    # Simpler: rebuild by replacing leaf tokens with a regex-like approach
    # Since n <= 32, leaf labels are 0..31. Replace from largest to smallest.
    # But "1" is a substring of "10", "11", etc. We need word-boundary-like matching.
    # Use a two-pass: first replace all leaf tokens with unique placeholders, then with final names.
    # Actually, since the Newick only contains leaf tokens as bare integers (no internal numeric names),
    # and internal nodes are wrapped in parens, we can safely replace if we ensure no leaf token is
    # a substring of another leaf token in a way that causes issues.
    # For n <= 32, tokens are 0-31. "1" is substring of "10"-"19". So replacing "1" first would corrupt "10".
    # Solution: replace from largest to smallest, but "1" still appears in "10".
    # Proper solution: use a temporary unique token per leaf.
    result = newick
    # Map each leaf integer to a placeholder, then to final label
    # Do it in one pass: scan and replace integer tokens
    # Since the structure is "(leaf,leaf)" etc., integers are separated by non-digit chars
    import re
    # Replace each maximal digit run with the corresponding label
    def _repl(m):
        return labels[int(m.group(0))]
    result = re.sub(r'\d+', _repl, newick)
    return result


def reconstruct(sequences):
    global _DEADLINE_START, _DEADLINE_LIMIT
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
        _DEADLINE_LIMIT = _DEADLINE_START + 265.0

    try:
        return _reconstruct_single(sequences)
    except Exception:
        # fallback: star tree
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        if len(labels) <= 2:
            return "(" + ",".join(labels) + ");"
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
