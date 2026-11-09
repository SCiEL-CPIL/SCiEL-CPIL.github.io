import numpy as np

N = 16
M = 4  # number of parts in the top-level non-binary split


def _coloring(m):
    """Build the edge-color array for N vertices using an m-way hierarchical split.

    Returns (colors, idx) where colors is the list of N*(N-1)//2 colors (lex order)
    and idx maps (i,j) to the position in that list.
    """
    idx = {}
    pos = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx[(i, j)] = pos
            pos += 1

    parts = [list(range(k * (N // m), (k + 1) * (N // m))) for k in range(m)]

    def rec(subset, base):
        n_sub = len(subset)
        if n_sub <= 1:
            return
        if n_sub <= m:
            # within-part min-coloring (transitive), uses n_sub-1 colors
            for a in range(n_sub):
                for b in range(a + 1, n_sub):
                    idx[(subset[a], subset[b])] = base + a
            return
        # non-binary split into m balanced subparts
        size = n_sub // m
        extra = n_sub % m
        start = 0
        for k in range(m):
            length = size + (1 if k < extra else 0)
            subpart = subset[start:start + length]
            rec(subpart, base + k * (n_sub - 1))
            start += length

    # within-part colors: base = 0, uses 0 .. (n_sub-1) per part
    for k in range(m):
        rec(parts[k], 0)

    # cross-part colors: by part index of the lower-index endpoint
    for k in range(m):
        for i in parts[k]:
            for j in range(i + 1, N):
                if j in parts[k]:
                    continue
                idx[(i, j)] = k

    colors = [0] * (N * (N - 1) // 2)
    for (i, j), p in idx.items():
        colors[p] = p
    return colors, idx


def _validate(colors):
    """Return True iff no 4-cycle has 4 distinct colors."""
    c = np.array(colors, dtype=np.int32)
    idx = {}
    pos = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx[(i, j)] = pos
            pos += 1

    def get(a, b):
        if a > b:
            a, b = b, a
        return c[idx[(a, b)]]

    for a in range(N):
        for b in range(a + 1, N):
            for c_ in range(b + 1, N):
                for d in range(c_ + 1, N):
                    # 4-cycle a-c-b-d-a: edges (a,c),(c,b),(b,d),(d,a)
                    cc = {get(a, c_), get(c_, b), get(b, d), get(d, a)}
                    if len(cc) >= 4:
                        return False
    return True


# Precompute the coloring once (fast) and sanity-check it.
_COLORS, _IDX = _coloring(M)
assert len(_COLORS) == N * (N - 1) // 2
assert _validate(_COLORS), "construction invalid"


# EVOLVE-BLOCK-START
def construct():
    return list(_COLORS)
# EVOLVE-BLOCK-END
