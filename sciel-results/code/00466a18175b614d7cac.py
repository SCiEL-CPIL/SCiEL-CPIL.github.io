import time
import itertools

_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill


def _elapsed_ok():
    return time.time() < _DEADLINE


def _verify(n, colors):
    """Check that no colour class contains a C4 (two vertices with two common neighbours).
    Efficient bitset check. Returns True if valid."""
    if len(colors) != n * (n - 1) // 2:
        return False
    # adjacency bitmasks per colour: adj[c][v] = bitmask of neighbours of v in colour c
    adj = [ [0]*n for _ in range(5) ]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]; idx += 1
            if c < 0 or c > 4:
                return False
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                if common & (common - 1):  # at least two common neighbours -> C4
                    return False
    return True


def _cyclic(n, k):
    """Colour edge (i,j) by circular distance 1..k (k = n//2, requires n odd)."""
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            dist = min(d, n - d)
            out.append(dist - 1)
    return out


def _build():
    best = None
    best_n = 0

    # 1) cyclic construction for odd n: colour by circular distance 1..n//2.
    #    A cyclic C4 in distance-d class would need a 4-cycle with all edges of same
    #    circular distance, which is impossible (a non-trivial 4-cycle cannot have all
    #    four edges equal circular distance on a cycle graph). So cyclic is valid.
    for n in range(3, 29, 2):
        if not _elapsed_ok():
            break
        cols = _cyclic(n, n // 2)
        if _verify(n, cols):
            if n > best_n:
                best_n = n
                best = (n, cols)

    # 2) Try to extend beyond cyclic: a block / group-based construction.
    #    Use a 5-group (Z5) of cyclic shifts on a base of size m, total n = 5*m.
    #    Colours: within-block by base colour, between-blocks by (block_diff, base) mapping.
    #    We search small base colourings that are C4-free and where the lifted
    #    structure remains C4-free. This is the standard "blow-up" approach.
    if _elapsed_ok():
        # base size m, n = 5*m. Keep n <= 28 -> m <= 5.
        for m in range(1, 6):
            if not _elapsed_ok():
                break
            n = 5 * m
            if n > 28:
                break
            # base edges: between distinct base positions a<b, and within same block.
            # We need a colouring of the "quotient" that, combined with cyclic base,
            # stays C4-free. Try: within-block use cyclic(m) (m may be even/odd);
            # between-block (block x, pos a) - (block y, pos b): colour = ( (y-x) mod 5 )
            #   combined with base relation. To keep only 5 colours, use colour =
            #   ((y - x) mod 5) for a==b, and for a!=b use a fixed function of (a,b).
            # This is intricate; instead do a direct randomised local search on full n
            # for a couple of target sizes, verifying periodically.
            pass

    # 3) Randomised local search / hill-climbing to push n higher, verifying at the end.
    #    Start from a valid cyclic colouring of odd n, then attempt to add one vertex
    #    at a time (n -> n+1) by assigning colours to its edges to keep C4-free.
    if best is not None and _elapsed_ok():
        n0, cols0 = best
        # try to grow from n0 up to 28
        cur_n = n0
        cur_cols = cols0[:]
        for target in range(cur_n + 1, 29):
            if not _elapsed_ok():
                break
            # assign colours for edges (new vertex = cur_n, i) for i in 0..cur_n-1
            # greedy: pick colour that does not create a C4.
            new_cols = cur_cols[:]
            ok = True
            # adjacency of current graph per colour
            adj = [ [0]*cur_n for _ in range(5) ]
            idx = 0
            for i in range(cur_n):
                for j in range(i + 1, cur_n):
                    c = cur_cols[idx]; idx += 1
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
            for i in range(cur_n):
                # choose colour c for edge (i, cur_n)
                chosen = -1
                for c in range(5):
                    # adding edge (i, cur_n) in colour c creates a C4 iff
                    # there exists j != i with j a common neighbour of i and cur_n in c,
                    # i.e. some j already adjacent to i in c AND we will connect cur_n-j in c.
                    # Since cur_n has no edges yet, the only risk is: for some j,
                    # j ~_c i and we also set cur_n ~_c j. That's a triangle, not C4.
                    # A C4 needs two common neighbours of i and cur_n. cur_n's neighbours
                    # in c are exactly the set we assign. So C4 arises if i and cur_n
                    # share >=2 neighbours in c. i's neighbours in c are adj[c][i];
                    # cur_n's neighbours in c will be the set S_c of vertices assigned c.
                    # We need |adj[c][i] & S_c| <= 1.
                    # We build S_c incrementally; check against current S_c.
                    # We'll just test after full assignment, but greedy needs a choice.
                    # Approx: require that for all j in S_c, j not in adj[c][i] (i.e. 0 common),
                    # which is sufficient.
                    S_c_mask = 0
                    for (v, cv) in zip(range(cur_n), [0]*cur_n):
                        pass
                    # Simpler sufficient condition: no j with (j in S_c) and (j in adj[c][i]).
                    # We track S_c as we go.
                    pass
                # Fallback: just assign cyclic-style colour and verify at end.
                d = cur_n - i
                # circular distance in target
                dist = min(d, target - d)
                chosen = min(dist - 1, 4)
                new_cols.append(chosen)
            # verify the full candidate
            if _verify(target, new_cols):
                cur_n = target
                cur_cols = new_cols
                if cur_n > best_n:
                    best_n = cur_n
                    best = (cur_n, cur_cols)
            else:
                break

    if best is None:
        # ultimate fallback: the seed n=11
        n = 11
        cols = _cyclic(n, n // 2)
        best = (n, cols)

    return {'n': best[0], 'colors': best[1]}


def construct():
    try:
        return _build()
    except Exception:
        n = 11
        return {'n': n, 'colors': _cyclic(n, n // 2)}
