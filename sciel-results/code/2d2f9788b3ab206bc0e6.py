"""Find a 4-coloring of K_n with no monochromatic C6, maximizing n."""
import time
import random
from collections import defaultdict

DEADLINE = 115.0  # safety margin under 120s wall-clock

def _edge_idx(i, j, n):
    # lexicographic index for pair (i,j), i<j
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Return list of 4 adjacency lists (sets) for fast C6 detection."""
    adj = [set() for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[_edge_idx(i, j, n)]
            adj[c].add(i)
            # store per-vertex adjacency within color c
    # build per-color per-vertex neighbor lists
    adjv = [defaultdict(set) for _ in range(4)]
    for c in range(4):
        s = adj[c]
        # s is a set of vertices? no, we need edges. Let's redo.
    # Simpler: build per-color adjacency as dict vertex -> set(neighbors)
    adjv = [defaultdict(set) for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[_edge_idx(i, j, n)]
            adjv[c][i].add(j)
            adjv[c][j].add(i)
    return adjv

def _has_c6(adjv, c):
    """Check if color c contains a (not necessarily induced) 6-cycle."""
    av = adjv[c]
    # For each edge (u,v) in color c, check if there is a path of length 4 from u to v
    # using only color-c edges, with all 6 vertices distinct.
    for u in av:
        for v in av[u]:
            if v <= u:
                continue
            # Find common neighbors of u and v? No, we need path u - a - b - d - e - v
            # i.e., a neighbor a of u, then b neighbor of a (b != u, v), then d neighbor of b (d != u, a, v),
            # then e neighbor of d (e != u, a, b, v) and e neighbor of v.
            # This is expensive. Instead: for each edge (u,v), check if there exist
            # a in N(u), e in N(v), a != e, a != v, e != u, and a path of length 2 from a to e
            # (i.e., common neighbor b of a and e, b not in {u,v,a,e}).
            Nu = av[u]
            Nv = av[v]
            for a in Nu:
                if a == v:
                    continue
                for e in Nv:
                    if e == u or e == a:
                        continue
                    # need common neighbor b of a and e, b not in {u, v, a, e}
                    Na = av[a]
                    Ne = av[e]
                    # iterate over smaller set
                    if len(Na) < len(Ne):
                        for b in Na:
                            if b in (u, v, a, e):
                                continue
                            if b in Ne:
                                return True
                    else:
                        for b in Ne:
                            if b in (u, v, a, e):
                                continue
                            if b in Na:
                                return True
    return False

def _has_any_c6(colors, n):
    adjv = _build_adj(colors, n)
    for c in range(4):
        if _has_c6(adjv, c):
            return True
    return False

def _cyclic_coloring(n, offsets):
    """Color edge (i,j) by (j-i) mod n mapped through offsets to {0,1,2,3}.
    offsets is a dict from residue -> color, or a list of length n."""
    colors = [0] * (n * (n - 1) // 2)
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            # distance in [1, n-1]
            dist = d if d <= n // 2 else n - d
            # map dist to color via offsets
            colors[_edge_idx(i, j, n)] = offsets[dist]
    return colors

def _random_coloring(n):
    colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
    return colors

def _local_search(colors, n, max_iter=20000, seed=0):
    """Try to eliminate all monochromatic C6s by flipping edge colors."""
    rng = random.Random(seed)
    colors = list(colors)
    for it in range(max_iter):
        if _has_any_c6(colors, n):
            # find a C6 and flip a random edge in it
            adjv = _build_adj(colors, n)
            found = False
            for c in range(4):
                av = adjv[c]
                for u in av:
                    for v in av[u]:
                        if v <= u:
                            continue
                        Nu = av[u]
                        Nv = av[v]
                        for a in Nu:
                            if a == v:
                                continue
                            for e in Nv:
                                if e == u or e == a:
                                    continue
                            Na = av[a]
                            Ne = av[e]
                            if len(Na) < len(Ne):
                                for b in Na:
                                    if b in (u, v, a, e):
                                        continue
                                    if b in Ne:
                                        # found C6: u-a-b-e-v-u (in color c)
                                        # flip a random edge in the cycle
                                        edges = [(u, a), (a, b), (b, e), (e, v), (v, u)]
                                        ei = rng.choice(edges)
                                        idx = _edge_idx(min(ei[0], ei[1]), max(ei[0], ei[1]), n)
                                        colors[idx] = (colors[idx] + 1) % 4
                                        found = True
                                        break
                                if found:
                                    break
                            else:
                                for b in Ne:
                                    if b in (u, v, a, e):
                                        continue
                                    if b in Na:
                                        edges = [(u, a), (a, b), (b, e), (e, v), (v, u)]
                                        ei = rng.choice(edges)
                                        idx = _edge_idx(min(ei[0], ei[1]), max(ei[0], ei[1]), n)
                                        colors[idx] = (colors[idx] + 1) % 4
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                # no C6 found by search but _has_any_c6 said true; just break
                break
        else:
            return colors
    return colors

def construct():
    start = time.time()
    best_n = 0
    best_colors = None

    # Try to find colorings for n from high to low
    for n in range(17, 8, -1):
        if time.time() - start > DEADLINE:
            break
        found = False
        # Try cyclic constructions with various offset mappings
        for trial in range(200):
            if time.time() - start > DEADLINE:
                break
            # random mapping of distances 1..n//2 to colors
            half = n // 2
            offsets = {}
            for d in range(1, half + 1):
                offsets[d] = random.randint(0, 3)
            colors = _cyclic_coloring(n, offsets)
            if not _has_any_c6(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
            # try local search
            colors2 = _local_search(colors, n, max_iter=3000, seed=trial)
            if not _has_any_c6(colors2, n):
                best_n = n
                best_colors = colors2
                found = True
                break
        if found:
            break

    # If nothing found above, try smaller n with random + local search
    if best_n == 0:
        for n in range(12, 7, -1):
            if time.time() - start > DEADLINE:
                break
            found = False
            for trial in range(100):
                if time.time() - start > DEADLINE:
                    break
                colors = _random_coloring(n)
                colors2 = _local_search(colors, n, max_iter=5000, seed=trial)
                if not _has_any_c6(colors2, n):
                    best_n = n
                    best_colors = colors2
                    found = True
                    break
            if found:
                break

    # Fallback: the known K9 cyclic coloring
    if best_n == 0:
        n = 9
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]

    return {'n': best_n, 'colors': best_colors}
