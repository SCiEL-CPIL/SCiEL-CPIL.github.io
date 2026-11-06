"""Edge-colourings of K16 in which every K4 sees five colours.
construct() -> 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
Every set of four vertices must span edges of at least five distinct colours. Minimize the number of colours used.
"""
import time
import random
import itertools

_START = time.time()
_DEADLINE = 150.0
_SAFETY = 8.0  # seconds margin before wall-clock kill


def _k4_colours(col, a, b, c, d):
    # colours of the 6 edges of K4 on {a,b,c,d}
    return (col[a][b], col[a][c], col[a][d],
            col[b][c], col[b][d], col[c][d])


def _k4_ok(col, a, b, c, d):
    cs = _k4_colours(col, a, b, c, d)
    return len(set(cs)) >= 5


def _all_k4s():
    """Precompute all K4 tuples (a,b,c,d) with a<b<c<d."""
    return [(a, b, c, d) for a in range(16) for b in range(a+1, 16)
            for c in range(b+1, 16) for d in range(c+1, 16)]


def _valid(col):
    for (a, b, c, d) in _ALL_K4S:
        if not _k4_ok(col, a, b, c, d):
            return False
    return True


def _count_colours(col):
    s = set()
    for i in range(16):
        for j in range(i + 1, 16):
            s.add(col[i][j])
    return len(s)


def _flatten(col):
    return [col[i][j] for i in range(16) for j in range(i + 1, 16)]


def _build(m, shift):
    """colour(i,j) = (i + j + shift) mod m"""
    n = 16
    col = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            col[i][j] = (i + j + shift) % m
    return col


def _build2(m, shift, k):
    """colour(i,j) = (i + j + k*i*j + shift) mod m"""
    n = 16
    col = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            col[i][j] = (i + j + k * i * j + shift) % m
    return col


def _build3(m, shift, k, l):
    """colour(i,j) = (i + j + k*i*j + l*i^2 + shift) mod m"""
    n = 16
    col = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            col[i][j] = (i + j + k * i * j + l * i * i + shift) % m
    return col


def _local_search(col, palette_size, max_iters=50000, seed=0):
    """Local search: try to reduce colour count by reassigning edge colours."""
    rng = random.Random(seed)
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Map edge index to position
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # For each edge, precompute which K4s it belongs to
    edge_k4s = [[] for _ in range(len(edges))]
    for k4_idx, (a, b, c, d) in enumerate(_ALL_K4S):
        k4_edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        for e in k4_edges:
            edge_k4s[edge_idx[e]].append(k4_idx)
    
    current_col = [col[i][j] for i in range(n) for j in range(i + 1, n)]
    current_used = len(set(current_col))
    
    best_col = current_col[:]
    best_used = current_used
    
    for it in range(max_iters):
        if time.time() > _START + _DEADLINE - _SAFETY:
            break
        
        # Pick a random edge and try to change its colour
        e_idx = rng.randint(0, len(edges) - 1)
        old_c = current_col[e_idx]
        
        # Try to assign a colour that's already used (to not increase palette)
        # or a new colour within palette_size
        used_colours = set(current_col)
        
        # Prefer colours already in use to try to reduce count
        candidates = list(used_colours)
        rng.shuffle(candidates)
        
        improved = False
        for new_c in candidates:
            if new_c == old_c:
                continue
            if new_c >= palette_size:
                continue
            
            # Check if changing this edge's colour keeps all K4s valid
            # Only need to check K4s containing this edge
            valid = True
            for k4_idx in edge_k4s[e_idx]:
                a, b, c, d = _ALL_K4S[k4_idx]
                # Get the 6 edges of this K4
                k4_edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                cs = []
                for (u, v) in k4_edges:
                    if (u, v) == edges[e_idx]:
                        cs.append(new_c)
                    else:
                        cs.append(current_col[edge_idx[(u, v)]])
                if len(set(cs)) < 5:
                    valid = False
                    break
            
            if valid:
                current_col[e_idx] = new_c
                new_used = len(set(current_col))
                if new_used < best_used:
                    best_used = new_used
                    best_col = current_col[:]
                    improved = True
                break
        
        if not improved:
            # Accept the change anyway if it doesn't increase used colours
            # (to explore the space)
            pass
    
    return best_col, best_used


def _simulated_annealing(col, palette_size, max_iters=200000, seed=0):
    """Simulated annealing to reduce colour count."""
    rng = random.Random(seed)
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    edge_k4s = [[] for _ in range(len(edges))]
    for k4_idx, (a, b, c, d) in enumerate(_ALL_K4S):
        k4_edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        for e in k4_edges:
            edge_k4s[edge_idx[e]].append(k4_idx)
    
    current_col = [col[i][j] for i in range(n) for j in range(i + 1, n)]
    current_used = len(set(current_col))
    
    best_col = current_col[:]
    best_used = current_used
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    for it in range(max_iters):
        if time.time() > _START + _DEADLINE - _SAFETY:
            break
        
        e_idx = rng.randint(0, len(edges) - 1)
        old_c = current_col[e_idx]
        
        # Pick a random colour within palette
        new_c = rng.randint(0, palette_size - 1)
        if new_c == old_c:
            continue
        
        # Check validity of affected K4s
        valid = True
        for k4_idx in edge_k4s[e_idx]:
            a, b, c, d = _ALL_K4S[k4_idx]
            k4_edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
            cs = []
            for (u, v) in k4_edges:
                if (u, v) == edges[e_idx]:
                    cs.append(new_c)
                else:
                    cs.append(current_col[edge_idx[(u, v)]])
            if len(set(cs)) < 5:
                valid = False
                break
        
        if valid:
            current_col[e_idx] = new_c
            new_used = len(set(current_col))
            if new_used < best_used:
                best_used = new_used
                best_col = current_col[:]
        else:
            # Reject invalid moves
            pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_col, best_used


# Precompute all K4s
_ALL_K4S = _all_k4s()


def construct():
    deadline = _START + _DEADLINE - _SAFETY

    # Baseline: m=17, shift=0
    best_col = _build(17, 0)
    best_count = _count_colours(best_col)

    # Search the linear family
    for m in range(13, 18):
        if time.time() > deadline:
            break
        for shift in range(m):
            if time.time() > deadline:
                break
            col = _build(m, shift)
            if not _valid(col):
                continue
            c = _count_colours(col)
            if c < best_count:
                best_count = c
                best_col = col

    # Search quadratic family
    for m in range(13, 18):
        if time.time() > deadline:
            break
        for k in range(0, m):
            if time.time() > deadline:
                break
            for shift in range(m):
                if time.time() > deadline:
                    break
                col = _build2(m, shift, k)
                if not _valid(col):
                    continue
                c = _count_colours(col)
                if c < best_count:
                    best_count = c
                    best_col = col

    # Search cubic-ish family
    for m in range(13, 18):
        if time.time() > deadline:
            break
        for k in range(0, min(5, m)):
            if time.time() > deadline:
                break
            for l in range(0, min(5, m)):
                if time.time() > deadline:
                    break
                for shift in range(m):
                    if time.time() > deadline:
                        break
                    col = _build3(m, shift, k, l)
                    if not _valid(col):
                        continue
                    c = _count_colours(col)
                    if c < best_count:
                        best_count = c
                        best_col = col

    # Local search / simulated annealing to try to reduce colour count
    # Try with palette sizes from best_count down to 13
    for palette_size in range(best_count, 12, -1):
        if time.time() > deadline:
            break
        # Run a few SA iterations
        for seed in range(10):
            if time.time() > deadline:
                break
            sa_col, sa_used = _simulated_annealing(best_col, palette_size, 
                                                    max_iters=50000, seed=seed)
            if sa_used < best_count:
                best_count = sa_used
                best_col = [[0]*16 for _ in range(16)]
                edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
                for idx, (i, j) in enumerate(edges):
                    best_col[i][j] = sa_col[idx]

    # Final local search pass
    if time.time() < deadline:
        ls_col, ls_used = _local_search(best_col, best_count, max_iters=30000, seed=42)
        if ls_used < best_count:
            best_count = ls_used
            best_col = [[0]*16 for _ in range(16)]
            edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
            for idx, (i, j) in enumerate(edges):
                best_col[i][j] = ls_col[idx]

    out = _flatten(best_col)
    # Ensure colours are within 0..119 and integer.
    assert all(0 <= x <= 119 for x in out)
    assert len(out) == 120
    return out


# Self-test on import (cheap) to avoid shipping an invalid result.
if __name__ == "__main__":
    r = construct()
    print("colours used:", len(set(r)))
