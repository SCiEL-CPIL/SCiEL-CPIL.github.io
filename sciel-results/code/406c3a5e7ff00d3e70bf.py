import time
import random
import numpy as np
from itertools import combinations

# ---------- Bit-mask based K2,2,2 detection (from inspirations) ----------

def _popcount(x):
    return bin(x).count('1')

def _contains_K222(neigh, n):
    """
    Check if the graph defined by neighbor bitmasks contains K2,2,2.
    neigh[v] is a bitmask of neighbors of vertex v.
    Uses bit operations for efficiency.
    """
    for a in range(n):
        na = neigh[a]
        for b in range(a + 1, n):
            S = na & neigh[b]
            if _popcount(S) < 4:
                continue
            verts_in_S = []
            m = S
            while m:
                lsb = m & (-m)
                verts_in_S.append(lsb.bit_length() - 1)
                m ^= lsb
            L = len(verts_in_S)
            for i_c in range(L):
                c = verts_in_S[i_c]
                nc = neigh[c] & S
                for i_d in range(i_c + 1, L):
                    d = verts_in_S[i_d]
                    T = nc & neigh[d]
                    if _popcount(T) >= 2:
                        return True
    return False

def _count_K222(neigh, n):
    """
    Count the number of (unordered) K2,2,2 subgraphs in the graph.
    Each K2,2,2 corresponds to a unique partition of 6 vertices into 3 pairs,
    so we count each partition once.
    """
    total = 0
    for a in range(n):
        na = neigh[a]
        for b in range(a + 1, n):
            S = na & neigh[b]
            if _popcount(S) < 4:
                continue
            verts_in_S = []
            m = S
            while m:
                lsb = m & (-m)
                verts_in_S.append(lsb.bit_length() - 1)
                m ^= lsb
            L = len(verts_in_S)
            for i_c in range(L):
                c = verts_in_S[i_c]
                nc = neigh[c] & S
                for i_d in range(i_c + 1, L):
                    d = verts_in_S[i_d]
                    T = nc & neigh[d]
                    if _popcount(T) >= 2:
                        total += 1
    return total

# ---------- Color list <-> bitmasks conversion ----------

def _colors_to_neigh(colors, n):
    neigh0 = [0] * n
    neigh1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                neigh0[i] |= (1 << j)
                neigh0[j] |= (1 << i)
            else:
                neigh1[i] |= (1 << j)
                neigh1[j] |= (1 << i)
            idx += 1
    return neigh0, neigh1

def _total_violations(colors, n):
    neigh0, neigh1 = _colors_to_neigh(colors, n)
    return _count_K222(neigh0, n) + _count_K222(neigh1, n)

def _is_valid_full(colors, n):
    return _total_violations(colors, n) == 0

# ---------- Edge index <-> (i,j) ----------

def _edge_to_ij(idx, n):
    """Invert: idx = i*(i-1)//2 + (j-i-1) for i<j."""
    # Find i such that i*(i-1)//2 <= idx < (i+1)*i//2
    i = 1
    while True:
        base = i * (i - 1) // 2
        nxt = (i + 1) * i // 2
        if base <= idx < nxt:
            j = i + 1 + (idx - base)
            return i, j
        i += 1

# ---------- Fast delta computation (FIXED: uses pre-flip colors) ----------

def _delta_violations_flip(colors, n, idx):
    """
    Compute the change in total K2,2,2 violations when flipping edge at index idx.
    Returns delta = new_total - old_total.
    IMPORTANT: uses the PRE-FLIP colors (the input 'colors') to evaluate both
    the destroyed (old_color) and created (new_color) subgraphs.
    """
    i, j = _edge_to_ij(idx, n)
    old_color = colors[idx]
    new_color = 1 - old_color
    ij = (min(i, j), max(i, j))

    other_vertices = [v for v in range(n) if v != i and v != j]
    delta = 0

    for four_combo in combinations(other_vertices, 4):
        six_set = tuple(sorted([i, j] + list(four_combo)))
        all_vertices = list(six_set)
        i_pos = all_vertices.index(i)
        j_pos = all_vertices.index(j)

        # Enumerate all 15 perfect matchings (partitions into 3 pairs) of the 6 vertices
        matchings = []
        for k in range(6):
            if k == i_pos:
                continue
            rest = [v for idx_v, v in enumerate(all_vertices) if idx_v != i_pos and idx_v != k]
            r = rest
            matchings.append(((all_vertices[i_pos], all_vertices[k]),
                              (r[0], r[1]), (r[2], r[3])))
            matchings.append(((all_vertices[i_pos], all_vertices[k]),
                              (r[0], r[2]), (r[1], r[3])))
            matchings.append(((all_vertices[i_pos], all_vertices[k]),
                              (r[0], r[3]), (r[1], r[2])))

        for p1, p2, p3 in matchings:
            # Edge (i,j) must be a cross-edge, i.e. i and j in different parts
            same = ((i in p1 and j in p1) or (i in p2 and j in p2) or (i in p3 and j in p3))
            if same:
                continue
            # Build the 12 cross-edges
            edges = []
            for a in p1:
                for b in p2:
                    edges.append((min(a, b), max(a, b)))
                for b in p3:
                    edges.append((min(a, b), max(a, b)))
            for a in p2:
                for b in p3:
                    edges.append((min(a, b), max(a, b)))
            if ij not in edges:
                continue
            other_edges = [e for e in edges if e != ij]

            # Destroyed: all 11 other edges in old_color (pre-flip state)
            all_old = True
            for u, v in other_edges:
                base = u * (u - 1) // 2
                e_idx = base + (v - u - 1)
                if colors[e_idx] != old_color:
                    all_old = False
                    break
            if all_old:
                delta -= 1

            # Created: all 11 other edges in new_color (pre-flip state)
            all_new = True
            for u, v in other_edges:
                base = u * (u - 1) // 2
                e_idx = base + (v - u - 1)
                if colors[e_idx] != new_color:
                    all_new = False
                    break
            if all_new:
                delta += 1

    return delta

# ---------- Paley construction ----------

def _build_paley(p):
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin within 150s

    # Start with the known-valid Paley 29 coloring
    p = 29
    n = p
    colors = _build_paley(p)

    if not _is_valid_full(colors, n):
        # Should never happen, but be safe
        return {'n': n, 'colors': colors}

    # Try to extend to 30 vertices via simulated annealing
    new_n = n + 1
    best_colors = None
    best_energy = float('inf')

    # Precompute the list of 4-vertex subsets of the 28 "other" vertices for speed
    base_vertices = list(range(n))  # 0..28
    four_subsets = list(combinations(base_vertices, 4))

    def _delta_fast(cur_colors, idx):
        """Delta using the same logic but with the precomputed 4-subsets."""
        i, j = _edge_to_ij(idx, new_n)
        old_color = cur_colors[idx]
        new_color = 1 - old_color
        ij = (min(i, j), max(i, j))
        delta = 0
        for four_combo in four_subsets:
            # Skip if i or j is in four_combo (shouldn't happen since i,j <= 28 and
            # four_combo is from 0..28; but if i or j is the new vertex 29, it's not in
            # base_vertices, so fine. If i or j <= 28, four_combo may contain them.)
            if i in four_combo or j in four_combo:
                continue
            six_set = tuple(sorted([i, j] + list(four_combo)))
            all_vertices = list(six_set)
            i_pos = all_vertices.index(i)
            j_pos = all_vertices.index(j)
            matchings = []
            for k in range(6):
                if k == i_pos:
                    continue
                rest = [v for idx_v, v in enumerate(all_vertices) if idx_v != i_pos and idx_v != k]
                r = rest
                matchings.append(((all_vertices[i_pos], all_vertices[k]),
                                  (r[0], r[1]), (r[2], r[3])))
                matchings.append(((all_vertices[i_pos], all_vertices[k]),
                                  (r[0], r[2]), (r[1], r[3])))
                matchings.append(((all_vertices[i_pos], all_vertices[k]),
                                  (r[0], r[3]), (r[1], r[2])))
            for p1, p2, p3 in matchings:
                same = ((i in p1 and j in p1) or (i in p2 and j in p2) or (i in p3 and j in p3))
                if same:
                    continue
                edges = []
                for a in p1:
                    for b in p2:
                        edges.append((min(a, b), max(a, b)))
                    for b in p3:
                        edges.append((min(a, b), max(a, b)))
                for a in p2:
                    for b in p3:
                        edges.append((min(a, b), max(a, b)))
                if ij not in edges:
                    continue
                other_edges = [e for e in edges if e != ij]
                all_old = True
                for u, v in other_edges:
                    base = u * (u - 1) // 2
                    e_idx = base + (v - u - 1)
                    if cur_colors[e_idx] != old_color:
                        all_old = False
                        break
                if all_old:
                    delta -= 1
                all_new = True
                for u, v in other_edges:
                    base = u * (u - 1) // 2
                    e_idx = base + (v - u - 1)
                    if cur_colors[e_idx] != new_color:
                        all_new = False
                        break
                if all_new:
                    delta += 1
        return delta

    num_restarts = 60
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Initialize: copy 29-vertex coloring, add random edges for vertex 29
        cur_colors = colors[:]
        v_new = new_n - 1
        for i in range(v_new):
            cur_colors.append(random.randint(0, 1))

        cur_energy = _total_violations(cur_colors, new_n)

        if cur_energy == 0:
            best_colors = cur_colors[:]
            best_energy = 0
            break

        T = 2.0
        T_min = 0.01
        cooling_rate = 0.99
        max_steps = 4000

        best_cur_colors = cur_colors[:]
        best_cur_energy = cur_energy

        for step in range(max_steps):
            if time.time() > deadline - 2:
                break

            idx = random.randint(0, len(cur_colors) - 1)
            cur_colors[idx] = 1 - cur_colors[idx]

            # Compute delta using PRE-flip colors: temporarily revert to get pre-flip state
            # Actually _delta_fast uses cur_colors which is now POST-flip. We need pre-flip.
            # So revert, compute delta, then re-apply.
            cur_colors[idx] = 1 - cur_colors[idx]  # back to pre-flip
            delta = _delta_fast(cur_colors, idx)
            cur_colors[idx] = 1 - cur_colors[idx]  # apply flip again

            new_energy = cur_energy + delta

            if new_energy < cur_energy:
                cur_energy = new_energy
                if cur_energy < best_cur_energy:
                    best_cur_energy = cur_energy
                    best_cur_colors = cur_colors[:]
                    if cur_energy == 0:
                        break
            else:
                delta_e = new_energy - cur_energy
                if delta_e > 0 and random.random() < np.exp(-delta_e / T):
                    cur_energy = new_energy
                else:
                    cur_colors[idx] = 1 - cur_colors[idx]

            T *= cooling_rate
            if T < T_min:
                T = T_min

        if best_cur_energy < best_energy:
            best_energy = best_cur_energy
            best_colors = best_cur_colors[:]

        if best_energy == 0:
            break

    if best_colors is not None and best_energy == 0:
        if _is_valid_full(best_colors, new_n):
            colors = best_colors
            n = new_n

    # Final validation
    if not _is_valid_full(colors, n):
        colors = _build_paley(29)
        n = 29

    return {'n': n, 'colors': colors}
