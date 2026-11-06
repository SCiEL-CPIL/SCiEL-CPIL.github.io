"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    num_quads = list(itertools.combinations(range(n), 4))
    n_quads = len(num_quads)  # 1820

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    # Precompute quad edge indices
    quad_edge_indices = np.zeros((n_quads, 6), dtype=np.int32)
    for qi, (a, b, c, d) in enumerate(num_quads):
        quad_edge_indices[qi] = [edge_idx(a,b), edge_idx(a,c), edge_idx(a,d),
                                 edge_idx(b,c), edge_idx(b,d), edge_idx(c,d)]

    # For each edge, list of quads containing it
    edge_to_quads = [[] for _ in range(num_edges)]
    for qi in range(n_quads):
        for pos in range(6):
            ei = int(quad_edge_indices[qi, pos])
            edge_to_quads[ei].append(qi)

    def count_colors(coloring):
        return max(coloring) + 1

    def build_ec(coloring):
        ec = np.zeros((n, n), dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                ec[i, j] = coloring[idx]
                ec[j, i] = coloring[idx]
                idx += 1
        return ec

    def compute_all_violations(coloring):
        ec = build_ec(coloring)
        total = 0
        for qi in range(n_quads):
            a, b, c, d = num_quads[qi]
            colors = [ec[a,b], ec[a,c], ec[a,d], ec[b,c], ec[b,d], ec[c,d]]
            if len(set(colors)) < 5:
                total += 1
        return total

    def is_valid(coloring):
        ec = build_ec(coloring)
        for qi in range(n_quads):
            a, b, c, d = num_quads[qi]
            colors = [ec[a,b], ec[a,c], ec[a,d], ec[b,c], ec[b,d], ec[c,d]]
            if len(set(colors)) < 5:
                return False
        return True

    def make_coloring_formula(formula_fn, n=16):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors

    def local_search(coloring, max_iter=50, n=16, use_random=False):
        """
        Delta-based hill climbing with optional random restarts.
        """
        coloring = np.array(coloring, dtype=np.int32)
        current_violations = compute_all_violations(coloring)
        if current_violations == 0:
            return coloring.tolist(), 0

        best_violations = current_violations
        best_coloring = coloring.copy()
        no_improve = 0
        max_no_improve = 20
        max_colors = int(coloring.max()) + 1

        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False

            # Optionally shuffle edge order for diversity
            edge_order = list(range(num_edges))
            if use_random:
                random.shuffle(edge_order)

            for idx in edge_order:
                if time.time() > deadline:
                    break
                old_color = int(coloring[idx])
                affected = edge_to_quads[idx]

                for new_color in range(max_colors):
                    if new_color == old_color:
                        continue

                    # Compute violation delta for affected quads
                    delta = 0
                    for qi in affected:
                        eidxs = quad_edge_indices[qi]
                        # Build color list with old color for this edge
                        old_colors = [int(coloring[e]) for e in eidxs]
                        old_colors[idx] = old_color
                        old_viol = 1 if len(set(old_colors)) < 5 else 0

                        new_colors = [int(coloring[e]) for e in eidxs]
                        new_colors[idx] = new_color
                        new_viol = 1 if len(set(new_colors)) < 5 else 0

                        delta += new_viol - old_viol

                    candidate_violations = best_violations + delta
                    if candidate_violations < best_violations:
                        coloring[idx] = new_color
                        best_violations = candidate_violations
                        best_coloring = coloring.copy()
                        improved = True
                        no_improve = 0
                        if new_color >= max_colors:
                            max_colors = new_color + 1
                    else:
                        coloring[idx] = old_color

                    if time.time() > deadline:
                        break

                if time.time() > deadline:
                    break

            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
            else:
                coloring = best_coloring.copy()

        return best_coloring.tolist(), best_violations

    # Generate candidates
    candidates = []

    # Linear forms: (a*i + b*j) mod m
    for mod in range(13, 20):
        for a in range(1, 9):
            for b in range(1, 9):
                if a == b:
                    continue
                cand = make_coloring_formula(lambda i, j, a=a, b=b, mod=mod: (a*i + b*j) % mod)
                candidates.append(cand)

    # Non-linear: (a*i*i + b*i*j + c*j*j) mod m
    for mod in range(13, 19):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    if a == b == c:
                        continue
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: (a*i*i + b*i*j + c*j*j) % mod
                    )
                    candidates.append(cand)

    # XOR-based
    for mod in range(13, 19):
        for k in range(1, 6):
            cand = make_coloring_formula(
                lambda i, j, k=k, mod=mod: ((i ^ j) * k) % mod
            )
            candidates.append(cand)

    # Product-based
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j) % mod)
        candidates.append(cand)

    # Mixed: (a*i + b*j + c*i*j) mod m
    for mod in range(13, 18):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 4):
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod: (a*i + b*j + c*i*j) % mod
                    )
                    candidates.append(cand)

    # Difference-based
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i - j)**2) % mod)
        candidates.append(cand)

    # Sum of squares
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i + j*j) % mod)
        candidates.append(cand)

    # Polynomial in (i-j)
    for mod in range(13, 19):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    cand = make_coloring_formula(
                        lambda i, j, a=a, b=b, c=c, mod=mod:
                        (a*(i-j) + b*(i-j)**2 + c*(i-j)**3) % mod
                    )
                    candidates.append(cand)

    # Full quadratic
    for mod in range(13, 17):
        for a in range(1, 4):
            for b in range(1, 4):
                for c in range(1, 3):
                    for d in range(1, 3):
                        for e in range(1, 3):
                            cand = make_coloring_formula(
                                lambda i, j, a=a, b=b, c=c, d=d, e=e, mod=mod:
                                (a*i + b*j + c*i*j + d*i*i + e*j*j) % mod
                            )
                            candidates.append(cand)

    # Bitwise operations
    for mod in range(13, 19):
        for k in range(1, 5):
            cand = make_coloring_formula(
                lambda i, j, k=k, mod=mod: ((i & j) * k + (i | j)) % mod
            )
            candidates.append(cand)
        for k in range(1, 5):
            cand = make_coloring_formula(
                lambda i, j, k=k, mod=mod: ((i ^ j) + (i & j) * k) % mod
            )
            candidates.append(cand)

    # (i + j)^2 mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i + j)**2 % mod)
        candidates.append(cand)

    # (i * j + i + j) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j + i + j) % mod)
        candidates.append(cand)

    # (i * (j - i)) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * (j - i)) % mod)
        candidates.append(cand)

    # (j * (j - i)) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (j * (j - i)) % mod)
        candidates.append(cand)

    # (i * j * (i + j)) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j * (i + j)) % mod)
        candidates.append(cand)

    # (i^3 + j^3) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i**3 + j**3) % mod)
        candidates.append(cand)

    # (i*j + i^2 + j^2) mod m
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*j + i*i + j*j) % mod)
        candidates.append(cand)

    # Additional diverse candidates
    for mod in range(13, 19):
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i * j) ^ (i + j)) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: ((i << 1) | j) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * 7 + j * 3) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * 11 + j * 5) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * 13 + j * 7) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * 17 + j * 11) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i * j + i * (j - i)) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: ((j - i) * (i + j)) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i**3 + i*j + j**3) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*j*(i-j)) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i + i*j + j*j) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*i - i*j + j*j) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*j + (i-j)**2) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i*j - (i-j)**2) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: (i**3 + j**3 + i*j*(i+j)) % mod)
        candidates.append(cand)
        cand = make_coloring_formula(lambda i, j, mod=mod: ((j-i)**2 * (i+j)) % mod)
        candidates.append(cand)

    # Find the best valid candidate
    best = None
    best_num_colors = float('inf')

    # Sort candidates by number of colors used (ascending)
    candidates.sort(key=lambda c: count_colors(c))

    # First pass: check candidates with <= 15 colors
    for cand in candidates:
        if time.time() > deadline:
            break
        nc = count_colors(cand)
        if nc >= best_num_colors:
            break
        if nc > 15:
            break
        if is_valid(cand):
            best_num_colors = nc
            best = cand

    # If we found a valid one with <= 15 colors, try local search
    if best is not None and best_num_colors <= 15:
        improved, violations = local_search(best, max_iter=50, n=16)
        if violations == 0:
            inc = count_colors(improved)
            if inc < best_num_colors:
                best = improved
                best_num_colors = inc

    # If no valid candidate with <= 15 found, try local search on near-miss candidates
    if best is None:
        best_violation_count = float('inf')
        best_near_miss = None
        for cand in candidates:
            if time.time() > deadline:
                break
            nc = count_colors(cand)
            if nc > 16:
                break
            v = compute_all_violations(cand)
            if v < best_violation_count:
                best_violation_count = v
                best_near_miss = cand
            if best_violation_count == 0:
                break

        if best_near_miss is not None:
            repaired, violations = local_search(best_near_miss, max_iter=100, n=16)
            if violations == 0:
                best = repaired
                best_num_colors = count_colors(repaired)

    if best is not None:
        return best

    # Fallback: return a known valid coloring
    return make_coloring_formula(lambda i, j: (i + j) % 17)

# EVOLVE-BLOCK-END
