import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = t0 + 118.0

    n = 11
    num_colorings = 1 << n  # 2048
    full_mask = (1 << num_colorings) - 1

    # All 5-subsets of vertices 0..10
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)  # 462

    # Precompute for each edge: which colorings it "kills" (makes monochromatic)
    # Use bitmasks of size num_colorings
    edge_masks = [0] * num_edges
    for i, e in enumerate(all_edges):
        mask_vs = 0
        for v in e:
            mask_vs |= (1 << v)
        mask = 0
        for c in range(num_colorings):
            bits = c & mask_vs
            if bits == 0 or bits == mask_vs:
                mask |= (1 << c)
        edge_masks[i] = mask

    best_count = num_edges
    best_set = set(range(num_edges))

    def try_cover(seed_idx):
        nonlocal best_count, best_set
        rng = random.Random(seed_idx)
        order = list(range(num_edges))
        rng.shuffle(order)

        chosen = set()
        uncovered = full_mask
        remaining = set(order)

        while uncovered:
            if time.time() > deadline:
                return
            # Find the edge covering the most uncovered colorings
            best_e = -1
            best_c = -1
            for idx in remaining:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > best_c:
                    best_c = c
                    best_e = idx
            if best_e < 0:
                break
            chosen.add(best_e)
            uncovered &= ~edge_masks[best_e]
            remaining.discard(best_e)

        # Local search: try to remove edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for e in list(chosen):
                # Check if removing e still covers all
                # i.e., for every coloring killed by e, some other edge in chosen also kills it
                ok = True
                mask_e = edge_masks[e]
                # Quick check: for each bit set in mask_e, is it also in union of others?
                others = 0
                for e2 in chosen:
                    if e2 != e:
                        others |= edge_masks[e2]
                if (mask_e & ~others) != 0:
                    ok = False
                if ok:
                    chosen.discard(e)
                    improved = True
                    break

        if len(chosen) < best_count:
            best_count = len(chosen)
            best_set = chosen.copy()
            if best_count <= 51:
                return

    # Try several random seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        try_cover(seed)
        if best_count <= 51:
            break

    # Fallback: deterministic greedy
    if best_count == num_edges:
        chosen = set()
        uncovered = full_mask
        remaining = set(range(num_edges))
        while uncovered:
            best_e = -1
            best_c = -1
            for idx in remaining:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > best_c:
                    best_c = c
                    best_e = idx
            if best_e < 0:
                break
            chosen.add(best_e)
            uncovered &= ~edge_masks[best_e]
            remaining.discard(best_e)
        best_set = chosen
        best_count = len(chosen)

    result_edges = [list(all_edges[i]) for i in sorted(best_set)]
    return {'n': n, 'edges': result_edges}
