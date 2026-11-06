"""Branch-and-bound with incremental bad-pair tracking for max sunflower-free family."""
import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 80.0

    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    N = len(all_sets)
    masks = [0] * N
    for i, s in enumerate(all_sets):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        masks[i] = m

    # Find all sunflower triples
    triples = []
    for i in range(N):
        mi = masks[i]
        for j in range(i + 1, N):
            mij = mi & masks[j]
            for k in range(j + 1, N):
                mik = mi & masks[k]
                if mik != mij:
                    continue
                if (masks[j] & masks[k]) == mij:
                    triples.append((i, j, k))

    # For each vertex v, precompute sunflower_partners[v] = set of (u, w) pairs
    # such that (u, v, w) is a sunflower. Store as list of (u, w) tuples.
    # When v is in family, for any other vertex x, x is "blocked" if there exists
    # y in family such that (x, y, v) is a sunflower, i.e., (y, x) in sunflower_partners[v]
    # or equivalently, x is in the set of vertices that form sunflower with v and some family member.
    
    # Better: for each v, precompute conflict_with[v][y] = set of x such that (x, y, v) is sunflower.
    # But that's a 3D structure. Instead:
    # For each v, bad_pairs[v] = set of frozenset({y, x}) such that (y, x, v) is sunflower.
    # Then v is compatible with family iff no pair in family is in bad_pairs[v].
    
    bad_pairs = [set() for _ in range(N)]
    for (i, j, k) in triples:
        bad_pairs[i].add(frozenset((j, k)))
        bad_pairs[j].add(frozenset((i, k)))
        bad_pairs[k].add(frozenset((i, j)))

    # Greedy with incremental tracking
    def greedy_seed(seed_val):
        rng = random.Random(seed_val)
        order = list(range(N))
        rng.shuffle(order)
        family = set()
        # bad_count[v] = number of pairs (a,b) in family such that frozenset({a,b}) in bad_pairs[v]
        bad_count = [0] * N
        for v in order:
            if bad_count[v] == 0:
                family.add(v)
                # Update bad_count for all other vertices
                # For each u not in family, if (v, y, u) is sunflower for some y in family,
                # then bad_count[u] increases.
                # Equivalently: for each u, bad_count[u] += number of y in family (y != v)
                # such that frozenset({v, y}) in bad_pairs[u].
                # Precompute: for each v, partners[v] = set of u such that there exists y
                # with (v, y, u) sunflower... no, we need it conditional on y.
                # 
                # Alternative: for each v, precompute adj[v] = set of (y, u) pairs where
                # (v, y, u) is a sunflower. Then when adding v to family, for each (y, u) in adj[v],
                # if y is in family, then bad_count[u] += 1.
                # But adj[v] could be large.
                #
                # Simpler: just recompute bad_count for all u by checking pairs in family.
                # This is O(N * |family|^2) per addition = O(N^3) total. Too slow.
                #
                # Let me precompute: for each v, adj_pairs[v] = list of (y, u) where (v,y,u) sunflower.
                # When adding v: for each (y, u) in adj_pairs[v]: if y in family: bad_count[u] += 1
                # Precompute adj_pairs:
                pass
        return family

    # Precompute adj_pairs: for each v, list of (y, u) such that (v, y, u) is a sunflower triple
    adj_pairs = [[] for _ in range(N)]
    for (i, j, k) in triples:
        # (i, j, k) is sunflower. So:
        # For vertex i: partners are (j, k) meaning if j is in family, k gets blocked
        adj_pairs[i].append((j, k))
        adj_pairs[i].append((k, j))
        adj_pairs[j].append((i, k))
        adj_pairs[j].append((k, i))
        adj_pairs[k].append((i, j))
        adj_pairs[k].append((j, i))

    def greedy_seed2(seed_val):
        rng = random.Random(seed_val)
        order = list(range(N))
        rng.shuffle(order)
        family = set()
        bad_count = [0] * N
        for v in order:
            if bad_count[v] == 0:
                family.add(v)
                # Update bad_count
                for (y, u) in adj_pairs[v]:
                    if y in family and u not in family:
                        bad_count[u] += 1
        return family

    # Get best greedy solution
    init_family = set()
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        fam = greedy_seed2(seed)
        if len(fam) > len(init_family):
            init_family = fam

    best_size = len(init_family)
    best_solution = set(init_family)

    # ---- Local search: try to swap out one vertex for another ----
    def can_add(v, family, bad_count):
        return bad_count[v] == 0

    # Local search with swaps
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        fam_list = list(best_solution)
        for idx in range(len(fam_list)):
            if time.time() > deadline - 10:
                break
            v_out = fam_list[idx]
            # Remove v_out
            temp_family = set(best_solution)
            temp_family.remove(v_out)
            # Recompute bad_count
            bad_count = [0] * N
            for v in temp_family:
                for (y, u) in adj_pairs[v]:
                    if y in temp_family and u not in temp_family:
                        bad_count[u] += 1
            # Try to add a vertex not in temp_family
            for v_in in range(N):
                if v_in in temp_family:
                    continue
                if bad_count[v_in] == 0:
                    # Add it
                    best_solution = temp_family | {v_in}
                    best_size = len(best_solution)
                    improved = True
                    break
            if improved:
                break

    # Final B&B attempt if time remains
    if time.time() < deadline - 5:
        # Simple B&B with the improved bound
        def branch_bnb(candidates, current, current_size):
            nonlocal best_size, best_solution
            if time.time() > deadline:
                return
            if not candidates:
                if current_size > best_size:
                    best_size = current_size
                    best_solution = set(current)
                return
            # Upper bound: current_size + len(candidates) (trivial)
            # Better: greedy on candidates
            # Greedy on candidates using bad_count logic
            cand_list = list(candidates)
            temp_bad = [0] * N
            temp_picked = set()
            for v in cand_list:
                if v not in candidates:
                    continue
                if temp_bad[v] == 0:
                    temp_picked.add(v)
                    for (y, u) in adj_pairs[v]:
                        if y in temp_picked and u in candidates and u not in temp_picked:
                            temp_bad[u] += 1
            ub = current_size + len(temp_picked)
            if ub <= best_size:
                return
            pivot = min(candidates)
            # Include pivot
            new_current = set(current)
            new_current.add(pivot)
            # New candidates: those with bad_count 0 w.r.t. new_current
            new_cands = set()
            for v in candidates:
                if v == pivot:
                    continue
                # Check if v compatible with new_current
                ok = True
                for (y, u) in adj_pairs[pivot]:
                    if y in new_current and u == v:
                        ok = False
                        break
                # Also need to check pairs within new_current
                nc = list(new_current)
                for i in range(len(nc)):
                    for j in range(i + 1, len(nc)):
                        if frozenset((nc[i], nc[j])) in bad_pairs[v]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    new_cands.add(v)
            branch_bnb(new_cands, new_current, current_size + 1)
            # Exclude pivot
            branch_bnb(candidates - {pivot}, current, current_size)

        branch_bnb(set(range(N)), set(), 0)

    result = [sorted(all_sets[v]) for v in sorted(best_solution)]
    return result
