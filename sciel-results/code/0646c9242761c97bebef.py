import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 138.0

    N = 16
    FULL_MASK = (1 << N) - 1
    ALL_TRIPLES = list(combinations(range(N), 3))
    ALL_TRIPLE_PAIRS = [((a, b), (a, c), (b, c)) for (a, b, c) in ALL_TRIPLES]
    ALL_TRIPLE_MASKS = [(1 << a) | (1 << b) | (1 << c) for (a, b, c) in ALL_TRIPLES]
    N_TRIPLES = len(ALL_TRIPLES)

    # Precompute for each point, the list of triple indices containing it
    POINT_TRIPLES = [[] for _ in range(N)]
    for i, (a, b, c) in enumerate(ALL_TRIPLES):
        POINT_TRIPLES[a].append(i)
        POINT_TRIPLES[b].append(i)
        POINT_TRIPLES[c].append(i)

    def time_check():
        return time.time() - start_time > DEADLINE

    def compute_alpha(triples):
        """Fast independence number using bitmask. Returns max size of independent set."""
        if not triples:
            return N
        n_t = len(triples)
        # Build list of triple masks
        t_masks = [0] * n_t
        for i, t in enumerate(triples):
            m = 0
            for p in t:
                m |= (1 << p)
            t_masks[i] = m

        # For each point, list of triple indices containing it
        pt = [[] for _ in range(N)]
        for i in range(n_t):
            m = t_masks[i]
            for p in range(N):
                if m & (1 << p):
                    pt[p].append(i)

        best = 0
        # Iterate all masks, find largest independent
        for mask in range(FULL_MASK, -1, -1):
            pc = bin(mask).count('1')
            if pc <= best:
                continue
            # Check if mask contains any triple
            indep = True
            for p in range(N):
                if mask & (1 << p):
                    for ti in pt[p]:
                        if (mask & t_masks[ti]) == t_masks[ti]:
                            indep = False
                            break
                    if not indep:
                        break
            if indep:
                best = pc
                if best >= 8:
                    break
        return best

    def greedy_construct(rng, bias='random'):
        """Greedy construction of linear triple system."""
        used_pairs = set()
        triples = []
        # Order triples
        if bias == 'random':
            order = list(range(N_TRIPLES))
            rng.shuffle(order)
        elif bias == 'balanced':
            # Sort by sum of point indices (favors low points first) then shuffle within groups
            order = list(range(N_TRIPLES))
            order.sort(key=lambda i: ALL_TRIPLES[i][0] + ALL_TRIPLES[i][1] + ALL_TRIPLES[i][2])
            rng.shuffle(order)
        else:
            order = list(range(N_TRIPLES))
            rng.shuffle(order)

        for i in order:
            a, b, c = ALL_TRIPLES[i]
            p1, p2, p3 = ALL_TRIPLE_PAIRS[i]
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
            triples.append(ALL_TRIPLES[i])
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
        return triples

    def local_search(triples, rng, max_iters=200, use_swaps=True, use_annealing=True):
        """Local search with add, swap, and simulated annealing."""
        if not triples:
            return triples

        current = list(triples)
        current_alpha = compute_alpha(current)
        if current_alpha <= 4:
            return current

        used_pairs = set()
        for t in current:
            a, b, c = t
            used_pairs.add((a, b))
            used_pairs.add((a, c))
            used_pairs.add((b, c))

        best_alpha = current_alpha
        best_triples = list(current)

        # Precompute all valid add candidates (triples that don't conflict)
        def get_add_candidates():
            cands = []
            for i in range(N_TRIPLES):
                t = ALL_TRIPLES[i]
                p1, p2, p3 = ALL_TRIPLE_PAIRS[i]
                if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                    cands.append(i)
            return cands

        def get_swap_candidates():
            """Return list of (remove_idx, add_idx) pairs."""
            swaps = []
            for ri in range(len(current)):
                rt = current[ri]
                rp1, rp2, rp3 = ALL_TRIPLE_PAIRS[ALL_TRIPLES.index(rt) if rt in ALL_TRIPLES else 0]
                # Temporarily remove
                temp_pairs = used_pairs - {rp1, rp2, rp3}
                for i in range(N_TRIPLES):
                    if i == ALL_TRIPLES.index(rt) if rt in ALL_TRIPLES else False:
                        continue
                    p1, p2, p3 = ALL_TRIPLE_PAIRS[i]
                    if p1 not in temp_pairs and p2 not in temp_pairs and p3 not in temp_pairs:
                        swaps.append((ri, i))
            return swaps

        # Simulated annealing parameters
        T = 2.0
        cooling = 0.995
        min_T = 0.01

        for iteration in range(max_iters):
            if time_check():
                break

            improved = False

            # Try adds
            add_cands = get_add_candidates()
            if add_cands:
                rng.shuffle(add_cands)
                for i in add_cands[:50]:
                    new_triples = current + [ALL_TRIPLES[i]]
                    new_alpha = compute_alpha(new_triples)
                    if new_alpha < current_alpha:
                        current = new_triples
                        current_alpha = new_alpha
                        p1, p2, p3 = ALL_TRIPLE_PAIRS[i]
                        used_pairs.add(p1)
                        used_pairs.add(p2)
                        used_pairs.add(p3)
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_triples = list(current)
                        improved = True
                        break
                    elif use_annealing and T > min_T and rng.random() < (1.0 - new_alpha) * T:
                        # Accept worse move (annealing)
                        current = new_triples
                        current_alpha = new_alpha
                        p1, p2, p3 = ALL_TRIPLE_PAIRS[i]
                        used_pairs.add(p1)
                        used_pairs.add(p2)
                        used_pairs.add(p3)
                        improved = True
                        break

            # Try swaps
            if use_swaps and not improved:
                # Sample a few swaps
                swap_cands = []
                for ri in range(min(len(current), 30)):
                    rt = current[ri]
                    idx = ALL_TRIPLES.index(rt)
                    rp1, rp2, rp3 = ALL_TRIPLE_PAIRS[idx]
                    temp_pairs = used_pairs - {rp1, rp2, rp3}
                    # Find up to 5 valid adds
                    for i in range(N_TRIPLES):
                        if i == idx:
                            continue
                        p1, p2, p3 = ALL_TRIPLE_PAIRS[i]
                        if p1 not in temp_pairs and p2 not in temp_pairs and p3 not in temp_pairs:
                            swap_cands.append((ri, i))
                            if len(swap_cands) >= 20:
                                break
                    if len(swap_cands) >= 20:
                        break

                if swap_cands:
                    rng.shuffle(swap_cands)
                    for (ri, ai) in swap_cands[:10]:
                        rt = current[ri]
                        idx = ALL_TRIPLES.index(rt)
                        rp1, rp2, rp3 = ALL_TRIPLE_PAIRS[idx]
                        used_pairs.discard(rp1)
                        used_pairs.discard(rp2)
                        used_pairs.discard(rp3)

                        new_triples = current[:ri] + current[ri+1:] + [ALL_TRIPLES[ai]]
                        new_alpha = compute_alpha(new_triples)

                        p1, p2, p3 = ALL_TRIPLE_PAIRS[ai]
                        used_pairs.add(p1)
                        used_pairs.add(p2)
                        used_pairs.add(p3)

                        if new_alpha < current_alpha:
                            current = new_triples
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_triples = list(current)
                            improved = True
                            break
                        elif use_annealing and T > min_T and rng.random() < (1.0 - new_alpha) * T:
                            current = new_triples
                            current_alpha = new_alpha
                            improved = True
                            break
                        else:
                            # Reject: restore
                            used_pairs.discard(p1)
                            used_pairs.discard(p2)
                            used_pairs.discard(p3)
                            used_pairs.add(rp1)
                            used_pairs.add(rp2)
                            used_pairs.add(rp3)

            if not improved:
                T *= cooling
                if T < min_T:
                    break

        return best_triples

    # Main search
    best_triples = None
    best_alpha = N

    # Phase 1: High-throughput greedy restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time_check():
            break
        rng = random.Random(i * 7919 + 42)
        bias = 'random' if i % 3 == 0 else ('balanced' if i % 3 == 1 else 'random')
        triples = greedy_construct(rng, bias)
        if not triples:
            continue
        alpha = compute_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if best_alpha <= 4:
                break

    # Phase 2: Local search on best and near-best
    if best_triples and best_alpha > 4 and not time_check():
        improved = local_search(best_triples, random.Random(12345), max_iters=300)
        imp_alpha = compute_alpha(improved)
        if imp_alpha < best_alpha:
            best_alpha = imp_alpha
            best_triples = improved

    # Phase 3: More restarts with local search
    for i in range(num_restarts, num_restarts + 200):
        if time_check():
            break
        rng = random.Random(i * 104729 + 7)
        triples = greedy_construct(rng, 'random')
        if not triples:
            continue
        alpha = compute_alpha(triples)
        if alpha <= best_alpha:
            improved = local_search(triples, random.Random(i), max_iters=100)
            imp_alpha = compute_alpha(improved)
            if imp_alpha < best_alpha:
                best_alpha = imp_alpha
                best_triples = improved
                if best_alpha <= 4:
                    break

    # Phase 4: Final polish on best
    if best_triples and best_alpha > 4 and not time_check():
        improved = local_search(best_triples, random.Random(99999), max_iters=500, use_annealing=True)
        imp_alpha = compute_alpha(improved)
        if imp_alpha < best_alpha:
            best_alpha = imp_alpha
            best_triples = improved

    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return best_triples
