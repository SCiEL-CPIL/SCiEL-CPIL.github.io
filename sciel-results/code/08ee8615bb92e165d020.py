import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin (evaluator kills at 150s)

    N = 16

    def check_alpha(triples):
        """
        Returns the independence number of the hypergraph.
        Precompute a 2^16 table: bad[mask] = True iff mask contains some triple.
        Then alpha = max k such that some mask of popcount k has bad[mask] == False.
        """
        if not triples:
            return N

        bad = bytearray(1 << N)  # 65536 bytes
        for t in triples:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            # Mark m and all supersets of m as bad
            # Supersets of m: m | x where x is subset of ~m (within N bits)
            full = (1 << N) - 1
            rest = full ^ m
            # iterate all submasks of rest
            sub = rest
            while True:
                bad[m | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & rest

        # popcount lookup
        pop = [0] * (1 << N)
        for i in range(1, 1 << N):
            pop[i] = pop[i >> 1] + (i & 1)

        # For each k from N down to 1, check if any mask of popcount k is independent
        # Group masks by popcount
        by_size = [[] for _ in range(N + 1)]
        for mask in range(1 << N):
            if not bad[mask]:
                by_size[pop[mask]].append(mask)

        for k in range(N, 0, -1):
            if by_size[k]:
                return k
        return 0

    def is_linear(triples):
        used_pairs = set()
        for t in triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            for p in pairs:
                if p in used_pairs or (p[1], p[0]) in used_pairs:
                    return False
                used_pairs.add(p)
        return True

    def greedy_build(seed_val):
        rng = random.Random(seed_val)
        triples = []
        used_pairs = set()
        all_triples = list(combinations(range(N), 3))
        rng.shuffle(all_triples)
        for t in all_triples:
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            conflict = False
            for p in pairs:
                if p in used_pairs or (p[1], p[0]) in used_pairs:
                    conflict = True
                    break
            if not conflict:
                triples.append(list(t))
                for p in pairs:
                    used_pairs.add(p)
        return triples

    def local_search(triples, seed_val, max_iters=2000):
        """
        Try to reduce alpha by swapping out a triple for a new one that
        creates more "covered" subsets. We use a heuristic: minimize alpha
        directly via swap moves.
        """
        rng = random.Random(seed_val)
        triples = [list(t) for t in triples]
        current_alpha = check_alpha(triples)

        # Precompute pair usage
        used_pairs = set()
        for t in triples:
            for p in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]:
                used_pairs.add(tuple(sorted(p)))

        # Build list of all valid triples that could replace any current one
        # A triple is "available" if none of its 3 pairs are in used_pairs
        # (except the pairs of the triple being removed)

        all_triples_list = list(combinations(range(N), 3))

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            rng.shuffle(triples)
            for idx, t in enumerate(triples):
                # Remove t
                t_pairs = [tuple(sorted(p)) for p in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]]
                for p in t_pairs:
                    used_pairs.discard(p)

                # Find a replacement triple that doesn't conflict with remaining used_pairs
                candidates = []
                for nt in all_triples_list:
                    nt_pairs = [tuple(sorted(p)) for p in [(nt[0], nt[1]), (nt[0], nt[2]), (nt[1], nt[2])]]
                    ok = True
                    for p in nt_pairs:
                        if p in used_pairs:
                            ok = False
                            break
                    if ok:
                        candidates.append(nt)

                if not candidates:
                    # Restore t
                    for p in t_pairs:
                        used_pairs.add(p)
                    continue

                # Try a few random candidates
                rng.shuffle(candidates)
                tried = 0
                found_better = False
                for nt in candidates[:20]:  # limit per removal
                    tried += 1
                    nt_pairs = [tuple(sorted(p)) for p in [(nt[0], nt[1]), (nt[0], nt[2]), (nt[1], nt[2])]]
                    # Tentatively add
                    for p in nt_pairs:
                        used_pairs.add(p)
                    triples[idx] = list(nt)
                    new_alpha = check_alpha(triples)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        found_better = True
                        improved = True
                        break
                    else:
                        # Undo
                        for p in nt_pairs:
                            used_pairs.discard(p)
                        triples[idx] = list(t)

                if not found_better:
                    # Restore t
                    for p in t_pairs:
                        used_pairs.add(p)
                else:
                    # t was replaced, pairs already updated
                    pass

        return triples, current_alpha

    best_triples = None
    best_alpha = N + 1

    num_restarts = 500

    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break

        # Greedy construction
        triples = greedy_build(i * 101 + 7)
        if not triples:
            continue

        # Calculate alpha
        alpha = check_alpha(triples)

        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]

        if best_alpha <= 5:
            break

    # Local search polish on the best solution found
    if best_triples is not None and best_alpha > 5:
        if time.time() - start_time < DEADLINE - 20:
            polished, polished_alpha = local_search(best_triples, 42, max_iters=1000)
            if polished_alpha < best_alpha:
                best_alpha = polished_alpha
                best_triples = polished

        # Try local search on a few other good solutions too
        if time.time() - start_time < DEADLINE - 10:
            for i in range(20):
                if time.time() - start_time > DEADLINE - 5:
                    break
                triples = greedy_build(10000 + i * 137)
                if not triples:
                    continue
                alpha = check_alpha(triples)
                if alpha <= best_alpha:
                    polished, polished_alpha = local_search(triples, 10000 + i * 137, max_iters=500)
                    if polished_alpha < best_alpha:
                        best_alpha = polished_alpha
                        best_triples = polished
                    if best_alpha <= 5:
                        break

    if best_triples is None:
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]

    return best_triples
