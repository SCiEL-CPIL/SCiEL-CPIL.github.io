import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    result = _greedy_build(20, deadline)
    if result:
        return result

    for n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 5:
            break
        result = _greedy_build(n, deadline)
        if result:
            return result

    return {'n': 5, 'edges': []}

def _greedy_build(n, deadline):
    """Build graph by adding vertices one at a time, choosing connections to minimize violations."""
    rng = random.Random(42)
    adj = []

    for nv in range(n):
        if time.time() > deadline:
            break

        if nv == 0:
            adj.append(0)
            continue

        existing = list(range(nv))

        # Precompute for each subset S of size 5 or 6:
        #   For each r in 0..|S|, if S is "r-compatible", store required pattern bits
        subsets_required = []

        for size in [5, 6]:
            if nv >= size:
                for S in itertools.combinations(existing, size):
                    mask_S = 0
                    for v in S:
                        mask_S |= 1 << v

                    deg_in_S = {}
                    for v in S:
                        d = 0
                        for w in S:
                            if v != w and (adj[v] & (1 << w)):
                                d += 1
                        deg_in_S[v] = d

                    required = {}
                    for r in range(size + 1):
                        compatible = True
                        count_r_minus_1 = 0
                        for v in S:
                            d = deg_in_S[v]
                            if r == 0:
                                if d != 0:
                                    compatible = False
                                    break
                            elif r == size:
                                if d != size - 1:
                                    compatible = False
                                    break
                            else:
                                if d == r - 1:
                                    count_r_minus_1 += 1
                                elif d == r:
                                    pass
                                else:
                                    compatible = False
                                    break
                        if not compatible:
                            continue
                        if r == 0:
                            if count_r_minus_1 != 0:
                                continue
                        elif r == size:
                            pass
                        else:
                            if count_r_minus_1 != r:
                                continue

                        req_bits = 0
                        if r == 0:
                            pass
                        elif r == size:
                            req_bits = mask_S
                        else:
                            for v in S:
                                if deg_in_S[v] == r - 1:
                                    req_bits |= 1 << v
                        required[r] = req_bits

                    if required:
                        subsets_required.append((mask_S, required))

        # Local search over P
        best_pattern = None
        best_viol = float('inf')

        num_starts = min(20, max(3, int((deadline - time.time()) / 0.1)))

        for si in range(num_starts):
            if time.time() > deadline:
                break

            P = 0
            for v in existing:
                if rng.random() < 0.5:
                    P |= 1 << v

            improved = True
            while improved and time.time() < deadline:
                improved = False
                cur_viol = _count_violations_v2(P, subsets_required)

                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best_pattern = P
                    if cur_viol == 0:
                        break

                for v in existing:
                    new_P = P ^ (1 << v)
                    new_viol = _count_violations_v2(new_P, subsets_required)
                    if new_viol < cur_viol:
                        P = new_P
                        cur_viol = new_viol
                        improved = True
                        if cur_viol < best_viol:
                            best_viol = cur_viol
                            best_pattern = P
                            if cur_viol == 0:
                                break
                if cur_viol == 0:
                    break

            if best_viol == 0:
                break

        if best_pattern is None:
            best_pattern = 0
            for v in existing:
                if rng.random() < 0.5:
                    best_pattern |= 1 << v

        adj.append(best_pattern)
        for v in existing:
            if best_pattern & (1 << v):
                adj[v] |= 1 << nv

    if _is_valid_full(adj, n):
        return _to_result(adj, n)

    if time.time() < deadline:
        repaired = _repair(adj, n, deadline)
        if repaired:
            return repaired

    return None

def _count_violations_v2(P, subsets_required):
    count = 0
    for mask_S, required in subsets_required:
        P_restricted = P & mask_S
        for r, req_bits in required.items():
            if P_restricted == req_bits:
                count += 1
                break
    return count

def _is_valid_full(adj, n):
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return False
    return True

def _is_regular(adj, subset):
    s = len(subset)
    if s < 2:
        return True
    mask = 0
    for v in subset:
        mask |= 1 << v
    degs = set()
    for v in subset:
        degs.add((adj[v] & mask).bit_count())
    return len(degs) == 1

def _repair(adj, n, deadline):
    rng = random.Random(123)
    while time.time() < deadline:
        viol = _find_violation(adj, n)
        if viol is None:
            return _to_result(adj, n)
        u, v = rng.sample(viol, 2)
        adj[u] ^= 1 << v
        adj[v] ^= 1 << u
    return None

def _find_violation(adj, n):
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return subset
    return None

def _to_result(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
