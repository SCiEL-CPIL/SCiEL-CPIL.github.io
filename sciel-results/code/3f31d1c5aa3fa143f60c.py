import numpy as np
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    supports4 = list(combinations(range(D), 4))
    N4 = len(supports4)
    sup4_sets = [set(s) for s in supports4]

    # Build conflict graph for 4-subsets: conflict if |S∩T| >= 3
    adj4 = [set() for _ in range(N4)]
    for i in range(N4):
        for j in range(i + 1, N4):
            if len(sup4_sets[i] & sup4_sets[j]) >= 3:
                adj4[i].add(j)
                adj4[j].add(i)

    def greedy_mis(adj, n, rng, deadline):
        """Greedy minimum-degree MIS."""
        selected = set()
        avail = set(range(n))
        while avail:
            if time.time() > deadline:
                break
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(adj[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            u = rng.choice(candidates)
            selected.add(u)
            avail -= {u} | (adj[u] & avail)
        return selected

    def local_search_mis(adj, n, iset, rng, deadline):
        """1-2 swap local search for MIS."""
        iset = set(iset)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in list(iset):
                if time.time() > deadline:
                    break
                sel_minus_v = iset - {v}
                potential = []
                for x in range(n):
                    if x in iset or x == v:
                        continue
                    if not (adj[x] & sel_minus_v):
                        potential.append(x)
                L = len(potential)
                found = False
                for i in range(L):
                    u1 = potential[i]
                    for j in range(i + 1, L):
                        u2 = potential[j]
                        if u2 not in adj[u1]:
                            iset.remove(v)
                            iset.add(u1)
                            iset.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break
        return iset

    # Find good MIS on 4-subset conflict graph
    best_iset4 = set()
    best_size = 0
    rng = np.random.default_rng(42)

    # Use more restarts and simulated annealing for better MIS
    for r in range(50):
        if time.time() > deadline - 25.0:
            break
        iset = greedy_mis(adj4, N4, rng, deadline - 25.0)
        iset = local_search_mis(adj4, N4, iset, rng, deadline - 25.0)
        if len(iset) > best_size:
            best_size = len(iset)
            best_iset4 = iset.copy()

    # Generate all weight-4 vectors from the selected supports
    vectors = []
    for i in best_iset4:
        sup = supports4[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    # Step 2: Try to add weight-3 vectors that are compatible with all existing vectors
    supports3 = list(combinations(range(D), 3))
    N3 = len(supports3)
    sup3_sets = [set(s) for s in supports3]

    # For each weight-3 candidate, check compatibility with all existing vectors
    # To speed up, group existing vectors by support
    sup_to_vecs = {}
    for idx, v in enumerate(vectors):
        sup = tuple(sorted(np.nonzero(v)[0].tolist()))
        if sup not in sup_to_vecs:
            sup_to_vecs[sup] = []
        sup_to_vecs[sup].append((idx, v))

    added_w3 = 0
    for t_idx, T in enumerate(supports3):
        if time.time() > deadline - 15.0:
            break
        T_set = sup3_sets[t_idx]
        for signs in product([1, -1], repeat=3):
            v3 = np.zeros(D, dtype=np.int8)
            for idx, s in zip(T, signs):
                v3[idx] = s

            # Check compatibility with all existing vectors
            compatible = True
            for S_tuple, vecs in sup_to_vecs.items():
                S_set = set(S_tuple)
                inter = S_set & T_set
                m = len(inter)
                if m == 0:
                    continue
                # Check each vector in this support
                for _, v4 in vecs:
                    uv = 0
                    for i in inter:
                        uv += int(v4[i]) * int(v3[i])
                    if uv > 1:
                        compatible = False
                        break
                if not compatible:
                    break

            if compatible:
                vectors.append(v3)
                added_w3 += 1

    # Step 3: Try to add weight-2 vectors
    supports2 = list(combinations(range(D), 2))
    N2 = len(supports2)

    # Rebuild sup_to_vecs with all vectors now
    sup_to_vecs = {}
    for idx, v in enumerate(vectors):
        sup = tuple(sorted(np.nonzero(v)[0].tolist()))
        if sup not in sup_to_vecs:
            sup_to_vecs[sup] = []
        sup_to_vecs[sup].append((idx, v))

    added_w2 = 0
    for S2 in supports2:
        if time.time() > deadline - 8.0:
            break
        S2_set = set(S2)
        for signs in product([1, -1], repeat=2):
            v2 = np.zeros(D, dtype=np.int8)
            for idx, s in zip(S2, signs):
                v2[idx] = s

            compatible = True
            for S_tuple, vecs in sup_to_vecs.items():
                S_set = set(S_tuple)
                inter = S_set & S2_set
                m = len(inter)
                if m == 0:
                    continue
                for _, v in vecs:
                    uv = 0
                    for i in inter:
                        uv += int(v[i]) * int(v2[i])
                    # Determine norm of v
                    vn2 = int(np.sum(v.astype(np.int64) ** 2))
                    if vn2 == 4:
                        # weight-4: need uv <= 1
                        if uv > 1:
                            compatible = False
                            break
                    elif vn2 == 3:
                        # weight-3: need uv <= 0
                        if uv > 0:
                            compatible = False
                            break
                    elif vn2 == 2:
                        # weight-2: need uv <= 0
                        if uv > 0:
                            compatible = False
                            break
                    elif vn2 == 1:
                        # weight-1: need uv <= -1
                        if uv > -1:
                            compatible = False
                            break
                if not compatible:
                    break

            if compatible:
                vectors.append(v2)
                added_w2 += 1

    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
