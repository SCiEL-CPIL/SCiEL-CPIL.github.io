import time
import math
import sys

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s wall-clock; leave room for return/serialization
    DEADLINE = start_time + 84

    LIMIT = 512
    N = LIMIT

    # 1. Build Compatibility Graph as bitmasks
    # compat[i] is a bitmask of nodes j (1..N) such that lcm(i, j) <= LIMIT
    compat = [0] * (N + 1)

    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                compat[i] |= (1 << (j - 1))
                compat[j] |= (1 << (i - 1))

    # Static degree for each node
    deg = [0] * (N + 1)
    for i in range(1, N + 1):
        deg[i] = compat[i].bit_count()

    # 2. Multiple greedy lower-bound strategies
    def greedy_sequential():
        """Sequential greedy: process nodes in descending degree, add if compatible."""
        nodes = list(range(1, N + 1))
        nodes.sort(key=lambda x: -deg[x])
        clique = []
        for node in nodes:
            if time.time() > DEADLINE:
                break
            ok = True
            for c in clique:
                if not (compat[node] & (1 << (c - 1))):
                    ok = False
                    break
            if ok:
                clique.append(node)
        return clique

    def greedy_max_degree():
        """Max-degree greedy: always pick node with max degree within remaining candidates."""
        clique = []
        # remaining as a bitmask for speed
        remaining = (1 << N) - 1
        while remaining:
            if time.time() > DEADLINE:
                break
            # Find node in remaining with max number of compatible neighbors in remaining
            best = 0
            best_count = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                v = idx + 1
                # Count neighbors of v within remaining
                cnt = (compat[v] & remaining).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best = v
                temp ^= lsb
            if best == 0:
                break
            clique.append(best)
            remaining &= ~(1 << (best - 1))
            # Remove nodes not compatible with best
            remaining &= compat[best]
        return clique

    def greedy_multi_seed():
        """Try greedy starting from different single seeds (highest degree nodes)."""
        best = []
        seeds = sorted(range(1, N + 1), key=lambda x: -deg[x])[:20]
        for seed in seeds:
            if time.time() > DEADLINE:
                break
            clique = [seed]
            # Candidates: neighbors of seed, sorted by degree
            cands = []
            temp = compat[seed] & ~(1 << (seed - 1))
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                cands.append(idx + 1)
                temp ^= lsb
            cands.sort(key=lambda x: -deg[x])
            for node in cands:
                if time.time() > DEADLINE:
                    break
                ok = True
                for c in clique:
                    if not (compat[node] & (1 << (c - 1))):
                        ok = False
                        break
                if ok:
                    clique.append(node)
            if len(clique) > len(best):
                best = clique[:]
        return best

    # Get initial lower bound from all greedy strategies
    best_clique = []
    for fn in [greedy_sequential, greedy_max_degree, greedy_multi_seed]:
        c = fn()
        if len(c) > len(best_clique):
            best_clique = c

    if not best_clique:
        return [1]

    # 3. Tomita's Algorithm with Bitmasks
    sys.setrecursionlimit(10000)

    def expand(P_mask, C, depth):
        nonlocal best_clique

        if time.time() > DEADLINE:
            return

        if P_mask == 0:
            if depth > len(best_clique):
                best_clique = C[:]
            return

        if depth + P_mask.bit_count() <= len(best_clique):
            return

        # Extract vertices from P_mask
        P = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            P.append(idx + 1)
            temp ^= lsb

        # Greedy Coloring - sort P by static degree descending
        P.sort(key=lambda v: -deg[v])

        color_list = []
        vertex_color = {}

        for v in P:
            assigned = False
            for k in range(len(color_list)):
                conflict = False
                for u in color_list[k]:
                    if compat[v] & (1 << (u - 1)):
                        conflict = True
                        break
                if not conflict:
                    color_list[k].append(v)
                    vertex_color[v] = k
                    assigned = True
                    break

            if not assigned:
                color_list.append([v])
                vertex_color[v] = len(color_list) - 1

        max_color = len(color_list)

        if depth + max_color <= len(best_clique):
            return

        # Branching: iterate colors in reverse (highest color first)
        for k in range(max_color - 1, -1, -1):
            for v in color_list[k]:
                if not (P_mask & (1 << (v - 1))):
                    continue

                # Include v
                C.append(v)
                new_P_mask = P_mask & compat[v] & ~(1 << (v - 1))
                expand(new_P_mask, C, depth + 1)
                C.pop()

                # Exclude v
                P_mask &= ~(1 << (v - 1))

                if time.time() > DEADLINE:
                    return

    initial_P_mask = (1 << N) - 1
    expand(initial_P_mask, [], 0)

    # 4. Local search polish: try to add or swap nodes to improve the clique
    def polish(clique):
        """Try to improve clique via single-node additions and swaps."""
        if time.time() > DEADLINE:
            return clique
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            clique_set = set(clique)
            # Try adding each non-member
            for v in range(1, N + 1):
                if v in clique_set:
                    continue
                if all(compat[v] & (1 << (c - 1)) for c in clique):
                    clique.append(v)
                    clique_set.add(v)
                    improved = True
            # Try swaps: remove one, add one
            if not improved:
                for i in range(len(clique)):
                    if time.time() > DEADLINE:
                        return clique
                    removed = clique[i]
                    clique.pop(i)
                    clique_set.discard(removed)
                    best_add = None
                    for v in range(1, N + 1):
                        if v in clique_set or v == removed:
                            continue
                        if all(compat[v] & (1 << (c - 1)) for c in clique):
                            best_add = v
                            break
                    if best_add is not None:
                        clique.append(best_add)
                        clique_set.add(best_add)
                        improved = True
                    else:
                        clique.insert(i, removed)
                        clique_set.add(removed)
        return clique

    best_clique = polish(best_clique)

    if not best_clique:
        return [1]

    return best_clique

if __name__ == "__main__":
    result = construct()
    print(f"Found clique of size {len(result)}")
    print(result)
