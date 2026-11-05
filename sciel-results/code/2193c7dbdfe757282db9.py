import time
from math import gcd, lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 128
    numbers = list(range(1, N + 1))

    # Build divisors_by_lcm
    maxL = N * N
    divisors_by_lcm = [[] for _ in range(maxL + 1)]
    for x in range(1, N + 1):
        for L in range(x, maxL + 1, x):
            divisors_by_lcm[L].append(x)

    # Precompute lcm for pairs
    lcm_pair = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lcm_pair[(a, b)] = lcm(a, b)

    # Enumerate bad triples
    bad_triples_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_pair[(a, b)]
            for c in divisors_by_lcm[lab]:
                if c <= b or c > N:
                    continue
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples_set.add((a, b, c))

    bad_triples = list(bad_triples_set)
    T = len(bad_triples)

    if time.time() > deadline:
        return numbers

    # Incidence
    vertex_to_triples = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c) in enumerate(bad_triples):
        vertex_to_triples[a].append(i)
        vertex_to_triples[b].append(i)
        vertex_to_triples[c].append(i)

    # Shadow adjacency as bitmasks
    shadow_adj_mask = [0] * N
    for (a, b, c) in bad_triples:
        ia, ib, ic = a - 1, b - 1, c - 1
        shadow_adj_mask[ia] |= (1 << ib) | (1 << ic)
        shadow_adj_mask[ib] |= (1 << ia) | (1 << ic)
        shadow_adj_mask[ic] |= (1 << ia) | (1 << ib)

    all_bits = (1 << N) - 1

    # Greedy coloring bound on shadow graph (for independent set upper bound)
    # We color the shadow graph itself; the independence number is at most the number of colors
    # in any proper coloring? No, that's for clique number.
    #
    # Correct approach: independence number of G = clique number of complement(G).
    # Greedy coloring of complement(G) gives upper bound on clique number of complement(G).
    #
    # So we need to color the complement of the shadow graph.
    # Comp edge: u~v if NOT in shadow (and u != v).
    
    def greedy_color_bound(candidate_mask):
        """Upper bound on max independent set in shadow graph on candidate_mask."""
        # Extract candidates
        candidates = []
        m = candidate_mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            candidates.append(idx)
            m ^= lb
        
        if not candidates:
            return 0
        
        # Greedy coloring of complement graph
        color_of = {}
        next_color = 0
        for v in candidates:
            used = set()
            for w in candidates:
                if w == v:
                    continue
                # w is comp-neighbor of v if NOT in shadow
                if not (shadow_adj_mask[v] & (1 << w)):
                    if w in color_of:
                        used.add(color_of[w])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            if c + 1 > next_color:
                next_color = c + 1
        return next_color

    # B&B
    best_set = set()
    best_size = 0
    degree = [len(vertex_to_triples[x]) for x in range(1, N + 1)]
    order = sorted(range(N), key=lambda i: degree[i + 1], reverse=True)

    def can_add(current_mask, v):
        for ti in vertex_to_triples[v + 1]:
            a, b, c = bad_triples[ti]
            others = [x for x in (a, b, c) if x != v + 1]
            if (current_mask & (1 << (others[0] - 1))) and (current_mask & (1 << (others[1] - 1))):
                return False
        return True

    def bbb(pos, current_mask, candidate_mask):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        ub = current_mask.bit_count() + greedy_color_bound(candidate_mask)
        if ub <= best_size:
            return
        
        if pos >= N:
            sz = current_mask.bit_count()
            if sz > best_size:
                best_size = sz
                best_set = set(i + 1 for i in range(N) if current_mask & (1 << i))
            return
        
        v = order[pos]
        v_bit = 1 << v
        
        # Exclude
        bbb(pos + 1, current_mask, candidate_mask & ~v_bit)
        
        # Include
        if candidate_mask & v_bit and can_add(current_mask, v):
            bbb(pos + 1, current_mask | v_bit, candidate_mask & ~v_bit)

    bbb(0, 0, all_bits)

    if time.time() > deadline:
        if best_set:
            return list(best_set)
        return numbers

    # Polish
    result_set = set(best_set)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x in result_set:
                continue
            ok = True
            for ti in vertex_to_triples[x]:
                a, b, c = bad_triples[ti]
                others = [y for y in (a, b, c) if y != x]
                if others[0] in result_set and others[1] in result_set:
                    ok = False
                    break
            if ok:
                result_set.add(x)
                improved = True

    if not result_set:
        return [1]
    return list(result_set)
