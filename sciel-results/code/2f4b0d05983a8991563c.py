"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s to be safe under 90s limit
    N = 512
    n = N

    # Precompute compatibility masks:
    # masks[i] is a bitmask where bit j is 1 if lcm(i+1, j+1) <= N
    masks = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(n):
            b = j + 1
            g = gcd(a, b)
            if a * b // g <= N:
                masks[i] |= (1 << j)

    # Degree of each vertex
    degree = [bin(masks[i]).count('1') for i in range(n)]
    
    # Compute number of divisors for each integer 1..512
    def count_divisors(x):
        cnt = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                cnt += 1
                if d != x // d:
                    cnt += 1
        return cnt
    
    div_count = [count_divisors(i+1) for i in range(n)]
    
    # Weight: prefer numbers with many divisors (more likely to be in large cliques)
    # Also consider prime factor density
    def prime_factor_count(x):
        """Count total prime factors with multiplicity"""
        cnt = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                cnt += 1
                x //= d
            d += 1
        if x > 1:
            cnt += 1
        return cnt
    
    pf_count = [prime_factor_count(i+1) for i in range(n)]
    
    # Combined weight: prioritize numbers that divide many others (high degree) 
    # and have many prime factors
    weight = [degree[i] * 0.5 + pf_count[i] * 10 + div_count[i] * 5 for i in range(n)]

    best_size = 0
    best_set = set()
    best_mask = 0

    def update_best(s, m):
        nonlocal best_size, best_set, best_mask
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            best_mask = m

    def greedy_clique(order):
        """Greedy: process vertices in given order, add if compatible with all already added."""
        clique = set()
        mask = 0
        for v in order:
            if (masks[v] & mask) == mask:
                clique.add(v)
                mask |= (1 << v)
        return clique, mask

    def greedy_by_weight(order):
        """Greedy weighted: at each step, pick highest-weight compatible vertex."""
        clique = set()
        mask = 0
        available = set(order)
        while available and time.time() < deadline:
            best_v = -1
            best_w = -1
            for v in available:
                if (masks[v] & mask) == mask:
                    if weight[v] > best_w:
                        best_w = weight[v]
                        best_v = v
            if best_v < 0:
                break
            clique.add(best_v)
            mask |= (1 << best_v)
            available.remove(best_v)
        return clique, mask

    def local_search(s, m):
        """Local search: greedy add, then single-swap, then double-swap if time permits."""
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Phase 1: greedy add (weighted)
            while True:
                added = False
                best_v = -1
                best_w = -1
                for v in range(n):
                    if v not in s:
                        if (masks[v] & m) == m:
                            if weight[v] > best_w:
                                best_w = weight[v]
                                best_v = v
                if best_v >= 0:
                    s.add(best_v)
                    m |= (1 << best_v)
                    added = True
                    update_best(s, m)
                else:
                    break
            if added:
                improved = True

            # Phase 2: single-swap (remove u, add v) - try multiple u's
            swapped = False
            s_list = list(s)
            random.shuffle(s_list)
            for u in s_list:
                if time.time() > deadline:
                    break
                u_bit = 1 << u
                m_wo_u = m ^ u_bit
                # Find best v to add by weight
                best_v = -1
                best_w = -1
                for v in range(n):
                    if v in s:
                        continue
                    if (masks[v] & m_wo_u) == m_wo_u:
                        if weight[v] > best_w:
                            best_w = weight[v]
                            best_v = v
                if best_v >= 0 and weight[best_v] > weight[u] * 0.8:  # only swap if significantly better
                    s.remove(u)
                    s.add(best_v)
                    m = m_wo_u | (1 << best_v)
                    swapped = True
                    update_best(s, m)
                    break
            if swapped:
                improved = True

            # Phase 3: double-swap (remove u1,u2, add v1,v2) - only if time remains
            if time.time() < deadline - 5:
                double_swapped = False
                s_list2 = list(s)
                random.shuffle(s_list2)
                for i in range(min(len(s_list2), 30)):
                    if time.time() > deadline - 5:
                        break
                    u1 = s_list2[i]
                    if i + 1 < len(s_list2):
                        u2 = s_list2[i + 1]
                    else:
                        continue
                    u1_bit = 1 << u1
                    u2_bit = 1 << u2
                    m_wo = m ^ u1_bit ^ u2_bit
                    # Find two vertices to add
                    candidates = []
                    for v in range(n):
                        if v not in s:
                            if (masks[v] & m_wo) == m_wo:
                                candidates.append(v)
                    if len(candidates) >= 2:
                        # Sort by weight descending
                        candidates.sort(key=lambda x: -weight[x])
                        # Try pairs of top candidates
                        found = False
                        for a_idx in range(min(len(candidates), 20)):
                            if time.time() > deadline - 5:
                                break
                            v1 = candidates[a_idx]
                            v1_bit = 1 << v1
                            m_with_v1 = m_wo | v1_bit
                            for b_idx in range(a_idx + 1, min(len(candidates), 20)):
                                v2 = candidates[b_idx]
                                v2_bit = 1 << v2
                                if (masks[v2] & m_with_v1) == m_with_v1:
                                    # Check if this swap improves weight
                                    if weight[v1] + weight[v2] > weight[u1] + weight[u2]:
                                        s.remove(u1)
                                        s.remove(u2)
                                        s.add(v1)
                                        s.add(v2)
                                        m = m_with_v1 | v2_bit
                                        double_swapped = True
                                        update_best(s, m)
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                if double_swapped:
                    improved = True

        return s, m

    def branch_and_bound(candidates, current_set, current_mask, depth):
        """Branch and bound for maximum clique in compatibility graph."""
        nonlocal best_size, best_set, best_mask
        if time.time() > deadline:
            return
        
        # Upper bound: current size + max possible additions
        remaining = len(candidates)
        if len(current_set) + remaining <= best_size:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                best_mask = current_mask
            return
        
        # Pick vertex with highest degree in candidate subgraph
        best_v = max(candidates, key=lambda v: sum(1 for u in candidates if v != u and (masks[v] >> u) & 1))
        
        # Branch 1: include best_v
        new_candidates = [v for v in candidates if v != best_v and (masks[best_v] >> v) & 1]
        current_set.add(best_v)
        current_mask |= (1 << best_v)
        branch_and_bound(new_candidates, current_set, current_mask, depth + 1)
        current_set.remove(best_v)
        current_mask ^= (1 << best_v)
        
        # Branch 2: exclude best_v
        branch_and_bound([v for v in candidates if v != best_v], current_set, current_mask, depth + 1)

    # Build initial seeds using various strategies
    inits = []

    # 1. Degree-sorted greedy (descending - prefer high degree)
    order_desc = sorted(range(n), key=lambda x: -degree[x])
    s1, m1 = greedy_clique(order_desc)
    inits.append((s1, m1))

    # 2. Weight-sorted greedy (prefer high weight)
    order_weight = sorted(range(n), key=lambda x: -weight[x])
    s2, m2 = greedy_clique(order_weight)
    inits.append((s2, m2))

    # 3. Weighted greedy (dynamic selection)
    s3, m3 = greedy_by_weight(list(range(n)))
    inits.append((s3, m3))

    # 4. Natural order
    s4, m4 = greedy_clique(list(range(n)))
    inits.append((s4, m4))

    # 5. Divisors of 512
    divs512 = [i + 1 for i in range(n) if 512 % (i + 1) == 0]
    s5 = set(d - 1 for d in divs512)
    m5 = 0
    for i in s5:
        m5 |= (1 << i)
    inits.append((s5, m5))

    # 6. Divisors of 360 (highly composite)
    divs360 = [i + 1 for i in range(n) if 360 % (i + 1) == 0]
    s6 = set(d - 1 for d in divs360)
    m6 = 0
    for i in s6:
        m6 |= (1 << i)
    inits.append((s6, m6))

    # 7. Divisors of 240
    divs240 = [i + 1 for i in range(n) if 240 % (i + 1) == 0]
    s7 = set(d - 1 for d in divs240)
    m7 = 0
    for i in s7:
        m7 |= (1 << i)
    inits.append((s7, m7))

    # 8. Powers of 2
    pows = set()
    p = 1
    while p <= 512:
        pows.add(p - 1)
        p *= 2
    m8 = 0
    for i in pows:
        m8 |= (1 << i)
    inits.append((pows, m8))

    # 9. Random restarts with weighted order
    random.seed(42)
    for _ in range(20):
        random_order = list(range(n))
        random.shuffle(random_order)
        s_r, m_r = greedy_clique(random_order)
        inits.append((s_r, m_r))

    # 10. Random restarts with weight-perturbed order
    for _ in range(20):
        perturbed_weight = [w + random.uniform(-5, 5) for w in weight]
        order_p = sorted(range(n), key=lambda x: -perturbed_weight[x])
        s_p, m_p = greedy_clique(order_p)
        inits.append((s_p, m_p))

    # Run local search on each initial seed
    for s, m in inits:
        if time.time() > deadline:
            break
        s, m = local_search(s, m)

    # Try branch-and-bound on a reduced candidate set
    if time.time() < deadline - 20:
        # Use top 50 vertices by weight as candidate pool for B&B
        top_candidates = sorted(range(n), key=lambda x: -weight[x])[:50]
        bb_set = set()
        bb_mask = 0
        branch_and_bound(top_candidates, bb_set, bb_mask, 0)

    # Additional: try to improve best solution further with more random swaps
    if best_set and time.time() < deadline - 10:
        s = set(best_set)
        m = best_mask
        # Run additional local search rounds
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            s, m = local_search(s, m)

    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
