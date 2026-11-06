import time
import sys
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before the 60s limit

    n = 256
    all_mask = (1 << n) - 1

    # Precompute conflict masks
    # conflict[i] is a bitmask of indices j such that (i+1) and (j+1) conflict
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask_i |= (1 << j)
        conflict[i] = mask_i

    # Complementary adjacency (H): edge (u,v) in H iff NOT conflict
    adj = [0] * n
    for i in range(n):
        adj[i] = (~conflict[i]) & all_mask
        adj[i] &= ~(1 << i)

    # Degree in H
    degrees = [adj[i].bit_count() for i in range(n)]

    # Order vertices by degree descending
    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    rank = [0] * n
    for idx, v in enumerate(order):
        rank[v] = idx

    # Greedy coloring bound for MIS on candidate set P
    def color_bound(P):
        if P == 0:
            return 0
        verts = []
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            verts.append(i)
            p ^= lsb

        # Sort by rank (degree-based)
        verts.sort(key=lambda x: rank[x])

        colors = [-1] * n
        max_color = 0

        for v in verts:
            used = set()
            neigh = conflict[v] & P
            np_ = neigh
            while np_:
                lsb = np_ & -np_
                u = lsb.bit_length() - 1
                if colors[u] >= 0:
                    used.add(colors[u])
                np_ ^= lsb

            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c

        return max_color

    best_size = 0
    best_mask = 0

    def bbmc(P, current):
        nonlocal best_size, best_mask

        if time.time() - start_time > timeout:
            return

        if P == 0:
            if len(current) > best_size:
                best_size = len(current)
                mask = 0
                for v in current:
                    mask |= (1 << v)
                best_mask = mask
            return

        bound = color_bound(P)
        if len(current) + bound <= best_size:
            return

        # Pick vertex with lowest rank (highest degree)
        v = -1
        min_rank = n + 1
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            if rank[i] < min_rank:
                min_rank = rank[i]
                v = i
            p ^= lsb

        if v == -1:
            return

        v_bit = 1 << v

        # Include v
        new_P = P & adj[v]
        current.append(v)
        bbmc(new_P, current)
        current.pop()

        # Exclude v
        new_P_exc = P & ~v_bit
        bbmc(new_P_exc, current)

    # Initial greedy solution
    greedy_mask = 0
    P = all_mask
    while P:
        # Pick vertex with highest degree in remaining P
        v = -1
        max_deg = -1
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            deg = (adj[i] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = i
            p ^= lsb
        if v == -1:
            break
        greedy_mask |= (1 << v)
        P &= adj[v]  # Keep only neighbors of v in H

    if greedy_mask:
        best_size = greedy_mask.bit_count()
        best_mask = greedy_mask

    # Run BBMC
    bbmc(all_mask, [])

    # Local search improvement on best found
    if best_mask:
        # Try to add vertices
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if not ((best_mask >> i) & 1):
                    # Check if i can be added
                    if (conflict[i] & best_mask) == 0:
                        best_mask |= (1 << i)
                        best_size += 1
                        improved = True
                        break

    # Simulated annealing refinement
    def mask_to_set(mask):
        s = set()
        p = mask
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            s.add(i)
            p ^= lsb
        return s

    def set_to_mask(s):
        m = 0
        for i in s:
            m |= (1 << i)
        return m

    def anneal(S_set, time_budget):
        # Simulated annealing over independent sets.
        # Uses set intersection for O(1)-amortised can_add checks, a high swap
        # probability to explore same-size configurations, and slow cooling so
        # the chain stays mobile. math.exp is used for the Boltzmann factor.
        S = set(S_set)
        T0 = 1.2
        T_end = 0.015
        cooling = 0.9995   # slow cooling -> more iterations at high T
        T = T0

        # Precompute conflict sets for fast lookup
        conflict_sets = [set() for _ in range(n)]
        for i in range(n):
            c = conflict[i]
            p = c
            while p:
                lsb = p & -p
                j = lsb.bit_length() - 1
                conflict_sets[i].add(j)
                p ^= lsb

        while time.time() - start_time < time_budget:
            if T < T_end:
                break

            if not S:
                break

            # choose a random vertex in S to try removing
            v = random.choice(tuple(S))
            S_new = S - {v}

            # neighbours of v not in S_new (in conflict graph)
            cands = [u for u in conflict_sets[v] if u not in S_new]

            # valid if no neighbour of u is in S_new (O(1) via set intersection)
            valid = [u for u in cands if not (conflict_sets[u] & S_new)]

            # greedy independent subset of valid
            rem = set(valid)
            add = set()
            while rem:
                vc = random.choice(tuple(rem))
                add.add(vc)
                rem -= conflict_sets[vc]
                rem.discard(vc)

            S_cand = S_new | add
            d = len(S_cand) - len(S)

            if d >= 0 or random.random() < math.exp(d / T):
                S = S_cand

            T *= cooling

        return S

    # Run annealing on the best solution
    if best_mask and time.time() - start_time < timeout - 5:
        S_set = mask_to_set(best_mask)
        anneal_budget = min(20.0, max(2.0, timeout - (time.time() - start_time) - 2))
        if time.time() - start_time + anneal_budget <= timeout - 1:
            S_annealed = anneal(S_set, timeout - (time.time() - start_time) - 2)
            if len(S_annealed) > best_size:
                best_size = len(S_annealed)
                best_mask = set_to_mask(S_annealed)

                # Try to add more vertices after annealing
                improved = True
                while improved:
                    improved = False
                    for i in range(n):
                        if not ((best_mask >> i) & 1):
                            if (conflict[i] & best_mask) == 0:
                                best_mask |= (1 << i)
                                best_size += 1
                                improved = True
                                break

    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)

    if not result:
        return [1]

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a = res[i]
            b = res[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
