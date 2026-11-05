"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    N = 256
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Determine valid vertices (no self-conflict: 2*a not a square)
    valid = []
    for a in range(1, N + 1):
        if (2 * a) not in squares:
            valid.append(a)
    M = len(valid)

    # Build adjacency in the COMPLEMENT graph:
    # comp_adj[i] is a bitset (Python int) of j (j != i) such that valid[i] + valid[j] is NOT a square
    # (i.e., i and j can coexist). We treat the complement graph for max-clique.
    comp_adj = [0] * M
    for i in range(M):
        a = valid[i]
        for j in range(M):
            if i == j:
                continue
            b = valid[j]
            if (a + b) not in squares:
                comp_adj[i] |= (1 << j)

    # Bit helper: count bits
    def popcount(x):
        return bin(x).count('1')

    # Greedy coloring of the candidate set R (as a bitset) to get upper bound on clique size.
    # Returns list of colors per vertex (in some order) and the max color (upper bound).
    def greedy_color(candidates):
        # candidates is a bitset; we color greedily by repeatedly picking a vertex
        # and assigning the smallest color not used by its already-colored neighbors.
        # We process vertices in a fixed order (by index) for determinism-ish.
        uncolored = candidates
        color_of = {}
        color_count = [0]  # number of vertices assigned each color (size per color)
        max_color = 0
        # We'll do a simple greedy: iterate, pick lowest index, assign smallest feasible color.
        # To speed up, we maintain per-color vertex bitsets.
        color_sets = []  # list of bitsets
        while uncolored:
            # pick lowest set bit
            lsb = uncolored & (-uncolored)
            v = lsb.bit_length() - 1
            uncolored ^= lsb
            # find smallest color c such that v has no neighbor already in color c
            # i.e., (color_sets[c] & comp_adj[v]) == 0
            c = 0
            while c < len(color_sets):
                if color_sets[c] & comp_adj[v]:
                    c += 1
                else:
                    break
            if c == len(color_sets):
                color_sets.append(0)
            color_sets[c] |= lsb
            color_of[v] = c
            if c + 1 > max_color:
                max_color = c + 1
        return color_of, max_color

    # Bron-Kerbosch with pivot and coloring bound (max clique).
    # R: current clique (bitset), P: candidates (bitset), X: excluded (bitset)
    best = {'size': 0, 'clique': 0}

    def bk(R, P, X):
        if time.time() - start_time > DEADLINE:
            return
        # Bound: |R| + |P|
        if popcount(R) + popcount(P) <= best['size']:
            return
        if P == 0:
            if popcount(R) > best['size']:
                best['size'] = popcount(R)
                best['clique'] = R
            return
        # Greedy coloring of P to get an upper bound
        color_of, max_color = greedy_color(P)
        if popcount(R) + max_color <= best['size']:
            return
        # Pivot selection: pick u in P ∪ X with max degree in P
        PX = P | X
        if PX:
            # pick vertex with max popcount(comp_adj[u] & P)
            best_u = -1
            best_deg = -1
            tmp = PX
            while tmp:
                lsb = tmp & (-tmp)
                u = lsb.bit_length() - 1
                tmp ^= lsb
                deg = popcount(comp_adj[u] & P)
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
            candidates = P & ~(comp_adj[best_u])
        else:
            candidates = P

        # Iterate over candidates in order of decreasing color (for pruning)
        # We process in order; for each v, recurse with R∪{v}, P∩N(v), X∩N(v)
        # Then remove v from P and add to X.
        # To prune using colors, we process colors from high to low.
        # Build list of vertices in candidates ordered by color descending.
        verts = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            verts.append((color_of[v], v))
        verts.sort(reverse=True)

        for c, v in verts:
            if time.time() - start_time > DEADLINE:
                return
            # Prune: if |R| + c <= best_size, no need to go deeper for this and lower colors
            if popcount(R) + c <= best['size']:
                break
            new_R = R | (1 << v)
            new_P = P & comp_adj[v]
            new_X = X & comp_adj[v]
            bk(new_R, new_P, new_X)
            P &= ~(1 << v)
            X |= (1 << v)

    # Initial candidates: all valid vertices
    all_mask = (1 << M) - 1
    bk(0, all_mask, 0)

    # Convert best clique (in complement graph) back to values
    if best['clique']:
        result = []
        tmp = best['clique']
        while tmp:
            lsb = tmp & (-tmp)
            i = lsb.bit_length() - 1
            tmp ^= lsb
            result.append(valid[i])
    else:
        # Fallback: shouldn't happen since at least one vertex is valid
        result = [valid[0]]

    # Local improvement via 1-2 exchange (remove one, add up to two) to possibly exceed
    # the clique found (in case BK pruned due to time, this can still help).
    def is_compatible_set(s, v):
        # v is a value; check against all in set s (values)
        for a in s:
            if (a + v) in squares:
                return False
        return True

    def try_add(s_set, v):
        # s_set is a set of values
        for a in s_set:
            if (a + v) in squares:
                return False
        return True

    result_set = set(result)
    improved = True
    while improved and (time.time() - start_time < DEADLINE):
        improved = False
        # Try single add
        for v in valid:
            if v not in result_set and try_add(result_set, v):
                result_set.add(v)
                improved = True
        # Try 1-2 exchange: remove one, add two
        if time.time() - start_time > DEADLINE:
            break
        for rem in list(result_set):
            temp = result_set - {rem}
            # find two to add
            added = []
            for v1 in valid:
                if v1 in temp or v1 in added:
                    continue
                if not try_add(temp, v1):
                    continue
                for v2 in valid:
                    if v2 in temp or v2 in added or v2 == v1:
                        continue
                    if not try_add(temp, v2):
                        continue
                    # check v1 and v2 compatible with each other and temp
                    if (v1 + v2) in squares:
                        continue
                    newset = temp | {v1, v2}
                    if len(newset) > len(result_set):
                        result_set = newset
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return list(result_set)
# EVOLVE-BLOCK-END
