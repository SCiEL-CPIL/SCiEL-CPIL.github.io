import time
import random
import math
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n != 7:
        return _standard_construct(n)
        
    # 1. Generate permutations of 1..6
    perms = list(itertools.permutations(range(1, 7)))
    num_perms = len(perms)  # 720
    
    # 2. Compute Overlap Matrix M
    # M[i][j] = maximum overlap (1 to 5) such that suffix of perms[i] of length k
    # equals prefix of perms[j] of length k.
    # We maximize total overlap to minimize total length.
    M = [[0] * num_perms for _ in range(num_perms)]
    
    # Precompute suffixes and prefixes as tuples for fast lookup?
    # Or just direct comparison.
    # Let's try to be smart:
    # For a given permutation p, its suffix of length k is p[6-k:]
    # Its prefix of length k is p[:k]
    
    # To speed up M calculation, we can iterate by overlap length k (5 down to 1)
    # and fill in M[i][j] if not already set.
    # Initialize M with 0.
    
    # Map from (suffix_tuple) to list of indices?
    # Or map from (prefix_tuple) to list of indices?
    
    # Since N=720 is small, let's just do the O(N^2 * 5) loop but optimized.
    for i in range(num_perms):
        p1 = perms[i]
        for j in range(num_perms):
            p2 = perms[j]
            ov = 0
            # Check k from 5 down to 1
            if p1[1] == p2[0] and p1[2] == p2[1] and p1[3] == p2[2] and p1[4] == p2[3] and p1[5] == p2[4]:
                ov = 5
            elif p1[2] == p2[0] and p1[3] == p2[1] and p1[4] == p2[2] and p1[5] == p2[3]:
                ov = 4
            elif p1[3] == p2[0] and p1[4] == p2[1] and p1[5] == p2[2]:
                ov = 3
            elif p1[4] == p2[0] and p1[5] == p2[1]:
                ov = 2
            elif p1[5] == p2[0]:
                ov = 1
            M[i][j] = ov

    # 3. Greedy Initialization (Nearest Neighbor)
    order = [0]
    used = [False] * num_perms
    used[0] = True
    for _ in range(num_perms - 1):
        last = order[-1]
        best_next = -1
        best_ov = -1
        for j in range(num_perms):
            if not used[j]:
                ov = M[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        order.append(best_next)
        used[best_next] = True

    def calc_total_overlap(ord_list):
        s = 0
        for i in range(len(ord_list) - 1):
            s += M[ord_list[i]][ord_list[i + 1]]
        return s

    current_overlap = calc_total_overlap(order)
    best_order = order[:]
    best_overlap = current_overlap

    # 4. 2-Opt Local Search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(num_perms - 1):
            if time.time() > deadline - 5.0:
                break
            u = order[i]
            v = order[i + 1]
            for j in range(i + 2, num_perms):
                if time.time() > deadline - 5.0:
                    break
                x = order[j]
                if j < num_perms - 1:
                    y = order[j + 1]
                    old_ov = M[u][v] + M[x][y]
                    new_ov = M[u][x] + M[v][y]
                    if new_ov > old_ov:
                        order[i + 1:j + 1] = reversed(order[i + 1:j + 1])
                        current_overlap += (new_ov - old_ov)
                        improved = True
                        if current_overlap > best_overlap:
                            best_overlap = current_overlap
                            best_order = order[:]
                else:
                    # Reversing suffix
                    # Old: M[u][v]
                    # New: M[u][x]
                    # Internal edges are reversed, so their contribution changes from
                    # sum(M[order[k]][order[k+1]] for k in i+1..j-1) to
                    # sum(M[order[k+1]][order[k]] for k in i+1..j-1)
                    # This is hard to compute in O(1).
                    # Let's just skip these moves for now to keep it fast.
                    pass

    # 5. Or-Opt Local Search
    # Move a block of length 1, 2, or 3.
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for L in [1, 2, 3]:
                if time.time() > deadline - 2.0:
                    break
                for i in range(num_perms - L):
                    if time.time() > deadline - 2.0:
                        break
                    # Segment is order[i ... i+L-1]
                    # Remove it and insert at j.
                    # To avoid duplicates, we can just iterate j.
                    # Delta calculation:
                    # Old edges: (prev, start), (end, next)
                    # New edges: (prev, next), (insert_prev, start), (end, insert_next)
                    # This is getting complex.
                    # Let's stick to 2-opt and maybe a simple "swap" or "insert" if time permits.
                    # Given the constraints, 2-opt is the most robust.
                    pass

    # 6. Construct Word
    # Use best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            ov = M[best_order[idx - 1]][best_order[idx]]
            word += block[ov:]

    return "".join(map(str, word))

def _standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
