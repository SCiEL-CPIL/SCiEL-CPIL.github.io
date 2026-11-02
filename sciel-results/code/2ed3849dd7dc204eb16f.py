import time
import itertools
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Standard construction
        word = [1]
        for m in range(2, n + 1):
            if time.time() > deadline:
                break
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 5
    # Build W_{n-1}
    word = [1]
    for m in range(2, n):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1, ):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # Extract perms of n-1 from word
    perms, seen = [], set()
    m_prev = n - 1
    for i in range(len(word) - m_prev + 1):
        window = tuple(word[i:i + m_prev])
        if len(set(window)) == m_prev and window not in seen:
            seen.add(window)
            perms.append(window)

    num_perms = len(perms)

    # Compute overlap matrix: overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    max_k = m_prev - 1
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]

    perm_strs = [''.join(map(str, p)) for p in perms]

    for i in range(num_perms):
        pi_str = perm_strs[i]
        pi_suffixes = {k: pi_str[-k:] for k in range(1, max_k + 1)}
        for j in range(num_perms):
            if i == j:
                continue
            pj_str = perm_strs[j]
            for k in range(max_k, 0, -1):
                if pi_suffixes[k] == pj_str[:k]:
                    overlap_matrix[i][j] = k
                    break
        if time.time() > deadline - 15:
            break

    # Initial order: natural order
    order = list(range(num_perms))

    # Calculate total overlap for circular arrangement
    def calc_total_overlap(ord_list):
        total = 0
        for i in range(len(ord_list)):
            u = ord_list[i]
            v = ord_list[(i + 1) % len(ord_list)]
            total += overlap_matrix[u][v]
        return total

    best_order = order[:]
    best_overlap = calc_total_overlap(best_order)

    # 2-opt with incremental cost
    # For circular arrangement, reversing segment [i, j] affects edges:
    # (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
    # After reversal: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
    # So only 2 edges change: (i-1, i)+(j, j+1) -> (i-1, j)+(i, j+1)
    # Wait, that's not right for a full reversal. Let me think again.
    #
    # Original: ... A, B, C, ..., Z, W ...
    # where A=order[i-1], B=order[i], ..., Z=order[j], W=order[j+1]
    # After reversing [i,j]: ... A, Z, ..., C, B, W ...
    # Edges removed: (A,B) and (Z,W)
    # Edges added: (A,Z) and (B,W)
    # Internal edges are reversed but the sum of overlaps is the same since
    # overlap[X][Y] + overlap[Y][X] is not necessarily symmetric, but the
    # internal edges in the reversed segment contribute the same total.
    # Wait, no. The internal edges change from (B,C), (C,D), ... to (Z,Y), (Y,X), ...
    # So we need to account for all internal edges too.
    #
    # Actually for 2-opt on a circular TSP-like problem:
    # Old edges: (A,B), (B,C), ..., (Y,Z), (Z,W)
    # New edges: (A,Z), (Z,Y), ..., (C,B), (B,W)
    # Delta = [overlap[A][Z] + overlap[B][W]] - [overlap[A][B] + overlap[Z][W]]
    # The internal edges: old has overlap[B][C]+overlap[C][D]+...+overlap[Y][Z]
    # New has overlap[Z][Y]+overlap[Y][X]+...+overlap[C][B]
    # These are different! So we need to compute the full delta.
    #
    # For efficiency, let's compute the delta properly.

    improved = True
    max_iterations = 100
    iteration = 0
    while improved and time.time() < deadline - 10 and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(num_perms - 1):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, num_perms):
                # Compute delta for reversing segment [i, j]
                # Old edges affected: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)

                u_idx = (i - 1) % num_perms
                v_idx = i
                w_idx = j
                x_idx = (j + 1) % num_perms

                u = best_order[u_idx]
                v = best_order[v_idx]
                w = best_order[w_idx]
                x = best_order[x_idx]

                # Old cost contribution from boundary edges
                old_boundary = overlap_matrix[u][v] + overlap_matrix[w][x]

                # New cost contribution from boundary edges
                new_boundary = overlap_matrix[u][w] + overlap_matrix[v][x]

                # Old internal cost
                old_internal = 0
                for k in range(i, j):
                    old_internal += overlap_matrix[best_order[k]][best_order[k + 1]]

                # New internal cost (reversed)
                new_internal = 0
                for k in range(i, j):
                    new_internal += overlap_matrix[best_order[k + 1]][best_order[k]]

                delta = (new_boundary + new_internal) - (old_boundary + old_internal)

                if delta > 0:
                    best_order[i:j + 1] = reversed(best_order[i:j + 1])
                    best_overlap += delta
                    improved = True

    # Or-opt: move segments of length 1, 2, or 3
    for seg_len in [1, 2, 3]:
        if time.time() > deadline - 8:
            break
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            for i in range(num_perms):
                if time.time() > deadline - 8:
                    break
                seg = best_order[i:i + seg_len]
                if len(seg) < seg_len:
                    continue

                for j in range(num_perms):
                    if j >= i and j < i + seg_len:
                        continue

                    # Compute delta for moving seg from position i to position j
                    # This is complex; let's compute it directly.

                    # Save current order
                    old_order = best_order[:]

                    # Create new order
                    new_order = best_order[:]
                    # Remove seg from position i
                    for _ in range(seg_len):
                        new_order.pop(i)

                    # Adjust j if necessary
                    if j > i:
                        j_adj = j - seg_len
                    else:
                        j_adj = j

                    # Insert seg at position j_adj
                    for s in seg:
                        new_order.insert(j_adj, s)
                        j_adj += 1

                    # Calculate old total overlap
                    old_total = 0
                    for idx in range(num_perms):
                        u = old_order[idx]
                        v = old_order[(idx + 1) % num_perms]
                        old_total += overlap_matrix[u][v]

                    # Calculate new total overlap
                    new_total = 0
                    for idx in range(num_perms):
                        u = new_order[idx]
                        v = new_order[(idx + 1) % num_perms]
                        new_total += overlap_matrix[u][v]

                    delta = new_total - old_total
                    if delta > 0:
                        best_order = new_order
                        best_overlap += delta
                        improved = True
                        break  # Restart j loop after successful move

    # Construct the final word efficiently
    blocks = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        blocks.append(block)

    new_word = []
    for bi, block in enumerate(blocks):
        if bi == 0:
            new_word.extend(block)
        else:
            max_possible_overlap = min(len(new_word), m_prev, len(block))
            overlap = 0
            for o in range(max_possible_overlap, 0, -1):
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])

        if time.time() > deadline - 2:
            break

    return "".join(map(str, new_word))
