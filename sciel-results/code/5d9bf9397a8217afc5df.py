import random
import time

def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin before 150s kill

    d = 7
    n = 1 << d  # 128 vertices

    # Precompute all edges with indices
    all_edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_index[(u, v)] = idx

    E = len(all_edges)  # 448

    # For each edge index, precompute blocking triples
    blocking_triples = [[] for _ in range(E)]
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u_j = u ^ bit_j
            v_j = v ^ bit_j
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            blocking_triples[idx].append((idx1, idx2, idx3))

    # Bitset-based operations for speed
    def is_blocked_bitset(edge_idx, chosen_bs):
        for triple in blocking_triples[edge_idx]:
            if ((chosen_bs >> triple[0]) & 1) and ((chosen_bs >> triple[1]) & 1) and ((chosen_bs >> triple[2]) & 1):
                return True
        return False

    def popcount(bs):
        return bin(bs).count('1')

    def greedy_random(rng):
        chosen_bs = 0
        order = list(range(E))
        rng.shuffle(order)
        for idx in order:
            if not is_blocked_bitset(idx, chosen_bs):
                chosen_bs |= (1 << idx)
        return chosen_bs

    def local_search(chosen_bs, rng, time_limit):
        improved = True
        all_idx = list(range(E))

        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Add any unblocked edges
            for idx in all_idx:
                if not ((chosen_bs >> idx) & 1) and not is_blocked_bitset(idx, chosen_bs):
                    chosen_bs |= (1 << idx)
                    improved = True

            if time.time() >= time_limit:
                break

            # Phase 2: Single-swap moves
            non_chosen = [i for i in range(E) if not ((chosen_bs >> i) & 1)]
            rng.shuffle(non_chosen)

            for e_idx in non_chosen[:150]:
                if time.time() >= time_limit:
                    break

                blocking = []
                for triple in blocking_triples[e_idx]:
                    if ((chosen_bs >> triple[0]) & 1) and ((chosen_bs >> triple[1]) & 1) and ((chosen_bs >> triple[2]) & 1):
                        blocking.append(triple)

                if not blocking:
                    chosen_bs |= (1 << e_idx)
                    improved = True
                    break

                if len(blocking) == 1:
                    t = blocking[0]
                    best_gain = -1
                    best_trial = None
                    for rem in t:
                        trial = (chosen_bs & ~(1 << rem)) | (1 << e_idx)
                        added = 0
                        for other in non_chosen:
                            if other != e_idx and other != rem:
                                if not ((trial >> other) & 1) and not is_blocked_bitset(other, trial):
                                    trial |= (1 << other)
                                    added += 1
                        if added > best_gain:
                            best_gain = added
                            best_trial = trial

                    if best_trial is not None and best_gain > 0:
                        chosen_bs = best_trial
                        improved = True
                        break

                elif len(blocking) == 2:
                    t1, t2 = blocking[0], blocking[1]
                    common = set(t1) & set(t2)
                    if common:
                        rem = next(iter(common))
                        trial = (chosen_bs & ~(1 << rem)) | (1 << e_idx)
                        for other in non_chosen:
                            if other != e_idx and other != rem:
                                if not ((trial >> other) & 1) and not is_blocked_bitset(other, trial):
                                    trial |= (1 << other)
                        chosen_bs = trial
                        improved = True
                        break

            # Phase 3: 2-edge removal swaps
            if not improved and time.time() < time_limit:
                chosen_list = [i for i in range(E) if (chosen_bs >> i) & 1]
                rng.shuffle(chosen_list)
                found = False
                for a in range(min(30, len(chosen_list))):
                    if time.time() >= time_limit:
                        break
                    for b in range(a + 1, min(a + 4, len(chosen_list))):
                        ra, rb = chosen_list[a], chosen_list[b]
                        trial = chosen_bs & ~(1 << ra) & ~(1 << rb)
                        added = 0
                        for other in range(E):
                            if other != ra and other != rb:
                                if not ((trial >> other) & 1) and not is_blocked_bitset(other, trial):
                                    trial |= (1 << other)
                                    added += 1
                        if added >= 3:
                            chosen_bs = trial
                            improved = True
                            found = True
                            break
                    if found:
                        break

        return chosen_bs

    best_bs = 0
    best_size = 0
    rng = random.Random(42)

    num_starts = 8
    for s in range(num_starts):
        if time.time() > deadline - 40:
            break

        chosen_bs = greedy_random(rng)
        size = popcount(chosen_bs)
        if size > best_size:
            best_bs = chosen_bs
            best_size = size

        time_limit = deadline - 30
        chosen_bs2 = local_search(chosen_bs, rng, time_limit)
        size2 = popcount(chosen_bs2)
        if size2 > best_size:
            best_bs = chosen_bs2
            best_size = size2

        if time.time() > deadline - 20:
            break

    # Final greedy fill on best
    if time.time() < deadline - 5:
        for idx in range(E):
            if not ((best_bs >> idx) & 1) and not is_blocked_bitset(idx, best_bs):
                best_bs |= (1 << idx)

    # Convert to list of [u, v]
    out = []
    for idx in range(E):
        if (best_bs >> idx) & 1:
            u, v = all_edges[idx]
            out.append([u, v])
    return out
