import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    N = 511
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512.0 - a)

    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        m = 0
        for b in range(1, N + 1):
            if a != b and math.gcd(a, b) > 1:
                m |= (1 << b)
        conflict[a] = m

    def can_add(a, chosen_mask):
        return (conflict[a] & chosen_mask) == 0

    def fill_greedy(chosen_list, chosen_mask, score):
        improved = True
        while improved and (time.time() < deadline - 20):
            improved = False
            for a in range(N, 0, -1):
                if (chosen_mask >> a) & 1:
                    continue
                if can_add(a, chosen_mask):
                    chosen_list.append(a)
                    chosen_mask |= (1 << a)
                    score += w[a]
                    improved = True
        return chosen_list, chosen_mask, score

    # Greedy init
    chosen = []
    chosen_mask = 0
    for a in range(N, 0, -1):
        if can_add(a, chosen_mask):
            chosen.append(a)
            chosen_mask |= (1 << a)
    chosen, chosen_mask, score = fill_greedy(chosen, chosen_mask, 0.0)
    best_score = score
    best_chosen = list(chosen)
    best_mask = chosen_mask

    random.seed(42)

    def local_search(chosen_list, chosen_mask, score):
        improved = True
        while improved and (time.time() < deadline - 15):
            improved = False
            for a in range(N, 0, -1):
                if (chosen_mask >> a) & 1:
                    continue
                if can_add(a, chosen_mask):
                    new_score = score + w[a]
                    if new_score > score + 1e-12:
                        chosen_list.append(a)
                        chosen_mask |= (1 << a)
                        score = new_score
                        improved = True
                        break
            if time.time() > deadline - 15:
                break
            for s in list(chosen_list):
                if time.time() > deadline - 15:
                    break
                for u in range(N, 0, -1):
                    if u == s or (chosen_mask >> u) & 1:
                        continue
                    temp_mask = chosen_mask & ~(1 << s)
                    if can_add(u, temp_mask):
                        new_score = score - w[s] + w[u]
                        if new_score > score + 1e-12:
                            chosen_list.remove(s)
                            chosen_list.append(u)
                            chosen_mask = (chosen_mask & ~(1 << s)) | (1 << u)
                            score = new_score
                            improved = True
                            break
            if time.time() > deadline - 15:
                break
            for s in list(chosen_list):
                if time.time() > deadline - 15:
                    break
                temp_mask = chosen_mask & ~(1 << s)
                candidates = [u for u in range(N, 0, -1)
                              if not ((chosen_mask >> u) & 1) and can_add(u, temp_mask)]
                for i in range(len(candidates)):
                    if time.time() > deadline - 15:
                        break
                    u1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        u2 = candidates[j]
                        if can_add(u2, temp_mask | (1 << u1)):
                            new_score = score - w[s] + w[u1] + w[u2]
                            if new_score > score + 1e-12:
                                chosen_list.remove(s)
                                chosen_list.append(u1)
                                chosen_list.append(u2)
                                chosen_mask = temp_mask | (1 << u1) | (1 << u2)
                                score = new_score
                                improved = True
                                break
                    if improved:
                        break
            if time.time() > deadline - 15:
                break
        chosen_list, chosen_mask, score = fill_greedy(chosen_list, chosen_mask, score)
        return chosen_list, chosen_mask, score

    chosen, chosen_mask, score = local_search(list(best_chosen), best_mask, best_score)
    if score > best_score:
        best_score = score
        best_chosen = list(chosen)
        best_mask = chosen_mask

    # B&B
    order = list(range(N, 0, -1))
    bb_deadline = deadline - 25
    best_bb_score = [best_score]
    best_bb_chosen = [best_chosen]
    best_bb_mask = [best_mask]
    nodes = [0]

    def bb(current_sum, chosen_mask, avail_mask, idx):
        if time.time() > bb_deadline:
            return
        nodes[0] += 1
        if idx >= len(order):
            if current_sum > best_bb_score[0] + 1e-12:
                best_bb_score[0] = current_sum
                best_bb_chosen[0] = list(chosen_list_ref)
                best_bb_mask[0] = chosen_mask
            return
        a = order[idx]
        # Exclude a
        new_avail = avail_mask & ~(1 << a)
        # Upper bound
        ub = current_sum
        for v in order:
            if v == a:
                continue
            if (new_avail >> v) & 1:
                ub += w[v]
        if ub > best_bb_score[0] + 1e-12:
            bb(current_sum, chosen_mask, new_avail, idx + 1)
        # Include a
        if can_add(a, chosen_mask):
            new_avail2 = new_avail & ~conflict[a]
            ub2 = current_sum + w[a]
            for v in order:
                if (new_avail2 >> v) & 1:
                    ub2 += w[v]
            if ub2 > best_bb_score[0] + 1e-12:
                chosen_list_ref.append(a)
                bb(current_sum + w[a], chosen_mask | (1 << a), new_avail2, idx + 1)
                chosen_list_ref.pop()

    chosen_list_ref = []
    initial_avail = 0
    for a in range(1, N + 1):
        initial_avail |= (1 << a)
    bb(0.0, 0, initial_avail, 0)

    if best_bb_score[0] > best_score:
        best_score = best_bb_score[0]
        best_chosen = best_bb_chosen[0]
        best_mask = best_bb_mask[0]

    # Final polish
    if time.time() < deadline - 5:
        chosen, chosen_mask, score = local_search(list(best_chosen), best_mask, best_score)
        if score > best_score:
            best_score = score
            best_chosen = list(chosen)
            best_mask = chosen_mask

    if time.time() < deadline - 3:
        for _ in range(3):
            improved = False
            for s in list(best_chosen):
                if time.time() > deadline - 3:
                    break
                for u in range(N, 0, -1):
                    if u == s or (best_mask >> u) & 1:
                        continue
                    temp_mask = best_mask & ~(1 << s)
                    if can_add(u, temp_mask):
                        new_score = best_score - w[s] + w[u]
                        if new_score > best_score + 1e-15:
                            best_chosen.remove(s)
                            best_chosen.append(u)
                            best_mask = temp_mask | (1 << u)
                            best_score = new_score
                            improved = True
                            break
                if time.time() > deadline - 3:
                    break
            if not improved:
                break
        best_chosen, best_mask, best_score = fill_greedy(best_chosen, best_mask, best_score)

    return best_chosen
