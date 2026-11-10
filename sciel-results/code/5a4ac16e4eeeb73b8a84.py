import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    candidates = list(range(20, 201))
    n = len(candidates)

    # Precompute gcd matrix
    gcd_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = gcd(candidates[i], candidates[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    def is_valid(selected, m_idx, r):
        for (m2_idx, r2) in selected:
            g = gcd_matrix[m_idx][m2_idx]
            if r % g == r2 % g:
                return False
        return True

    def compute_score(selected):
        return sum(1.0 / candidates[idx] for idx, _ in selected)

    def greedy_construct(order, deadline):
        selected = []
        for m_idx in order:
            if time.time() > deadline:
                break
            m = candidates[m_idx]
            for r in range(m):
                if is_valid(selected, m_idx, r):
                    selected.append((m_idx, r))
                    break
        return selected

    def local_search(selected, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_sel = len(selected)

            # Residue reassignment
            for i in range(n_sel):
                if time.time() > deadline:
                    break
                m_idx, r = selected[i]
                m = candidates[m_idx]
                for new_r in range(m):
                    if new_r == r:
                        continue
                    valid = True
                    for j in range(n_sel):
                        if j == i:
                            continue
                        g = gcd_matrix[m_idx][selected[j][0]]
                        if new_r % g == selected[j][1] % g:
                            valid = False
                            break
                    if valid:
                        selected[i] = (m_idx, new_r)
                        improved = True
                        break

            # Insertion
            sel_set = set(idx for idx, _ in selected)
            for m_idx in range(n):
                if time.time() > deadline:
                    break
                if m_idx in sel_set:
                    continue
                m = candidates[m_idx]
                for r in range(m):
                    if is_valid(selected, m_idx, r):
                        selected.append((m_idx, r))
                        improved = True
                        break

            # 2-opt: remove one, try to add two
            if len(selected) >= 2:
                for i in range(len(selected) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    m_idx, r = selected[i]
                    val = 1.0 / candidates[m_idx]
                    temp = selected[:i] + selected[i+1:]
                    temp_set = set(idx for idx, _ in temp)
                    best_add = None
                    best_gain = 0
                    for m2_idx in range(n):
                        if m2_idx in temp_set or m2_idx == m_idx:
                            continue
                        m2 = candidates[m2_idx]
                        for r2 in range(m2):
                            if is_valid(temp, m2_idx, r2):
                                gain = 1.0 / m2 - val
                                if gain > best_gain:
                                    best_gain = gain
                                    best_add = (m2_idx, r2)
                                break
                    if best_add and best_gain > 0:
                        selected = temp + [best_add]
                        improved = True
                        break

        return selected

    best_selected = None
    best_score = 0.0

    # Ascending greedy
    order = list(range(n))
    sel = greedy_construct(order, deadline - 40)
    sel = local_search(sel, deadline - 15)
    s = compute_score(sel)
    if s > best_score:
        best_score = s
        best_selected = sel[:]

    # Randomized restarts with bias toward small moduli
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        # Weighted random order
        weights = [1.0 / candidates[i] for i in range(n)]
        order = []
        avail = list(range(n))
        while avail:
            tw = sum(weights[i] for i in avail)
            r = random.random() * tw
            cum = 0
            for idx in avail:
                cum += weights[idx]
                if r <= cum:
                    order.append(idx)
                    avail.remove(idx)
                    break
        sel = greedy_construct(order, deadline - 15)
        sel = local_search(sel, deadline - 5)
        s = compute_score(sel)
        if s > best_score:
            best_score = s
            best_selected = sel[:]

    if not best_selected:
        return [[20, 0]]
    return [[candidates[idx], r] for idx, r in best_selected]
