"""Strong edge colouring: multi-strategy DSATUR + improved induced matching peeling."""
import time
from random import Random

_T0 = time.time()
_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _build_instance(n, d, seed):
    rng = Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if e in edges:
            continue
        if degree[u] < d and degree[v] < d:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _build_conflict(edges, n):
    m = len(edges)
    edge_set = set(edges)
    conflict = [set() for _ in range(m)]
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            if a == c or a == d or b == c or b == d:
                conflict[i].add(j)
                conflict[j].add(i)
            else:
                if (min(a, c), max(a, c)) in edge_set or \
                   (min(a, d), max(a, d)) in edge_set or \
                   (min(b, c), max(b, c)) in edge_set or \
                   (min(b, d), max(b, d)) in edge_set:
                    conflict[i].add(j)
                    conflict[j].add(i)
    return conflict


def _build_bitset_conflict(edges, n):
    m = len(edges)
    edge_set = set(edges)
    conflict = [0] * m
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            if a == c or a == d or b == c or b == d:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
            else:
                if (min(a, c), max(a, c)) in edge_set or \
                   (min(a, d), max(a, d)) in edge_set or \
                   (min(b, c), max(b, c)) in edge_set or \
                   (min(b, d), max(b, d)) in edge_set:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
    return conflict


def _dsatur_color(conflict):
    m = len(conflict)
    if m == 0:
        return []
    colors = [-1] * m
    sat = [set() for _ in range(m)]
    uncolored = list(range(m))
    color_count = 0

    while uncolored:
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in uncolored:
            s = len(sat[v])
            dg = len(conflict[v])
            if s > best_sat or (s == best_sat and dg > best_deg):
                best_v = v
                best_sat = s
                best_deg = dg

        c = 0
        while c in sat[best_v]:
            c += 1
        colors[best_v] = c
        if c == color_count:
            color_count += 1

        for nb in conflict[best_v]:
            if colors[nb] == -1:
                sat[nb].add(c)

        uncolored.remove(best_v)

    return colors


def _dsatur_randomized(conflict, rng):
    m = len(conflict)
    if m == 0:
        return []
    colors = [-1] * m
    sat = [set() for _ in range(m)]
    uncolored = list(range(m))
    color_count = 0

    while uncolored:
        max_sat = max(len(sat[v]) for v in uncolored)
        candidates = [v for v in uncolored if len(sat[v]) == max_sat]
        if len(candidates) > 1:
            best_deg = max(len(conflict[v]) for v in candidates)
            top = [v for v in candidates if len(conflict[v]) == best_deg]
            best_v = rng.choice(top)
        else:
            best_v = candidates[0]

        c = 0
        while c in sat[best_v]:
            c += 1
        colors[best_v] = c
        if c == color_count:
            color_count += 1

        for nb in conflict[best_v]:
            if colors[nb] == -1:
                sat[nb].add(c)

        uncolored.remove(best_v)

    return colors


def _mis_bnb(conflict, candidates, current_mask, current_size, best_mask, best_size, time_limit):
    if time.time() > time_limit:
        return
    
    if candidates == 0:
        if current_size > best_size[0]:
            best_size[0] = current_size
            best_mask[0] = current_mask
        return
        
    if current_size + candidates.bit_count() <= best_size[0]:
        return
        
    max_deg = -1
    best_v_bit = 0
    temp_cand = candidates
    while temp_cand:
        v_bit = temp_cand & -temp_cand
        v = v_bit.bit_length() - 1
        deg = (conflict[v] & candidates).bit_count()
        if deg > max_deg:
            max_deg = deg
            best_v_bit = v_bit
        temp_cand &= temp_cand - 1
        
    v = best_v_bit.bit_length() - 1
    v_bit = best_v_bit
    
    new_cand_include = candidates & ~(1 << v) & ~conflict[v]
    _mis_bnb(conflict, new_cand_include, current_mask | v_bit, current_size + 1, best_mask, best_size, time_limit)
    
    new_cand_exclude = candidates & ~(1 << v)
    _mis_bnb(conflict, new_cand_exclude, current_mask, current_size, best_mask, best_size, time_limit)


def _peeling_color_bnb(edges, n, conflict_bitset, rng, time_limit):
    m = len(edges)
    colors = [0] * m
    remaining_mask = (1 << m) - 1
    color_count = 0
    
    while remaining_mask and time.time() < time_limit:
        time_left_total = time_limit - time.time()
        if time_left_total < 0.5:
            break
            
        mis_time_limit = min(5.0, time_left_total * 0.8)
        
        best_mask = [0]
        best_size = [0]
        
        _mis_bnb(conflict_bitset, remaining_mask, 0, 0, best_mask, best_size, time.time() + mis_time_limit)
        
        if best_size[0] == 0:
            v_bit = remaining_mask & -remaining_mask
            best_mask[0] = v_bit
            best_size[0] = 1
            
        mask = best_mask[0]
        temp_mask = mask
        while temp_mask:
            v_bit = temp_mask & -temp_mask
            v = v_bit.bit_length() - 1
            colors[v] = color_count
            temp_mask &= temp_mask - 1
            
        remaining_mask &= ~mask
        color_count += 1
    
    if remaining_mask:
        temp_mask = remaining_mask
        while temp_mask:
            v_bit = temp_mask & -temp_mask
            v = v_bit.bit_length() - 1
            colors[v] = color_count
            color_count += 1
            temp_mask &= temp_mask - 1

    return colors, color_count


def _try_reduce(conflict, colors, k, time_limit):
    m = len(colors)
    last_color = k - 1
    vertices_with_last = [v for v in range(m) if colors[v] == last_color]

    if not vertices_with_last:
        return None

    vertices_with_last.sort(key=lambda v: sum(1 for c in range(k - 1) 
                                              if all(new_colors[nb] != c for nb in conflict[v])))
    new_colors = colors[:]

    def backtrack(idx):
        if time.time() > time_limit:
            return False
        if idx == len(vertices_with_last):
            return True
        v = vertices_with_last[idx]
        for c in range(k - 1):
            ok = True
            for nb in conflict[v]:
                if new_colors[nb] == c:
                    ok = False
                    break
            if ok:
                new_colors[v] = c
                if backtrack(idx + 1):
                    return True
                new_colors[v] = last_color
        return False

    if backtrack(0):
        return new_colors
    return None


def _local_search_reduce(conflict, colors, k, time_limit):
    m = len(colors)
    best = list(colors)
    best_k = k

    while best_k > 1 and time.time() < time_limit and _time_left() > 10.0:
        new_colors = _try_reduce(conflict, best, best_k, time_limit)
        if new_colors is not None:
            best = new_colors
            best_k -= 1
            continue

        last_color = best_k - 1
        last_class = [v for v in range(m) if best[v] == last_color]
        improved = False
        for v in last_class:
            for c in range(last_color):
                ok = True
                for nb in conflict[v]:
                    if best[nb] == c:
                        ok = False
                        break
                if ok:
                    best[v] = c
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best, best_k


def _solve_instance(n, d, seed):
    edges = _build_instance(n, d, seed)
    m = len(edges)
    if m == 0:
        return []

    conflict_set = _build_conflict(edges, n)
    conflict_bitset = _build_bitset_conflict(edges, n)

    best_colors = None
    best_k = m + 1
    rng = Random(seed + 7777)

    colors1 = _dsatur_color(conflict_set)
    k1 = max(colors1) + 1 if colors1 else 0
    if k1 < best_k:
        best_colors = colors1
        best_k = k1

    if _time_left() > 15.0:
        t_lim = _T0 + _DEADLINE - 10.0
        colors_p, k_p = _peeling_color_bnb(edges, n, conflict_bitset, rng, t_lim)
        if k_p < best_k:
            best_colors = colors_p
            best_k = k_p

    if _time_left() > 10.0 and best_colors is not None:
        t_lim = min(time.time() + 10.0, _T0 + _DEADLINE - 5.0)
        best_colors, best_k = _local_search_reduce(conflict_set, best_colors, best_k, t_lim)

    return best_colors


def construct():
    global _T0
    _T0 = time.time()
    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if _time_left() < 15.0:
            edges = _build_instance(*spec)
            n = spec[0]
            m = len(edges)
            if m == 0:
                results.append([])
                continue
            edge_set = set(edges)
            colors = [0] * m
            for i in range(m):
                a, b = edges[i]
                used = set()
                for j in range(i):
                    c, d = edges[j]
                    if a == c or a == d or b == c or b == d:
                        used.add(colors[j])
                    elif (min(a, c), max(a, c)) in edge_set or \
                         (min(a, d), max(a, d)) in edge_set or \
                         (min(b, c), max(b, c)) in edge_set or \
                         (min(b, d), max(b, d)) in edge_set:
                        used.add(colors[j])
                colors[i] = 0
                while colors[i] in used:
                    colors[i] += 1
            results.append(colors)
            continue

        colors = _solve_instance(*spec)
        results.append(colors)

    return results
