import time
import random
import sys

# ---------- Precomputation at import ----------
_N = 128

def _build_progressions():
    progs = []
    for d in range(1, _N // 3 + 1):
        for a in range(_N - 3 * d):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            if dd < _N:
                progs.append((a, b, c, dd))
    return progs

_PROGS = _build_progressions()

def _distinct4(ta, tb, tc, td):
    """Return number of distinct values among ta,tb,tc,td."""
    nd = 1
    if tb != ta:
        nd += 1
        if tc != ta and tc != tb:
            nd += 1
            if td != ta and td != tb and td != tc:
                nd += 1
        else:
            if td != ta and td != tb:
                nd += 1
    else:
        if tc != ta:
            nd += 1
            if td != ta and td != tc:
                nd += 1
        else:
            if td != ta:
                nd += 1
    return nd

def _is_valid_fast(trial):
    for (a, b, c, dd) in _PROGS:
        if _distinct4(trial[a], trial[b], trial[c], trial[dd]) < 3:
            return False
    return True

def _greedy_valid():
    ending_at = [[] for _ in range(_N)]
    for (a, b, c, dd) in _PROGS:
        ending_at[dd].append((a, b, c))
    
    for max_colors in range(3, _N + 1):
        trial = [0] * _N
        valid = True
        for i in range(_N):
            found = False
            for c in range(max_colors):
                ok = True
                for (a, b, cc) in ending_at[i]:
                    if _distinct4(trial[a], trial[b], trial[cc], c) < 3:
                        ok = False
                        break
                if ok:
                    trial[i] = c
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid:
            return trial
    return list(range(_N))

_INITIAL_VALID = _greedy_valid()

def _count_distinct(trial):
    return len(set(trial))

def _local_search(trial, max_time, rng):
    n = _N
    progs_for_pos = [[] for _ in range(n)]
    for (a, b, c, dd) in _PROGS:
        progs_for_pos[a].append((a, b, c, dd))
        progs_for_pos[b].append((a, b, c, dd))
        progs_for_pos[c].append((a, b, c, dd))
        progs_for_pos[dd].append((a, b, c, dd))
    
    best = list(trial)
    best_distinct = _count_distinct(best)
    start = time.time()
    
    current = list(trial)
    cur_distinct = _count_distinct(current)
    
    while time.time() - start < max_time:
        progress = False
        positions = list(range(n))
        rng.shuffle(positions)
        
        for pos in positions:
            if time.time() - start > max_time:
                break
            old = current[pos]
            in_use = set(current)
            candidates = list(in_use)
            for c in range(0, min(6, best_distinct + 1)):
                if c not in candidates:
                    candidates.append(c)
            rng.shuffle(candidates)
            
            for new_c in candidates:
                if new_c == old:
                    continue
                # Check validity
                valid = True
                for (a, b, c, dd) in progs_for_pos[pos]:
                    ta = current[a] if a != pos else new_c
                    tb = current[b] if b != pos else new_c
                    tc = current[c] if c != pos else new_c
                    td = current[dd] if dd != pos else new_c
                    if _distinct4(ta, tb, tc, td) < 3:
                        valid = False
                        break
                if valid:
                    current[pos] = new_c
                    new_distinct = _count_distinct(current)
                    if new_distinct < cur_distinct:
                        cur_distinct = new_distinct
                        if cur_distinct < best_distinct:
                            best_distinct = cur_distinct
                            best = list(current)
                        progress = True
                        break
                    else:
                        current[pos] = old
        
        if not progress:
            # Perturb
            for _ in range(8):
                if time.time() - start > max_time:
                    break
                pos = rng.randint(0, n - 1)
                old = current[pos]
                for _ in range(15):
                    new_c = rng.randint(0, best_distinct)
                    if new_c == old:
                        continue
                    valid = True
                    for (a, b, c, dd) in progs_for_pos[pos]:
                        ta = current[a] if a != pos else new_c
                        tb = current[b] if b != pos else new_c
                        tc = current[c] if c != pos else new_c
                        td = current[dd] if dd != pos else new_c
                        if _distinct4(ta, tb, tc, td) < 3:
                            valid = False
                            break
                    if valid:
                        current[pos] = new_c
                        new_distinct = _count_distinct(current)
                        if new_distinct < cur_distinct:
                            cur_distinct = new_distinct
                            if cur_distinct < best_distinct:
                                best_distinct = cur_distinct
                                best = list(current)
                        break
            cur_distinct = _count_distinct(current)
    
    return best, best_distinct

def construct():
    start_time = time.time()
    deadline = 85.0
    
    initial = list(_INITIAL_VALID)
    initial_distinct = _count_distinct(initial)
    
    rng = random.Random(12345)
    
    best = list(initial)
    best_distinct = initial_distinct
    
    attempts = 0
    while time.time() - start_time < deadline - 2.0:
        attempts += 1
        trial = list(initial)
        labels = sorted(set(trial))
        rng.shuffle(labels)
        relabel = {old: new for old, new in zip(sorted(labels), labels)}
        trial = [relabel[c] for c in trial]
        
        remaining = deadline - 2.0 - (time.time() - start_time)
        if remaining < 0.5:
            break
        res, res_distinct = _local_search(trial, remaining, rng)
        if res_distinct < best_distinct:
            best = res
            best_distinct = res_distinct
        
        if best_distinct <= 3:
            break
    
    if not _is_valid_fast(best):
        best = list(initial)
        best_distinct = initial_distinct
    
    return best
