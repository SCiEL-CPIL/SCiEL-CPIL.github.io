"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _count_comparable(family):
    fs = [frozenset(s) for s in family]
    n = len(fs)
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if fs[i] < fs[j] or fs[j] < fs[i]:
                cnt += 1
    return cnt

def _greedy_construct(n_verts, m, deadline):
    all_sets = []
    for r in range(n_verts + 1):
        for combo in itertools.combinations(range(n_verts), r):
            all_sets.append(list(combo))
    
    empty = []
    candidates = [s for s in all_sets if s != empty]
    family = [empty]
    fs = [frozenset(empty)]
    
    while len(family) < m and time.time() < deadline:
        best_score = -1
        best_idx = -1
        for i, s in enumerate(candidates):
            fs_s = frozenset(s)
            if fs_s in fs:
                continue
            score = 0
            for existing in fs:
                if fs_s < existing or existing < fs_s:
                    score += 1
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        family.append(candidates[best_idx])
        fs.append(frozenset(candidates[best_idx]))
        candidates.pop(best_idx)
    
    return family

def _local_improve(family, all_sets_pool, deadline):
    """Try swapping out a set for a better one."""
    fs = [frozenset(s) for s in family]
    family_set = set(fs)
    candidates = [s for s in all_sets_pool if frozenset(s) not in family_set]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_score = _count_comparable(family)
        for i in range(len(family)):
            if time.time() > deadline:
                return family
            for c in candidates:
                fs_c = frozenset(c)
                # Remove family[i], add c
                new_fs = fs[:i] + fs[i+1:] + [fs_c]
                new_score = _count_comparable([list(x) for x in new_fs])
                if new_score > current_score:
                    family[i] = list(c)
                    fs[i] = fs_c
                    current_score = new_score
                    improved = True
                    # Update candidates
                    candidates = [s for s in candidates if frozenset(s) != fs_c]
                    candidates.append(list(fs[i-1]) if i > 0 else list(fs[i+1]) if i+1 < len(fs) else [])
                    # This is getting messy, let's just break and restart
                    break
            if improved:
                break
    return family

def _small_vertex_family(n_verts, m):
    all_sets = []
    for r in range(n_verts + 1):
        for combo in itertools.combinations(range(n_verts), r):
            all_sets.append(list(combo))
    all_sets.sort(key=len)
    return all_sets[:m]

def construct():
    deadline = time.time() + 80
    
    best_family = None
    best_score = -1
    
    # Strategy 1: Small vertex sets, take smallest by size
    for nv in [6, 7]:
        if time.time() > deadline:
            break
        fam = _small_vertex_family(nv, 40)
        if len(fam) == 40:
            sc = _count_comparable(fam)
            if sc > best_score:
                best_score = sc
                best_family = fam
    
    # Strategy 2: Greedy construction
    for nv in [6, 7]:
        if time.time() > deadline:
            break
        fam = _greedy_construct(nv, 40, deadline)
        if len(fam) == 40:
            sc = _count_comparable(fam)
            if sc > best_score:
                best_score = sc
                best_family = fam
    
    # Fallback
    if best_family is None or len(best_family) != 40:
        best_family = [[b for b in range(12) if (mask >> b) & 1] for mask in range(40)]
    
    result = [[int(x) for x in s] for s in best_family]
    return result
# EVOLVE-BLOCK-END
