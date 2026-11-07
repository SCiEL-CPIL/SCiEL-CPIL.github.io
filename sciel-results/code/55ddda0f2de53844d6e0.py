import time
import random

def _count_comparable(sets):
    n = len(sets)
    count = 0
    for i in range(n):
        si = sets[i]
        li = len(si)
        for j in range(i + 1, n):
            sj = sets[j]
            lj = len(sj)
            if li < lj:
                if all(x in sj for x in si):
                    count += 1
            elif lj < li:
                if all(x in si for x in sj):
                    count += 1
    return count

def _comparability_with_set(s, existing_sets):
    count = 0
    ls = len(s)
    for e in existing_sets:
        le = len(e)
        if ls < le:
            if all(x in e for x in s):
                count += 1
        elif le < ls:
            if all(x in s for x in e):
                count += 1
    return count

def _construct_initial():
    random.seed(42)
    chain = [list(range(i)) for i in range(13)]
    sets = list(chain)
    all_set_tuples = set(tuple(s) for s in chain)
    
    for _ in range(27):
        best_score = -1
        best_set = None
        for size in range(13):
            if size == 0:
                candidates = [()]
            elif size == 12:
                candidates = [tuple(range(12))]
            else:
                candidates = set()
                for _ in range(100):
                    s = tuple(sorted(random.sample(range(12), size)))
                    candidates.add(s)
                for start in range(0, 12 - size + 1):
                    s = tuple(range(start, start + size))
                    candidates.add(s)
            for s in candidates:
                if s in all_set_tuples:
                    continue
                score = _comparability_with_set(list(s), sets)
                if score > best_score:
                    best_score = score
                    best_set = s
        if best_set is not None:
            sets.append(list(best_set))
            all_set_tuples.add(best_set)
    return sets

def _local_search(sets, max_iters=50):
    current_score = _count_comparable(sets)
    best_sets = list(sets)
    best_score = current_score
    
    for it in range(max_iters):
        improved = False
        for i in range(len(sets)):
            old_set = sets[i]
            existing_other = [sets[j] for j in range(len(sets)) if j != i]
            existing_set_of_tuples = set(tuple(s) for s in existing_other)
            
            candidates = set()
            for x in range(12):
                if x not in old_set:
                    candidates.add(tuple(sorted(old_set + [x])))
            for x in old_set:
                candidates.add(tuple(sorted(set(old_set) - {x})))
            for x in old_set:
                for y in range(12):
                    if y not in old_set:
                        candidates.add(tuple(sorted((set(old_set) - {x}) | {y})))
            for size in range(13):
                for _ in range(20):
                    candidates.add(tuple(sorted(random.sample(range(12), size))))
            
            best_candidate = None
            best_candidate_score = -1
            old_comparability = _comparability_with_set(old_set, existing_other)
            
            for cand in candidates:
                if cand in existing_set_of_tuples:
                    continue
                score = _comparability_with_set(list(cand), existing_other)
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = cand
            
            if best_candidate is not None and best_candidate_score > old_comparability:
                sets[i] = list(best_candidate)
                current_score -= old_comparability
                current_score += best_candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_sets = list(sets)
                improved = True
        if not improved:
            break
    return best_sets

def construct():
    start_time = time.time()
    deadline = start_time + 80
    random.seed(42)
    sets = _construct_initial()
    if time.time() < deadline - 10:
        sets = _local_search(sets, max_iters=30)
    if time.time() < deadline - 5:
        sets = _local_search(sets, max_iters=30)
    if time.time() < deadline - 2:
        sets = _local_search(sets, max_iters=20)
    return [[int(x) for x in s] for s in sets]
