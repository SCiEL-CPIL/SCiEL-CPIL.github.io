"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _build_triples(n_max=180):
    """Build all triples (a, b, c) with a < b < c, a in [1,n_max], and 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            diff = b - a
            if (a * b) % diff != 0:
                continue
            c = (a * b) // diff
            if c > n_max or c <= b:
                continue
            triples.append((a, b, c))
    return triples

def _greedy_independent_set(triples, elem_to_triples, n_max, deadline):
    """Greedy: start with all, remove highest-conflict element iteratively."""
    S = set(range(1, n_max + 1))
    triple_incount = [3] * len(triples)
    elem_conflict = {i: 0 for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_conflict[a] += 1
        elem_conflict[b] += 1
        elem_conflict[c] += 1
    total_conflicts = len(triples)
    
    while total_conflicts > 0:
        if time.time() > deadline:
            break
        best_elem = max(S, key=lambda x: elem_conflict[x])
        if elem_conflict[best_elem] == 0:
            break
        S.remove(best_elem)
        for tidx in elem_to_triples[best_elem]:
            if triple_incount[tidx] == 3:
                total_conflicts -= 1
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x in S:
                        elem_conflict[x] -= 1
            triple_incount[tidx] -= 1
    return sorted(S)

def _local_search(triples, elem_to_triples, n_max, start_set, deadline, max_iter=5000):
    """Local search with 1-opt and 2-opt moves: try adding removed elements, or remove one to add two."""
    S = set(start_set)
    removed = set(range(1, n_max + 1)) - S
    
    def count_conflicts(s):
        cnt = 0
        for a, b, c in triples:
            if a in s and b in s and c in s:
                cnt += 1
        return cnt
    
    if count_conflicts(S) > 0:
        # First fix by removing
        S = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
        removed = set(range(1, n_max + 1)) - S
    
    best_size = len(S)
    no_improve = 0
    max_no_improve = 200
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve > max_no_improve:
            break
        
        # Try 1-opt: add a single removed element
        candidates = list(removed)
        if not candidates:
            break
        random.shuffle(candidates)
        improved = False
        for elem in candidates[:min(80, len(candidates))]:
            new_conflicts = 0
            for tidx in elem_to_triples[elem]:
                a, b, c = triples[tidx]
                others = [x for x in (a, b, c) if x != elem]
                if all(x in S for x in others):
                    new_conflicts += 1
            if new_conflicts == 0:
                S.add(elem)
                removed.discard(elem)
                best_size = max(best_size, len(S))
                improved = True
                no_improve = 0
                break
        
        if improved:
            continue
        
        # Try 2-opt: remove one element from S, then try to add two from removed
        s_candidates = list(S)
        if not s_candidates:
            break
        random.shuffle(s_candidates)
        two_opt_improved = False
        
        for rem_elem in s_candidates[:min(30, len(s_candidates))]:
            # Temporarily remove rem_elem
            S.discard(rem_elem)
            removed.add(rem_elem)
            
            # Try to add two elements from removed
            add_candidates = list(removed)
            random.shuffle(add_candidates)
            best_pair = None
            best_net_gain = 0
            
            for i, elem1 in enumerate(add_candidates[:min(60, len(add_candidates))]):
                # Check if adding elem1 alone is conflict-free
                conflicts1 = 0
                for tidx in elem_to_triples[elem1]:
                    a, b, c = triples[tidx]
                    others = [x for x in (a, b, c) if x != elem1]
                    if all(x in S for x in others):
                        conflicts1 += 1
                if conflicts1 > 0:
                    continue
                
                # Try adding elem1, then check elem2
                for elem2 in add_candidates[i+1:i+1+min(60, len(add_candidates)-i-1)]:
                    if elem2 == elem1:
                        continue
                    # Check if adding both elem1 and elem2 is conflict-free
                    conflicts2 = 0
                    for tidx in elem_to_triples[elem2]:
                        a, b, c = triples[tidx]
                        others = [x for x in (a, b, c) if x != elem2]
                        if all(x in S for x in others) or (elem1 in S and elem1 in others):
                            conflicts2 += 1
                    if conflicts2 == 0:
                        # Check for conflict between elem1 and elem2 themselves
                        triple_conflict = False
                        for tidx in elem_to_triples[elem1]:
                            a, b, c = triples[tidx]
                            if elem2 in (a, b, c) and elem1 in (a, b, c):
                                triple_conflict = True
                                break
                        if not triple_conflict:
                            best_pair = (elem1, elem2)
                            best_net_gain = 1
                            break
                if best_pair:
                    break
            
            if best_pair:
                elem1, elem2 = best_pair
                S.add(elem1)
                S.add(elem2)
                removed.discard(elem1)
                removed.discard(elem2)
                best_size = max(best_size, len(S))
                two_opt_improved = True
                no_improve = 0
                break
            else:
                # Restore rem_elem
                S.add(rem_elem)
                removed.discard(rem_elem)
        
        if not two_opt_improved:
            no_improve += 1
    
    return sorted(S)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    n_max = 180
    triples = _build_triples(n_max)
    
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
    
    # Strategy 1: Greedy removal (highest conflict first)
    best_set = _greedy_independent_set(triples, elem_to_triples, n_max, deadline)
    
    # Strategy 2: Local search with 2-opt from greedy result
    if time.time() < deadline - 10.0:
        ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline - 5.0, max_iter=3000)
        if len(ls_set) > len(best_set):
            best_set = ls_set
    
    # Strategy 3: Greedy with different tie-breaking (lowest element first)
    if time.time() < deadline - 10.0:
        S = set(range(1, n_max + 1))
        triple_incount = [3] * len(triples)
        elem_conflict = {i: 0 for i in range(1, n_max + 1)}
        for idx, (a, b, c) in enumerate(triples):
            elem_conflict[a] += 1
            elem_conflict[b] += 1
            elem_conflict[c] += 1
        total_conflicts = len(triples)
        while total_conflicts > 0:
            if time.time() > deadline - 8.0:
                break
            max_conf = max((elem_conflict[x] for x in S), default=0)
            if max_conf == 0:
                break
            best_elem = min(x for x in S if elem_conflict[x] == max_conf)
            S.remove(best_elem)
            for tidx in elem_to_triples[best_elem]:
                if triple_incount[tidx] == 3:
                    total_conflicts -= 1
                    a, b, c = triples[tidx]
                    for x in (a, b, c):
                        if x in S:
                            elem_conflict[x] -= 1
                triple_incount[tidx] -= 1
        alt_set = sorted(S)
        if len(alt_set) > len(best_set):
            best_set = alt_set
    
    # Strategy 4: Greedy with tie-breaking (highest element first)
    if time.time() < deadline - 10.0:
        S = set(range(1, n_max + 1))
        triple_incount = [3] * len(triples)
        elem_conflict = {i: 0 for i in range(1, n_max + 1)}
        for idx, (a, b, c) in enumerate(triples):
            elem_conflict[a] += 1
            elem_conflict[b] += 1
            elem_conflict[c] += 1
        total_conflicts = len(triples)
        while total_conflicts > 0:
            if time.time() > deadline - 8.0:
                break
            max_conf = max((elem_conflict[x] for x in S), default=0)
            if max_conf == 0:
                break
            best_elem = max(x for x in S if elem_conflict[x] == max_conf)
            S.remove(best_elem)
            for tidx in elem_to_triples[best_elem]:
                if triple_incount[tidx] == 3:
                    total_conflicts -= 1
                    a, b, c = triples[tidx]
                    for x in (a, b, c):
                        if x in S:
                            elem_conflict[x] -= 1
                triple_incount[tidx] -= 1
        alt_set = sorted(S)
        if len(alt_set) > len(best_set):
            best_set = alt_set
    
    # Strategy 5: Randomized greedy with multiple restarts
    if time.time() < deadline - 10.0:
        for restart in range(3):
            if time.time() > deadline - 8.0:
                break
            S = set(range(1, n_max + 1))
            triple_incount = [3] * len(triples)
            elem_conflict = {i: 0 for i in range(1, n_max + 1)}
            for idx, (a, b, c) in enumerate(triples):
                elem_conflict[a] += 1
                elem_conflict[b] += 1
                elem_conflict[c] += 1
            total_conflicts = len(triples)
            while total_conflicts > 0:
                if time.time() > deadline - 8.0:
                    break
                max_conf = max((elem_conflict[x] for x in S), default=0)
                if max_conf == 0:
                    break
                # Random tie-breaking
                candidates = [x for x in S if elem_conflict[x] == max_conf]
                random.shuffle(candidates)
                best_elem = candidates[0]
                S.remove(best_elem)
                for tidx in elem_to_triples[best_elem]:
                    if triple_incount[tidx] == 3:
                        total_conflicts -= 1
                        a, b, c = triples[tidx]
                        for x in (a, b, c):
                            if x in S:
                                elem_conflict[x] -= 1
                    triple_incount[tidx] -= 1
            alt_set = sorted(S)
            if len(alt_set) > len(best_set):
                best_set = alt_set
    
    # Final local search on best_set with more iterations
    if time.time() < deadline - 3.0:
        ls_set = _local_search(triples, elem_to_triples, n_max, best_set, deadline, max_iter=5000)
        if len(ls_set) > len(best_set):
            best_set = ls_set
    
    if not best_set:
        best_set = list(range(91, 181))
    
    return best_set

# EVOLVE-BLOCK-END
