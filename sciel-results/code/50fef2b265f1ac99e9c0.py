import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n_max = 180
    all_elements = list(range(1, n_max + 1))

    # Precompute all forbidden triples (a, b, c) with a < b < c, 1/a = 1/b + 1/c
    def _precompute_triples(n_max=180):
        triples = []
        elem_to_triples = {i: [] for i in range(1, n_max + 1)}
        seen = set()
        for a in range(1, n_max + 1):
            a_sq = a * a
            for x in range(1, a):
                if a_sq % x == 0:
                    y = a_sq // x
                    if x >= y:
                        continue
                    b = a + x
                    c = a + y
                    if b <= n_max and c <= n_max:
                        t = (a, b, c)
                        if t not in seen:
                            seen.add(t)
                            idx = len(triples)
                            triples.append(t)
                            elem_to_triples[a].append(idx)
                            elem_to_triples[b].append(idx)
                            elem_to_triples[c].append(idx)
        return triples, elem_to_triples

    triples, elem_to_triples = _precompute_triples(n_max)
    m = len(triples)

    def _check_conflict_fast(x, S, triples, elem_to_triples):
        """Check if adding x to set S creates any forbidden triple."""
        for t_idx in elem_to_triples[x]:
            a, b, c = triples[t_idx]
            others = [v for v in (a, b, c) if v != x]
            if others[0] in S and others[1] in S:
                return True
        return False

    def _greedy_removal(S, triples, elem_to_triples, n_max, deadline):
        """Remove elements to eliminate all conflicts, prioritizing high-conflict elements."""
        triple_state = np.zeros(m, dtype=np.int32)
        for idx, (a, b, c) in enumerate(triples):
            cnt = (1 if a in S else 0) + (1 if b in S else 0) + (1 if c in S else 0)
            triple_state[idx] = cnt

        conflict_count = np.zeros(n_max + 1, dtype=np.int32)
        total_conflicts = 0
        for idx in range(m):
            if triple_state[idx] == 3:
                total_conflicts += 1
                a, b, c = triples[idx]
                conflict_count[a] += 1
                conflict_count[b] += 1
                conflict_count[c] += 1

        while total_conflicts > 0:
            if time.time() > deadline:
                break
            # Find element with max conflict count in S
            max_conflict = 0
            best_elem = None
            for x in S:
                if conflict_count[x] > max_conflict:
                    max_conflict = conflict_count[x]
                    best_elem = x
            if best_elem is None or max_conflict == 0:
                break
            S.remove(best_elem)
            for t_idx in elem_to_triples[best_elem]:
                old_cnt = triple_state[t_idx]
                if old_cnt == 3:
                    total_conflicts -= 1
                    a, b, c = triples[t_idx]
                    for x in (a, b, c):
                        if x != best_elem:
                            conflict_count[x] -= 1
                triple_state[t_idx] = old_cnt - 1
        return S

    def _try_add_back(S, triples, elem_to_triples, n_max, deadline):
        """Try to add back elements that were removed."""
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for elem in range(1, n_max + 1):
                if elem in S:
                    continue
                if not _check_conflict_fast(elem, S, triples, elem_to_triples):
                    S.add(elem)
                    improved = True
                    break
        return S

    def _local_search(S, triples, elem_to_triples, n_max, deadline, max_iters=200):
        """Local search with add-back and swap moves."""
        best_S = set(S)
        best_size = len(S)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try adding back any missing element
            added = False
            for elem in range(1, n_max + 1):
                if elem in S:
                    continue
                if not _check_conflict_fast(elem, S, triples, elem_to_triples):
                    S.add(elem)
                    added = True
                    break
            
            if added:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                continue
            
            # Try swap: remove one element, try to add two others
            S_list = list(S)
            random.shuffle(S_list)
            swap_found = False
            for v_remove in S_list:
                if time.time() > deadline:
                    break
                S.remove(v_remove)
                added_count = 0
                added_elems = []
                for elem in range(1, n_max + 1):
                    if elem in S:
                        continue
                    if not _check_conflict_fast(elem, S, triples, elem_to_triples):
                        S.add(elem)
                        added_elems.append(elem)
                        added_count += 1
                    if added_count >= 2:
                        break
                
                if added_count > 1:
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    swap_found = True
                    break
                else:
                    # Revert
                    for e in added_elems:
                        S.discard(e)
                    S.add(v_remove)
            
            if not swap_found and not added:
                break
        
        return best_S

    # ILP solver using scipy
    def _solve_ilp(triples, n_max, deadline):
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            return None
        
        num_vars = n_max
        num_constraints = len(triples)
        
        if num_constraints == 0:
            return set(range(1, n_max + 1))
        
        c = -np.ones(num_vars)
        A = np.zeros((num_constraints, num_vars))
        for idx, (a, b, c_val) in enumerate(triples):
            A[idx, a - 1] = 1
            A[idx, b - 1] = 1
            A[idx, c_val - 1] = 1
        
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(num_vars)
        constraint = LinearConstraint(A, ub=np.full(num_constraints, 2))
        
        try:
            result = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
            if result.status not in (0, 1):
                return None
            x = np.round(result.x).astype(int)
            S = set()
            for i in range(num_vars):
                if x[i] == 1:
                    S.add(i + 1)
            if not S:
                return None
            return S
        except Exception:
            return None

    best_set = set()
    best_size = 0

    # Phase 1: Try ILP for a strong initial solution
    ilp_result = _solve_ilp(triples, n_max, deadline)
    if ilp_result is not None and len(ilp_result) > best_size:
        best_set = set(ilp_result)
        best_size = len(ilp_result)

    # Phase 2: Multi-restart greedy + local search
    num_restarts = 300
    for run in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break

        random.seed(12345 + run)
        S = set(all_elements)
        S = _greedy_removal(S, triples, elem_to_triples, n_max, deadline)
        S = _try_add_back(S, triples, elem_to_triples, n_max, deadline)
        S = _local_search(S, triples, elem_to_triples, n_max, deadline, max_iters=100)

        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Phase 3: Final polish on the best solution found
    if time.time() < deadline - 3 and best_set:
        polished = _local_search(best_set, triples, elem_to_triples, n_max, deadline, max_iters=500)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(polished)

    if not best_set:
        best_set = set(range(91, 181))

    return sorted(best_set)
