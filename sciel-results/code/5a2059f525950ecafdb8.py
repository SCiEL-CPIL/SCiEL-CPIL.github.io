"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
def _check_addition(triple, current_set):
    """
    Check if adding 'triple' to 'current_set' creates a K(2,2,2).
    """
    a, b, c = triple
    
    for a_prime in range(16):
        if a_prime == a or a_prime == b or a_prime == c:
            continue
        
        t4 = tuple(sorted((a_prime, b, c)))
        if t4 not in current_set:
            continue
            
        for b_prime in range(16):
            if b_prime == a or b_prime == b or b_prime == c or b_prime == a_prime:
                continue
            
            t2 = tuple(sorted((a, b_prime, c)))
            if t2 not in current_set:
                continue
                
            t6 = tuple(sorted((a_prime, b_prime, c)))
            if t6 not in current_set:
                continue
            
            for c_prime in range(16):
                if c_prime == a or c_prime == b or c_prime == c or c_prime == a_prime or c_prime == b_prime:
                    continue
                
                t1 = tuple(sorted((a, b, c_prime)))
                if t1 not in current_set:
                    continue
                
                t3 = tuple(sorted((a, b_prime, c_prime)))
                if t3 not in current_set:
                    continue
                
                t5 = tuple(sorted((a_prime, b, c_prime)))
                if t5 not in current_set:
                    continue
                
                t7 = tuple(sorted((a_prime, b_prime, c_prime)))
                if t7 in current_set:
                    return False
            
    return True

def _greedy_fallback(deadline):
    """Fast greedy construction as fallback."""
    all_triples = list(itertools.combinations(range(16), 3))
    random.shuffle(all_triples)
    
    current_set = set()
    
    for t in all_triples:
        if time.time() > deadline - 5:
            break
        if _check_addition(t, current_set):
            current_set.add(t)
    
    return current_set

def _solve_ilp(time_limit=80.0):
    """
    Solve the MILP to maximize the number of triples.
    Returns a set of triples or None if it fails.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
    except ImportError:
        return None, 0.0
    
    start_time = time.time()
    
    triple_list = list(itertools.combinations(range(16), 3))
    triple_to_idx = {t: i for i, t in enumerate(triple_list)}
    n_vars = len(triple_list)
    
    constraints_data = []
    seen_configs = set()
    
    for six_points in itertools.combinations(range(16), 6):
        a, b, c, d, e, f = six_points
        pairings = [
            [(a,b), (c,d), (e,f)],
            [(a,b), (c,e), (d,f)],
            [(a,b), (c,f), (d,e)],
            [(a,c), (b,d), (e,f)],
            [(a,c), (b,e), (d,f)],
            [(a,c), (b,f), (d,e)],
            [(a,d), (b,c), (e,f)],
            [(a,d), (b,e), (c,f)],
            [(a,d), (b,f), (c,e)],
            [(a,e), (b,c), (d,f)],
            [(a,e), (b,d), (c,f)],
            [(a,e), (b,f), (c,d)],
            [(a,f), (b,c), (d,e)],
            [(a,f), (b,d), (c,e)],
            [(a,f), (b,e), (c,d)],
        ]
        
        for pairing in pairings:
            norm_pairs = tuple(tuple(sorted(p)) for p in sorted(pairing))
            if norm_pairs in seen_configs:
                continue
            seen_configs.add(norm_pairs)
            
            (p1, p2, p3) = norm_pairs
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        transversals.append(tuple(sorted((x, y, z))))
            
            row_indices = [triple_to_idx[t] for t in transversals]
            constraints_data.append(row_indices)
    
    n_constraints = len(constraints_data)
    
    row_idx = []
    col_idx = []
    data = []
    
    for i, cols in enumerate(constraints_data):
        for c in cols:
            row_idx.append(i)
            col_idx.append(c)
            data.append(1)
    
    A = csr_matrix((data, (row_idx, col_idx)), shape=(n_constraints, n_vars))
    
    c_obj = -np.ones(n_vars)
    
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    lb = np.zeros(n_constraints)
    ub = np.full(n_constraints, 7)
    lin_cons = LinearConstraint(A, lb=lb, ub=ub)
    
    integrality = np.ones(n_vars)
    
    result = milp(
        c=c_obj,
        constraints=lin_cons,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit, "mip_rel_gap": 0.05}
    )
    
    elapsed = time.time() - start_time
    
    if result.x is not None:
        selected = set()
        for i, val in enumerate(result.x):
            if val > 0.5:
                selected.add(triple_list[i])
        if len(selected) > 0:
            return selected, elapsed
    
    return None, elapsed

def _local_search(initial_set, time_limit=10.0):
    """Improve the triple set using local search."""
    start_time = time.time()
    current_set = set(initial_set)
    all_triples = list(itertools.combinations(range(16), 3))
    
    while time.time() - start_time < time_limit:
        improved = False
        
        missing = [t for t in all_triples if t not in current_set]
        random.shuffle(missing)
        
        for t in missing:
            if _check_addition(t, current_set):
                current_set.add(t)
                improved = True
                break
        
        if not improved:
            if time.time() - start_time > time_limit * 0.9:
                break
                
            remove_candidates = list(current_set)
            random.shuffle(remove_candidates)
            
            for t_rem in remove_candidates[:20]:
                current_set.remove(t_rem)
                
                added_count = 0
                added_triples = []
                
                missing = [t for t in all_triples if t not in current_set and t != t_rem]
                random.shuffle(missing)
                
                for t_add in missing:
                    if _check_addition(t_add, current_set):
                        current_set.add(t_add)
                        added_triples.append(t_add)
                        added_count += 1
                        
                if added_count > 1:
                    improved = True
                    break
                else:
                    for t_add in added_triples:
                        current_set.remove(t_add)
                    current_set.add(t_rem)
                    
            if not improved:
                break

    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    g_set = _greedy_fallback(deadline)
    best_set = g_set
    best_size = len(g_set)
    
    remaining = deadline - time.time() - 15.0
    if remaining > 25.0:
        ilp_set, elapsed = _solve_ilp(time_limit=remaining)
        if ilp_set is not None and len(ilp_set) > best_size:
            best_set = ilp_set
            best_size = len(ilp_set)
    
    remaining = deadline - time.time() - 5.0
    if remaining > 2.0:
        ls_set = _local_search(best_set, time_limit=remaining)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(ls_set)
    
    return [list(t) for t in best_set]
# EVOLVE-BLOCK-END
