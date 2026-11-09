"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START

def _build_milp_data():
    """
    Precompute the MILP formulation data.
    Variables: x_i for each of the 560 triples (binary).
    Objective: maximize sum(x_i).
    Constraints: for every K(2,2,2) configuration, sum of its 8 triple vars <= 7.
    """
    all_triples = list(itertools.combinations(range(16), 3))
    n_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Find all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs of points: {a,a'}, {b,b'}, {c,c'}
    # where a,a',b,b',c,c' are all distinct.
    # The 8 transversal triples are all combinations of one from each pair.
    k222_constraints = []  # list of lists of triple indices
    
    points = range(16)
    # Choose 3 disjoint pairs from 16 points
    # First pick 6 distinct points, then partition into 3 pairs
    for six_points in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = six_points
        # All ways to partition 6 points into 3 unordered pairs
        # There are 15 such partitions
        # Pair (p0,p1) with one of (p2,p3), (p2,p4), (p2,p5), (p3,p4), (p3,p5), (p4,p5)
        # Then the remaining two form the third pair
        pairings = [
            ((p0, p1), (p2, p3), (p4, p5)),
            ((p0, p1), (p2, p4), (p3, p5)),
            ((p0, p1), (p2, p5), (p3, p4)),
            ((p0, p2), (p1, p3), (p4, p5)),
            ((p0, p2), (p1, p4), (p3, p5)),
            ((p0, p2), (p1, p5), (p3, p4)),
            ((p0, p3), (p1, p2), (p4, p5)),
            ((p0, p3), (p1, p4), (p2, p5)),
            ((p0, p3), (p1, p5), (p2, p4)),
            ((p0, p4), (p1, p2), (p3, p5)),
            ((p0, p4), (p1, p3), (p2, p5)),
            ((p0, p4), (p1, p5), (p2, p3)),
            ((p0, p5), (p1, p2), (p3, p4)),
            ((p0, p5), (p1, p3), (p2, p4)),
            ((p0, p5), (p1, p4), (p2, p3)),
        ]
        
        for (pair_a, pair_b, pair_c) in pairings:
            # The 8 transversals
            a, a_prime = pair_a
            b, b_prime = pair_b
            c, c_prime = pair_c
            
            transversals = [
                tuple(sorted((a, b, c))),
                tuple(sorted((a, b, c_prime))),
                tuple(sorted((a, b_prime, c))),
                tuple(sorted((a, b_prime, c_prime))),
                tuple(sorted((a_prime, b, c))),
                tuple(sorted((a_prime, b, c_prime))),
                tuple(sorted((a_prime, b_prime, c))),
                tuple(sorted((a_prime, b_prime, c_prime))),
            ]
            
            # Verify all 8 are distinct
            if len(set(transversals)) != 8:
                continue
            
            # Get indices
            idxs = []
            valid = True
            for t in transversals:
                if t in triple_to_idx:
                    idxs.append(triple_to_idx[t])
                else:
                    valid = False
                    break
            
            if valid:
                k222_constraints.append(idxs)
    
    return all_triples, triple_to_idx, k222_constraints

def _solve_milp(all_triples, triple_to_idx, k222_constraints, time_limit=100.0):
    """
    Solve the MILP to maximize the number of K(2,2,2)-free triples.
    """
    n = len(all_triples)
    
    # Objective: maximize sum(x_i) => minimize -sum(x_i)
    c = -np.ones(n)
    
    # Build constraint matrix
    n_constraints = len(k222_constraints)
    
    if n_constraints == 0:
        return set(range(n))
    
    # Use sparse-friendly approach: build A as a list of (row, col, val)
    # A[i, j] = 1 if triple j is in K(2,2,2) constraint i
    # Build using a dense matrix for simplicity (560 cols, constraints rows)
    # Actually let's count constraints first
    print(f"MILP: {n} variables, {n_constraints} constraints")
    
    # Build A matrix
    A = np.zeros((n_constraints, n), dtype=np.float64)
    for i, idxs in enumerate(k222_constraints):
        for j in idxs:
            A[i, j] = 1.0
    
    # Constraints: A @ x <= 7 for each row
    lb = np.zeros(n_constraints)
    ub = np.full(n_constraints, 7.0)
    
    # Variable bounds: 0 <= x_i <= 1
    var_lb = np.zeros(n)
    var_ub = np.ones(n)
    
    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(var_lb, var_ub)
    
    # Integrality: all binary
    integrality = np.ones(n)
    
    start_time = time.time()
    
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "mip_rel_gap": 0.01}
        )
    except Exception as e:
        print(f"MILP error: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"MILP solved in {elapsed:.1f}s, status: {result.status}, obj: {result.fun}")
    
    if result.x is not None:
        solution = set(int(round(result.x[i])) for i in range(n) if result.x[i] > 0.5)
        return solution
    
    return None

def _greedy_initial():
    """Generate an initial valid triple set using greedy."""
    all_triples = list(itertools.combinations(range(16), 3))
    random.shuffle(all_triples)
    
    current_set = set()
    
    # Precompute K(2,2,2) membership for speed
    # For each triple, store which K(2,2,2)s it belongs to
    # And for each K(2,2,2), track how many of its 8 triples are present
    # Actually, let's just use the slow check for the greedy (it's fast enough for 560 triples)
    
    for t in all_triples:
        if _check_addition(t, current_set):
            current_set.add(t)
            
    return current_set

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

def _local_search(initial_set, time_limit=10.0):
    """Improve the triple set using local search."""
    start_time = time.time()
    current_set = set(initial_set)
    current_size = len(current_set)
    all_triples = list(itertools.combinations(range(16), 3))
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add any missing triple
        missing = [t for t in all_triples if t not in current_set]
        random.shuffle(missing)
        
        for t in missing:
            if _check_addition(t, current_set):
                current_set.add(t)
                current_size += 1
                improved = True
                break
        
        if not improved:
            # 2. Try a 1-remove, 2-add move (net +1)
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
                    current_size += (added_count - 1)
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
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Get a good initial solution via greedy
    print("Phase 1: Greedy initialization...")
    greedy_set = _greedy_initial()
    best_set = set(greedy_set)
    best_size = len(best_set)
    print(f"Greedy found {best_size} triples")
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 60.0:
        print("Phase 2: Local search...")
        ls_set = _local_search(greedy_set, time_limit=20.0)
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_set = ls_set
            print(f"Local search improved to {best_size}")
    
    # Phase 3: Try MILP
    if time.time() < deadline - 10.0:
        print("Phase 3: MILP optimization...")
        try:
            all_triples, triple_to_idx, k222_constraints = _build_milp_data()
            print(f"Built MILP data: {len(all_triples)} triples, {len(k222_constraints)} K(2,2,2) constraints")
            
            remaining_time = deadline - time.time() - 5.0  # 5s safety margin
            if remaining_time > 5.0:
                milp_solution = _solve_milp(all_triples, triple_to_idx, k222_constraints, time_limit=remaining_time)
                
                if milp_solution is not None:
                    # Convert indices back to triples
                    milp_triples = set(all_triples[i] for i in milp_solution)
                    if len(milp_triples) > best_size:
                        best_size = len(milp_triples)
                        best_set = milp_triples
                        print(f"MILP improved to {best_size}")
                    else:
                        print(f"MILP got {len(milp_triples)}, not better than {best_size}")
        except Exception as e:
            print(f"MILP phase error: {e}")
    
    # Phase 4: If we have time, do more local search on the best solution
    if time.time() < deadline - 5.0:
        print("Phase 4: Final local search polish...")
        remaining = deadline - time.time() - 2.0
        if remaining > 2.0:
            final_set = _local_search(best_set, time_limit=remaining)
            if len(final_set) > best_size:
                best_size = len(final_set)
                best_set = final_set
                print(f"Final polish improved to {best_size}")
    
    print(f"Final result: {best_size} triples")
    return [list(t) for t in best_set]

# EVOLVE-BLOCK-END
