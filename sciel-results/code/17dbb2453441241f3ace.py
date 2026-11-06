"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

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
            if c > n_max:
                continue
            if c <= b:
                continue
            triples.append((a, b, c))
    return triples

def _local_polish(S, triples, elem_to_triples, time_limit=5.0):
    """Try to add back elements that were removed, if possible."""
    S = set(S)
    n_max = max(S) if S else 180
    all_elements = set(range(1, n_max + 1))
    candidates = all_elements - S
    
    # Precompute conflict counts for current S
    triple_incount = {}
    for idx, (a, b, c) in enumerate(triples):
        cnt = (a in S) + (b in S) + (c in S)
        triple_incount[idx] = cnt
    
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # For each candidate, check if adding it creates a conflict
        for x in sorted(candidates, reverse=True):
            if x in S:
                continue
            # Check if adding x creates any triple with all 3 in S
            can_add = True
            for tidx in elem_to_triples.get(x, []):
                a, b, c = triples[tidx]
                others = [y for y in (a, b, c) if y != x]
                if all(y in S for y in others):
                    can_add = False
                    break
            if can_add:
                S.add(x)
                candidates.discard(x)
                # Update triple_incount
                for tidx in elem_to_triples.get(x, []):
                    triple_incount[tidx] += 1
                improved = True
    
    return sorted(S)

def construct():
    start_time = time.time()
    deadline = 60.0  # seconds
    
    n_max = 180
    all_elements = list(range(1, n_max + 1))
    
    # Build the hypergraph: edges are triples (a,b,c) that violate the constraint
    triples = _build_triples(n_max)
    n_triples = len(triples)
    
    # Build incidence
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    for idx, (a, b, c) in enumerate(triples):
        for x in (a, b, c):
            elem_to_triples[x].append(idx)
    
    # ILP formulation: maximize sum(x_i) subject to x_a + x_b + x_c <= 2 for each triple
    # x_i in {0, 1}
    
    # Objective: maximize sum(x_i) => minimize -sum(x_i)
    c_obj = -np.ones(n_max)
    
    # Constraints: for each triple (a,b,c), x_a + x_b + x_c <= 2
    # Build constraint matrix
    if n_triples > 0:
        A_ub = np.zeros((n_triples, n_max))
        for idx, (a, b, c) in enumerate(triples):
            A_ub[idx, a-1] = 1
            A_ub[idx, b-1] = 1
            A_ub[idx, c-1] = 1
        b_ub = np.full(n_triples, 2)
        constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    else:
        constraints = LinearConstraint(np.zeros((1, n_max)), 0, 1)
    
    bounds = Bounds(0, 1)
    integrality = np.ones(n_max)  # all variables integer
    
    # Solve ILP
    try:
        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if result.status == 0 and result.x is not None:
            # Extract solution
            sol = np.round(result.x).astype(int)
            S_ilp = set(i+1 for i in range(n_max) if sol[i] == 1)
            
            # Local polish: try to add more elements
            S_polished = _local_polish(S_ilp, triples, elem_to_triples, time_limit=10.0)
            
            elapsed = time.time() - start_time
            if elapsed < deadline - 2.0:
                return S_polished
            
            return sorted(S_ilp)
        else:
            # Fallback to greedy
            pass
    except Exception:
        pass
    
    # Fallback: greedy approach
    S = set(all_elements)
    
    triple_incount = [0] * n_triples
    for idx, (a, b, c) in enumerate(triples):
        cnt = (a in S) + (b in S) + (c in S)
        triple_incount[idx] = cnt
    
    elem_conflict_count = {i: 0 for i in range(1, n_max + 1)}
    for idx, cnt in enumerate(triple_incount):
        if cnt == 3:
            a, b, c = triples[idx]
            elem_conflict_count[a] += 1
            elem_conflict_count[b] += 1
            elem_conflict_count[c] += 1
    
    total_conflicts = sum(1 for cnt in triple_incount if cnt == 3)
    
    while total_conflicts > 0:
        elapsed = time.time() - start_time
        if elapsed > deadline - 2.0:
            break
        
        max_conflict = 0
        best_elem = None
        for x in S:
            if elem_conflict_count[x] > max_conflict:
                max_conflict = elem_conflict_count[x]
                best_elem = x
        
        if best_elem is None or max_conflict == 0:
            break
        
        S.remove(best_elem)
        
        for tidx in elem_to_triples[best_elem]:
            old_cnt = triple_incount[tidx]
            if old_cnt == 3:
                total_conflicts -= 1
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x != best_elem:
                        elem_conflict_count[x] -= 1
            triple_incount[tidx] = old_cnt - 1
    
    result = sorted(S)
    if not result:
        result = list(range(91, 181))
    
    return result

# EVOLVE-BLOCK-END
