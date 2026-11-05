import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_N = 256
_DEADLINE = 85.0

def _construct_neighbors():
    """
    Precompute neighbors for each node 1..256.
    i and j are neighbors if (i*j) % (i+j) == 0.
    """
    neighbors = [set() for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors

def _milp_solve(neighbors, time_limit):
    """
    Solve the maximum independent set problem using MILP.
    Variables: x_i in {0,1} for i in 1..256
    Maximize: sum(x_i)
    Subject to: x_i + x_j <= 1 for all edges (i,j)
    """
    n = _N
    # Objective: maximize sum of x_i
    c = np.ones(n)
    
    # Constraints: for each edge (i,j), x_i + x_j <= 1
    A_rows = []
    b_upper = []
    for i in range(1, n + 1):
        for j in neighbors[i]:
            if i < j:  # avoid duplicates
                row = np.zeros(n)
                row[i-1] = 1
                row[j-1] = 1
                A_rows.append(row)
                b_upper.append(1)
    
    if not A_rows:
        return list(range(1, n + 1))
    
    A = np.array(A_rows)
    
    # Variable bounds: 0 <= x_i <= 1, integer
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer
    integrality = np.ones(n)
    
    # Linear constraint: A @ x <= b_upper
    constraint = LinearConstraint(A, ub=np.array(b_upper))
    
    try:
        result = milp(
            c=-c,  # minimize -sum(x_i) to maximize sum(x_i)
            constraints=constraint,
            integrality=integrality,
            bounds=bounds
        )
        
        if result.status == 0 and result.x is not None:
            solution = [i + 1 for i in range(n) if result.x[i] > 0.5]
            return solution
        else:
            return None
    except Exception:
        return None

def _can_add(x, S_set, neighbors):
    """Check if x can be added to S_set."""
    return not (neighbors[x] & S_set)

def _spectral_init(neighbors, rng):
    """
    Spectral clustering approach: compute eigenvector of conflict graph,
    use it to identify a low-conflict core, then greedily expand.
    """
    # Build adjacency matrix of the conflict graph
    A = np.zeros((_N + 1, _N + 1))
    for i in range(1, _N + 1):
        for j in neighbors[i]:
            A[i, j] = 1
            A[j, i] = 1
    
    # Compute largest eigenvector (principal component)
    # Use power iteration for efficiency
    v = np.ones(_N + 1)
    v[0] = 0
    v = v / np.linalg.norm(v)
    
    for _ in range(100):
        new_v = A @ v
        norm = np.linalg.norm(new_v)
        if norm < 1e-15:
            break
        v = new_v / norm
    
    # Eigenvector values: higher value = more "central" in conflict graph
    # For independent set, we want nodes with LOW eigenvector values (peripheral)
    # Sort by eigenvector value ascending (least conflicting first)
    order = list(range(1, _N + 1))
    order.sort(key=lambda x: v[x])
    
    # Greedy expansion using spectral ordering
    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, neighbors):
            S.append(x)
            S_set.add(x)
    
    return S, S_set

def _greedy_init(neighbors, rng, mode='degree'):
    """
    Greedy initialization.
    mode: 'degree', 'random', 'least_conflicting', 'spectral'
    """
    if mode == 'degree':
        degrees = [len(neighbors[i]) for i in range(_N + 1)]
        order = list(range(1, _N + 1))
        order.sort(key=lambda x: (degrees[x], x))
    elif mode == 'random':
        order = list(range(1, _N + 1))
        rng.shuffle(order)
    elif mode == 'spectral':
        S, S_set = _spectral_init(neighbors, rng)
        return S, S_set
    else:
        order = list(range(1, _N + 1))

    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, neighbors):
            S.append(x)
            S_set.add(x)
    return S, S_set

def _local_search(S, neighbors, rng, time_limit):
    """
    Local search to improve S.
    1. Try to add any available node.
    2. Try 1-for-k swaps (remove u, greedily refill).
    3. Try 2-for-k swaps (remove two, greedily refill).
    """
    S = list(S)
    S_set = set(S)
    complement = set(range(1, _N + 1)) - S_set
    start = time.time()
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Phase 1: Add any node that fits
        for x in list(complement):
            if _can_add(x, S_set, neighbors):
                S.append(x)
                S_set.add(x)
                complement.discard(x)
                improved = True
        
        if time.time() >= time_limit:
            break
            
        # Phase 2: Try 1-for-k swaps
        if not improved:
            best_size = len(S)
            found_improvement = False
            
            for u in list(S):
                if time.time() >= time_limit:
                    break
                
                S_set.remove(u)
                complement.add(u)
                
                new_S_set = set(S_set)
                new_complement = set(complement)
                
                for x in list(new_complement):
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        new_complement.discard(x)
                        
                if len(new_S_set) > best_size:
                    S = list(new_S_set)
                    S_set = new_S_set
                    complement = set(range(1, _N + 1)) - S_set
                    best_size = len(new_S_set)
                    improved = True
                    found_improvement = True
                    break 
                else:
                    S_set.add(u)
                    complement.discard(u)
            
            if found_improvement:
                continue
        
        # Phase 3: Try 2-for-k swaps
        if not improved:
            best_size = len(S)
            found_improvement = False
            S_list = list(S)
            n = len(S_list)
            
            for i in range(n):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() >= time_limit:
                        break
                    
                    r1, r2 = S_list[i], S_list[j]
                    S_set.discard(r1)
                    S_set.discard(r2)
                    complement.add(r1)
                    complement.add(r2)
                    
                    new_S_set = set(S_set)
                    new_complement = set(complement)
                    
                    for x in list(new_complement):
                        if _can_add(x, new_S_set, neighbors):
                            new_S_set.add(x)
                            new_complement.discard(x)
                            
                    if len(new_S_set) > best_size:
                        S = list(new_S_set)
                        S_set = new_S_set
                        complement = set(range(1, _N + 1) - S_set)
                        best_size = len(new_S_set)
                        improved = True
                        found_improvement = True
                        break
                    else:
                        S_set.add(r1)
                        S_set.add(r2)
                        complement.discard(r1)
                        complement.discard(r2)
                
                if found_improvement:
                    break
            
            if found_improvement:
                continue

    return S

def _simulated_annealing(S, neighbors, rng, time_limit):
    """
    Simulated annealing to escape local optima.
    Try random swaps and accept with probability based on temperature.
    """
    S = list(S)
    S_set = set(S)
    start = time.time()
    
    # Initial temperature
    T = 2.0
    cooling = 0.99
    
    best_S = list(S)
    best_size = len(S)
    
    while time.time() < time_limit and T > 0.01:
        # Pick a random element from S and a random element from complement
        S_list = list(S)
        complement = set(range(1, _N + 1)) - S_set
        
        if not S_list or not complement:
            break
            
        u = rng.choice(S_list)
        v = rng.choice(list(complement))
        
        # Try swap: remove u, add v
        if u == v:
            continue
            
        S_set.remove(u)
        S_set.add(v)
        
        # Check validity
        valid = True
        for x in S_set:
            if x != v and (x * v) % (x + v) == 0:
                valid = False
                break
        
        if valid:
            # New size is same (swap), but might enable further additions
            new_S_set = set(S_set)
            new_complement = set(range(1, _N + 1)) - new_S_set
            
            # Greedily add any available nodes
            for x in list(new_complement):
                if _can_add(x, new_S_set, neighbors):
                    new_S_set.add(x)
                    new_complement.discard(x)
            
            new_size = len(new_S_set)
            
            if new_size > best_size:
                best_S = list(new_S_set)
                best_size = new_size
                S = best_S
                S_set = best_S_set
            elif new_size >= len(S):
                # Accept with some probability to allow exploration
                if rng.random() < 0.5:
                    pass
                else:
                    S_set.remove(v)
                    S_set.add(u)
            else:
                # Revert
                S_set.remove(v)
                S_set.add(u)
        else:
            # Invalid swap, revert
            S_set.remove(v)
            S_set.add(u)
        
        T *= cooling
    
    return best_S

def _two_swap_search(S, neighbors, rng, time_limit):
    """
    Try 2-for-3 swaps: remove two elements from S, add up to three from remaining.
    """
    S = list(S)
    S_set = set(S)
    remaining = set(range(1, _N + 1)) - S_set
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            if time.time() >= time_limit:
                break
            for j in range(i + 1, n):
                if time.time() >= time_limit:
                    break
                r1, r2 = S_list[i], S_list[j]
                S_set.discard(r1)
                S_set.discard(r2)
                
                new_S_set = set(S_set)
                added = 0
                for x in list(remaining):
                    if added >= 3:
                        break
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        added += 1
                
                if added > 2:
                    S = list(new_S_set)
                    S_set = new_S_set
                    remaining = set(range(1, _N + 1)) - S_set
                    improved = True
                    break
                else:
                    S_set.add(r1)
                    S_set.add(r2)
            if improved:
                break
    return S

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    neighbors = _construct_neighbors()
    
    best_S = []
    best_size = 0
    
    # First, try MILP for exact solution
    if time.time() - t0 < _DEADLINE - 30:
        milp_result = _milp_solve(neighbors, _DEADLINE - 30)
        if milp_result is not None and len(milp_result) > best_size:
            best_S = milp_result
            best_size = len(milp_result)
    
    # Try multiple random restarts of greedy init
    for i in range(100):
        if time.time() - t0 > _DEADLINE - 15:
            break
        
        # Variation 1: Degree sorted
        if i % 5 == 0:
            S, S_set = _greedy_init(neighbors, rng, mode='degree')
        elif i % 5 == 1:
            S, S_set = _greedy_init(neighbors, rng, mode='random')
        elif i % 5 == 2:
            # Least conflicting greedy (dynamic)
            S, S_set = [], set()
            remaining = set(range(1, _N + 1))
            while remaining:
                best_x = None
                best_c = None
                for x in remaining:
                    c = len(neighbors[x] & S_set)
                    if best_c is None or c < best_c or (c == best_c and x < best_x):
                        best_x = x
                        best_c = c
                    if c == 0:
                        break
                if best_c == 0:
                    S.append(best_x)
                    S_set.add(best_x)
                    remaining.discard(best_x)
                else:
                    break
        elif i % 5 == 3:
            # Spectral init
            S, S_set = _greedy_init(neighbors, rng, mode='spectral')
        else:
            # Randomized spectral: perturb eigenvector ordering
            S, S_set = _greedy_init(neighbors, rng, mode='random')
        
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    # Local search on best found
    if time.time() - t0 < _DEADLINE - 10:
        S_final = _local_search(best_S, neighbors, rng, _DEADLINE - 10)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            
    # Two-swap search
    if time.time() - t0 < _DEADLINE - 5:
        S_final = _two_swap_search(best_S, neighbors, rng, _DEADLINE - 5)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            
    # Simulated annealing polish
    if time.time() - t0 < _DEADLINE - 2:
        S_final = _simulated_annealing(best_S, neighbors, rng, _DEADLINE - 2)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
          
    # Final local search polish
    if time.time() - t0 < _DEADLINE - 1:
        S_final = _local_search(best_S, neighbors, rng, _DEADLINE - 1)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
          
    return best_S
