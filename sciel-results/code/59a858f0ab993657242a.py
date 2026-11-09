"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _build_conflict_matrix():
    """Build a 256x256 boolean matrix where conflict[i,j] is True if (i+1)+(j+1) is a perfect square."""
    n = 256
    conflict = np.zeros((n, n), dtype=bool)
    # Perfect squares up to 512: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484
    squares = [s*s for s in range(1, 23) if s*s <= 512]
    for i in range(n):
        for j in range(i, n):  # symmetric
            s = (i+1) + (j+1)
            if s in squares:
                conflict[i, j] = True
                conflict[j, i] = True
    return conflict

def _validate_solution(solution, conflict):
    """Verify that no pair a<=b in solution has a+b as a perfect square."""
    n = len(solution)
    sol_set = set(solution)
    for i in range(n):
        for j in range(i, n):
            a = solution[i] - 1  # 0-indexed
            b = solution[j] - 1  # 0-indexed
            if conflict[a, b]:
                return False
    return True

def _greedy_construct(conflict):
    """Greedy: pick nodes in order of fewest conflicts first."""
    n = 256
    # Count conflicts for each node
    conflict_count = conflict.sum(axis=1)
    order = np.argsort(conflict_count)  # nodes with fewest conflicts first
    selected = []
    selected_set = set()
    for node in order:
        node = int(node)
        # Check if this node conflicts with any already selected
        if any(conflict[node, s] for s in selected_set):
            continue
        selected.append(node + 1)  # 1-indexed
        selected_set.add(node)
    return sorted(selected)

def _solve_ilp(conflict, timeout=45):
    """Solve maximum independent set via MILP using scipy's milp."""
    n = 256
    start_time = time.time()
    
    # Variables: x_i in [0,1], maximize sum x_i
    # Constraints: x_i + x_j <= 1 for all conflicting pairs (i,j), i < j
    c = -np.ones(n)  # minimize -sum x_i
    
    # Build constraint matrix for all conflicting pairs
    rows = []
    cols = []
    rhs = []
    for i in range(n):
        for j in range(i+1, n):
            if conflict[i, j]:
                rows.append(len(rhs))
                cols.append(i)
                rhs.append(1)
                rows.append(len(rhs))
                cols.append(j)
                rhs.append(1)
    
    if len(rhs) == 0:
        return list(range(1, 257))
    
    # Use scipy sparse matrix
    from scipy.sparse import coo_matrix
    A = coo_matrix((rhs, (rows, cols)), shape=(len(rhs), n))
    
    constraints = LinearConstraint(A, ub=1)
    bounds = Bounds(lb=0, ub=1)
    
    try:
        result = milp(c, constraints=constraints, integrality=np.ones(n), bounds=bounds)
        if result.status == 0:
            solution = [int(i+1) for i in range(n) if result.x[i] > 0.5]
            if solution and _validate_solution(solution, conflict):
                return solution
    except Exception:
        pass
    
    return None

def _simulated_annealing(conflict, start_solution, max_iters=50000, initial_temp=1.0, final_temp=0.01, deadline=None):
    """Simulated annealing to improve the solution."""
    n = 256
    current = set(start_solution)
    current_size = len(current)
    best = set(current)
    best_size = current_size
    
    temp = initial_temp
    cooling = (final_temp / initial_temp) ** (1.0 / max_iters)
    
    # Precompute conflict neighbors for each node
    neighbors = [np.where(conflict[i])[0].tolist() for i in range(n)]
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Choose a random move: remove a node and/or add a non-conflicting node
        move_type = random.random()
        
        if move_type < 0.4 and current:
            # Remove a random node
            to_remove = random.choice(list(current))
            new_set = current - {to_remove}
            delta = -1
        elif move_type < 0.7:
            # Add a random non-conflicting node
            candidates = [i for i in range(n) if i not in current and not any(conflict[i, s] for s in current)]
            if not candidates:
                continue
            to_add = random.choice(candidates)
            new_set = current | {to_add}
            delta = 1
        elif move_type < 0.85:
            # Swap: remove one, add another
            if not current:
                continue
            to_remove = random.choice(list(current))
            temp_current = current - {to_remove}
            candidates = [i for i in range(n) if i not in temp_current and not any(conflict[i, s] for s in temp_current)]
            if not candidates:
                continue
            to_add = random.choice(candidates)
            new_set = temp_current | {to_add}
            delta = 0
        else:
            # Local search: try to add any non-conflicting node
            candidates = [i for i in range(n) if i not in current and not any(conflict[i, s] for s in current)]
            if not candidates:
                continue
            to_add = random.choice(candidates)
            new_set = current | {to_add}
            delta = 1
        
        new_size = len(new_set)
        
        if delta >= 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
            current = new_set
            current_size = new_size
            if current_size > best_size:
                best = set(current)
                best_size = current_size
        
        temp *= cooling
    
    return sorted(best)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Build conflict matrix
    conflict = _build_conflict_matrix()
    
    # First, get a good initial solution via greedy
    greedy_sol = _greedy_construct(conflict)
    best = greedy_sol
    best_size = len(best)
    
    # Try ILP if time permits
    if time.time() < deadline - 10:
        ilp_sol = _solve_ilp(conflict, timeout=40)
        if ilp_sol and len(ilp_sol) > best_size:
            best = ilp_sol
            best_size = len(ilp_sol)
    
    # Try simulated annealing to improve
    if time.time() < deadline - 5:
        sa_sol = _simulated_annealing(conflict, best, max_iters=100000, 
                                       initial_temp=1.0, final_temp=0.005,
                                       deadline=deadline - 3)
        if len(sa_sol) > best_size:
            best = sa_sol
            best_size = len(best)
    
    # Final verification
    if not _validate_solution(best, conflict):
        # Fallback to greedy
        best = greedy_sol
    
    return best

# EVOLVE-BLOCK-END
