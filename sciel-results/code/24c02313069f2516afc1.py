import time
import math
from fractions import Fraction
import random

# EVOLVE-BLOCK-START

def _is_disjoint(m1, r1, m2, r2):
    """Check if residue classes (m1, r1) and (m2, r2) are disjoint."""
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _score(classes):
    """Compute the sum of 1/m for the given classes."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _can_add(classes, m_new, r_new):
    """Check if a new residue class can be added to the existing set."""
    for m, r in classes:
        if not _is_disjoint(m, r, m_new, r_new):
            return False
    return True

def _greedy_construct():
    """
    Greedy construction: try to add residue classes in order of increasing modulus.
    For each modulus, try all residues and pick the first one that is disjoint
    from all existing classes.
    """
    classes = []
    for m in range(20, 201):
        added = False
        for r in range(m):
            if _can_add(classes, m, r):
                classes.append([m, r])
                added = True
                break
    return classes

def _local_search(classes, time_budget=10.0):
    """
    Local search: try to improve the score by swapping residues or adding/removing classes.
    """
    start_time = time.time()
    classes = [list(c) for c in classes]
    best_score = _score(classes)
    
    for m_target in range(20, 201):
        if time.time() - start_time > time_budget:
            break
        if any(c[0] == m_target for c in classes):
            continue
        for r_target in range(m_target):
            if _can_add(classes, m_target, r_target):
                classes.append([m_target, r_target])
                new_score = _score(classes)
                if new_score > best_score:
                    best_score = new_score
                break
    
    for _ in range(100):
        if time.time() - start_time > time_budget:
            break
        improved = False
        for i, (m1, r1) in enumerate(classes):
            for r_new in range(m1):
                if r_new == r1:
                    continue
                old_r = classes[i][1]
                classes[i][1] = r_new
                valid = True
                for j, (m2, r2) in enumerate(classes):
                    if i == j:
                        continue
                    if not _is_disjoint(m1, r_new, m2, r2):
                        valid = False
                        break
                if valid:
                    for m_add in range(20, 201):
                        if any(c[0] == m_add for c in classes):
                            continue
                        for r_add in range(m_add):
                            if _can_add(classes, m_add, r_add):
                                classes.append([m_add, r_add])
                                new_score = _score(classes)
                                if new_score > best_score:
                                    best_score = new_score
                                else:
                                    classes.pop()
                                classes[i][1] = r_new
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                else:
                    classes[i][1] = old_r
        if not improved:
            break
    
    return classes

def _build_conflict_graph(candidates):
    """
    Build conflict graph for candidate (m, r) pairs.
    Returns: list of sets where conflict[i] contains indices j that conflict with i.
    """
    n = len(candidates)
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        m1, r1 = candidates[i]
        for j in range(i + 1, n):
            m2, r2 = candidates[j]
            if not _is_disjoint(m1, r1, m2, r2):
                conflicts[i].add(j)
                conflicts[j].add(i)
    return conflicts

def _greedy_color_bound(conflicts, weights, unprocessed):
    """
    Greedy coloring to compute upper bound on maximum weight independent set.
    unprocessed: set of candidate indices still available.
    Returns: upper bound on total weight of any independent set from unprocessed.
    """
    if not unprocessed:
        return 0.0
    
    # Order by weight descending
    ordered = sorted(unprocessed, key=lambda i: -weights[i])
    
    # Greedy coloring: assign colors to vertices
    color_of = {}
    color_classes = []  # list of sets of vertices per color
    
    for v in ordered:
        used_colors = set()
        for u in conflicts[v]:
            if u in color_of:
                used_colors.add(color_of[u])
        # Assign smallest available color
        c = 0
        while c in used_colors:
            c += 1
        color_of[v] = c
        if c >= len(color_classes):
            color_classes.append(set())
        color_classes[c].add(v)
    
    # Upper bound: sum of max weight in each color class
    bound = 0.0
    for cc in color_classes:
        bound += max(weights[v] for v in cc)
    
    return bound

def _branch_and_bound(candidates, weights, conflicts, deadline, initial_solution=None):
    """
    Branch and bound for maximum weight independent set.
    candidates: list of (m, r) pairs
    weights: list of 1/m values
    conflicts: conflict graph
    deadline: time limit
    initial_solution: set of indices already selected (from greedy)
    """
    start_time = time.time()
    n = len(candidates)
    
    # Initialize with greedy solution if provided
    best_set = set()
    best_weight = 0.0
    if initial_solution:
        best_set = set(initial_solution)
        best_weight = sum(weights[i] for i in best_set)
    
    # Available candidates not in initial solution
    available = set(range(n)) - best_set
    
    # Stack for DFS: (available_set, current_set, current_weight)
    # Use recursive approach with pruning
    
    def _dfs(available_set, current_set, current_weight, depth=0):
        nonlocal best_set, best_weight
        nonlocal start_time, deadline
        
        if time.time() > deadline:
            return
        
        if not available_set:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        # Upper bound using greedy coloring
        ub = current_weight + _greedy_color_bound(conflicts, weights, available_set)
        if ub <= best_weight + 1e-12:
            return
        
        # Choose pivot vertex: highest weight
        pivot = max(available_set, key=lambda v: weights[v])
        pivot_weight = weights[pivot]
        
        # Branch 1: Include pivot
        new_current = set(current_set)
        new_current.add(pivot)
        # Remove pivot and its neighbors from available
        new_available = set(available_set)
        new_available.discard(pivot)
        for neighbor in conflicts[pivot]:
            new_available.discard(neighbor)
        
        _dfs(new_available, new_current, current_weight + pivot_weight, depth + 1)
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude pivot
        new_available2 = set(available_set)
        new_available2.discard(pivot)
        _dfs(new_available2, current_set, current_weight, depth + 1)
    
    _dfs(available, best_set, best_weight)
    
    # Convert back to classes
    result = []
    for idx in best_set:
        m, r = candidates[idx]
        result.append([m, r])
    
    return result

def _is_valid(classes):
    """Check if the list of classes is valid (distinct moduli, disjoint residues)."""
    seen_moduli = set()
    for m, r in classes:
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if not (0 <= r < m):
            return False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            m1, r1 = classes[i]
            m2, r2 = classes[j]
            if not _is_disjoint(m1, r1, m2, r2):
                return False
    return True

def _branch_and_bound_solve(deadline):
    """
    Full branch and bound solver.
    Creates candidate set, builds conflict graph, and solves.
    """
    start_time = time.time()
    
    # Create all candidate (m, r) pairs
    candidates = []
    weights = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
            weights.append(1.0 / m)
    
    n = len(candidates)
    
    # Build conflict graph
    conflicts = _build_conflict_graph(candidates)
    
    # Get initial greedy solution
    greedy_classes = _greedy_construct()
    greedy_set = set()
    for m, r in greedy_classes:
        for idx, (cm, cr) in enumerate(candidates):
            if cm == m and cr == r:
                greedy_set.add(idx)
                break
    
    # Run branch and bound
    result = _branch_and_bound(candidates, weights, conflicts, deadline, greedy_set)
    
    # Validate result
    if not _is_valid(result):
        return greedy_classes
        
    return result

def construct():
    """
    Main construction function.
    Uses greedy initialization, local search, then branch-and-bound with greedy coloring bounds.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (150s limit)
    
    # Phase 1: Greedy construction
    classes = _greedy_construct()
    
    if time.time() > deadline - 30:
        if _is_valid(classes):
            return classes
        return []
    
    # Phase 2: Local search to improve greedy solution
    classes = _local_search(classes, time_budget=15.0)
    
    if time.time() > deadline - 10:
        if _is_valid(classes):
            return classes
        return _greedy_construct()
    
    # Phase 3: Branch and bound with greedy coloring
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        bb_classes = _branch_and_bound_solve(time.time() + remaining_time - 2.0)
        if _is_valid(bb_classes) and _score(bb_classes) > _score(classes):
            classes = bb_classes
    
    # Final validation
    if not _is_valid(classes):
        classes = _greedy_construct()
        if not _is_valid(classes):
            classes = []
            
    return classes

# EVOLVE-BLOCK-END
