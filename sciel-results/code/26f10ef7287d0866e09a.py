"""Improved sum-free subset finder using exact branch-and-bound on conflict graph."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x != 0:
            values.add(x)
    return sorted(values)


def _is_sum_free(S):
    """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
    s = set(S)
    for a in s:
        for b in s:
            if a + b in s:
                return False
    return True


def _build_conflict_graph(elements):
    """
    Build conflict graph where vertex i conflicts with vertex j if adding both
    would violate sum-freeness in some way.
    
    Two elements x, y conflict if:
    - x + y is in elements (then {x, y, x+y} would have x+y = x+y, but we need all three in subset)
      Actually, the conflict is: if we include x and y, and x+y is also included, that's bad.
      But for pairwise conflict: x and y conflict if x+y is one of the elements.
    - Also, x and x conflict if 2*x is in elements (but this is self-loop, handled differently)
    
    For the independent set formulation:
    - Vertex i represents element elements[i]
    - Edge between i and j if elements[i] + elements[j] is in the set of all elements
      (because if we pick both i and j, and also pick the element equal to their sum, we get a violation)
    
    Wait, that's not quite right. The sum-free condition is: no a,b,c in S with a+b=c.
    So the constraint is a 3-uniform hypergraph, not a graph.
    
    Let me reconsider. For maximum independent set in a graph:
    - We want to select a subset with no edges.
    - But our constraint is 3-uniform: no triple (a,b,c) with a+b=c.
    
    Alternative approach: For each element c, if we include c, then for any pair (a,b) with a+b=c,
    we cannot include both a and b. This is like a "forbidden pair" constraint conditional on including c.
    
    This is harder to model as a simple graph independent set. Let me use a different approach:
    
    Branch and bound directly on the sum-free constraint.
    """
    n = len(elements)
    elem_set = set(elements)
    
    # For each element c, find all pairs (a, b) such that a + b = c
    # If we include c, we cannot include both a and b
    forbidden_pairs = [set() for _ in range(n)]
    for i, c in enumerate(elements):
        for j in range(i):
            a = elements[j]
            b = c - a
            if b in elem_set and b != a:
                # Find index of b
                k = elements.index(b) if b in elements else -1
                if k != -1:
                    forbidden_pairs[i].add((j, k) if j < k else (k, j))
            elif b == a:
                # 2*a = c, so if we include c, we can still include a (only one a)
                # No conflict for a single element
                pass
    
    # Actually, let me use a simpler branch and bound:
    # State: (index, current_sum_free_set, best_size)
    # At each step, decide to include or exclude elements[index]
    
    # For efficiency, precompute for each element x, which other elements conflict
    # Two elements x and y "conflict" in the sense that if both are in S, 
    # then x+y must not be in S. But this is a 3-way constraint.
    
    # Let me use a different structural approach:
    # Sort elements. Use the fact that if we take all odd numbers, that's sum-free.
    # Then try to add even numbers that don't create violations.
    
    # Actually, for the branch and bound, let me define:
    # can_add(x, S) = True if adding x to S keeps it sum-free
    # This requires checking: for all a, b in S, a + b != x and x + a != b (i.e., x + a not in S) and 2*x not in S
    # More precisely: x can be added to S iff:
    #   - For all a in S: a + x not in S and x + x not in S (i.e., 2*x not in S)
    #   - For all a, b in S: a + b != x (but this is implied if we check a+x not in S for all a? No)
    #   Actually: a + b = x means x is the sum. So we need: for no a, b in S is a + b = x.
    #   And for no a in S is a + x in S (which means a + x = c for some c in S).
    #   And 2*x not in S.
    
    return elements, elem_set


def _can_add(x, S_set):
    """Check if x can be added to S_set while maintaining sum-freeness."""
    # Check: no a, b in S_set with a + b = x
    # Check: no a in S_set with a + x in S_set (i.e., a + x = c for c in S_set)
    # Check: 2*x not in S_set
    for a in S_set:
        if a + x in S_set:
            return False
        if a + a == x:
            return False
    if 2 * x in S_set:
        return False
    return True


def _branch_and_bound(elements, time_budget=30.0, init_solution=None):
    """
    Branch and bound to find maximum sum-free subset.
    
    elements: sorted list of elements
    time_budget: maximum time in seconds
    init_solution: initial solution to use as lower bound
    """
    n = len(elements)
    elem_set = set(elements)
    start_time = time.time()
    
    # Initial lower bound
    if init_solution is not None:
        best_size = len(init_solution)
        best_set = set(init_solution)
    else:
        # Start with odd numbers as baseline
        best_set = set(x for x in elements if x % 2 != 0)
        best_size = len(best_set)
    
    # Sort elements by some heuristic: try to include large-magnitude elements first?
    # Or sort by how many conflicts they create
    # For now, use natural order but try both forward and backward
    
    # Precompute: for each element, which other elements would conflict if both included
    # conflict[i][j] = True if including both elements[i] and elements[j] would prevent
    # including elements[i]+elements[j] (if it exists)
    
    # Use recursive branch and bound with pruning
    # State: (idx, current_set)
    # Prune if len(current_set) + (n - idx) <= best_size
    
    def _solve(idx, current_set):
        nonlocal best_size, best_set
        nonlocal start_time
        
        if time.time() - start_time > time_budget:
            return
        
        # Pruning
        if len(current_set) + (n - idx) <= best_size:
            return
        
        if idx == n:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        x = elements[idx]
        
        # Option 1: Include x
        if _can_add(x, current_set):
            current_set.add(x)
            _solve(idx + 1, current_set)
            current_set.discard(x)
        
        # Option 2: Exclude x
        _solve(idx + 1, current_set)
    
    # Try with a timeout wrapper
    _solve(0, set())
    
    return list(best_set)


def _solve_instance(elements, time_budget=40.0):
    """Solve a single instance using branch and bound with fallbacks."""
    start = time.time()
    n = len(elements)
    
    # Get a good initial solution using greedy methods
    # Strategy 1: Odd numbers
    odd_subset = [x for x in elements if x % 2 != 0]
    best = list(odd_subset)
    
    # Strategy 2: Greedy with various orderings
    def _greedy_build(elements, order_key=None, rng=None):
        if order_key is not None:
            elems = sorted(elements, key=order_key)
        else:
            elems = list(elements)
            if rng is not None:
                rng.shuffle(elems)
        
        chosen = []
        chosen_set = set()
        
        for x in elems:
            if x in chosen_set:
                continue
            can_add = True
            for y in chosen_set:
                if y + x in chosen_set:
                    can_add = False
                    break
                if y + y == x and y != x:
                    can_add = False
                    break
            if can_add and 2 * x in chosen_set:
                can_add = False
            if can_add:
                chosen.append(x)
                chosen_set.add(x)
        
        return chosen
    
    orderings = [
        lambda x: x,
        lambda x: -x,
        lambda x: abs(x),
        lambda x: -abs(x),
        lambda x: (x > 0),
        lambda x: (x < 0),
    ]
    
    for order_key in orderings:
        if time.time() - start > time_budget * 0.2:
            break
        candidate = _greedy_build(elements, order_key=order_key)
        if len(candidate) > len(best):
            best = candidate
    
    # Randomized greedy
    rng = random.Random(42)
    for trial in range(10):
        if time.time() - start > time_budget * 0.3:
            break
        candidate = _greedy_build(elements, rng=rng)
        if len(candidate) > len(best):
            best = candidate
    
    # Now try branch and bound with remaining time
    remaining_time = time_budget - (time.time() - start) - 2.0  # 2s safety margin
    if remaining_time > 5.0 and n <= 150:
        # Branch and bound only feasible for smaller instances
        try:
            bb_result = _branch_and_bound(elements, time_budget=remaining_time, init_solution=best)
            if len(bb_result) > len(best):
                best = bb_result
        except Exception:
            pass
    
    # Final verification
    if not _is_sum_free(best):
        best = odd_subset
    
    return best


def construct():
    total_deadline = 120.0
    start = time.time()
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803),
    ]
    
    results = []
    for i, elements in enumerate(instances):
        elapsed = time.time() - start
        remaining = total_deadline - elapsed - 5.0  # 5s safety margin
        if remaining < 1.0:
            # Not enough time, use odd numbers
            results.append([x for x in elements if x % 2 != 0])
            continue
        
        # Allocate time based on instance size
        # Smaller instances can afford more exact search time
        if len(elements) <= 80:
            time_budget = min(remaining, 50.0)
        elif len(elements) <= 150:
            time_budget = min(remaining, 40.0)
        else:
            time_budget = min(remaining, 30.0)
        
        result = _solve_instance(elements, time_budget=time_budget)
        results.append(result)
    
    return results

# EVOLVE-BLOCK-END
