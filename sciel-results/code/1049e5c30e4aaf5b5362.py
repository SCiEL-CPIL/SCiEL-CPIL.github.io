"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 128}

def _is_valid(S):
    """Check if set S has no three distinct members a,b,c with all pair sums in S."""
    s = set(S)
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j+1, n):
                c = lst[k]
                if (a+b) in s and (a+c) in s and (b+c) in s:
                    return False
    return True

def _build_conflict_graph(n=128):
    """
    Build a conflict graph where vertices are 1..n.
    Two elements u,v conflict if there exists w such that {u,v,w} is a forbidden triple,
    i.e., all of u+v, u+w, v+w are in {1..n} and could be in the set.
    
    Actually, the conflict is: u and v cannot both be in a valid set if there exists w
    such that u+v, u+w, v+w are all <= n. Because if u,v,w are all selected, then
    u+v, u+w, v+w must also be selected for it to be a forbidden triple.
    
    Wait, that's not quite right. Let me reconsider.
    
    A forbidden triple is {a,b,c} where a+b, a+c, b+c are ALL in the set.
    So the constraint is: for any three distinct elements a,b,c in the set,
    it is NOT the case that all three pair sums are also in the set.
    
    This is a 3-uniform hypergraph constraint, not a pairwise one.
    However, we can think of it differently: two elements u,v "conflict" 
    with a third element w if {u,v,w} would form a forbidden triple given
    the current state.
    
    For branch-and-bound, we need to check: can we add element x to current set S?
    x can be added if for all pairs (a,b) in S, it's not the case that
    a+x, b+x, a+b are all in S ∪ {x}.
    
    Let me reformulate: the constraint is that no three elements a,b,c in the set
    have all their pair sums in the set. This is equivalent to saying:
    for any a,b,c in S with a<b<c, at least one of a+b, a+c, b+c is NOT in S.
    
    For the conflict graph approach to work directly, we'd need a pairwise constraint.
    But this is a 3-way constraint. So a simple independent set won't work directly.
    
    Instead, let's use a different approach: branch and bound with incremental checking.
    """
    # This function is kept for interface compatibility but the actual approach
    # uses incremental validity checking during branch-and-bound
    pass

def _can_add_element(S_mask, x, n=128):
    """Check if element x can be added to set represented by S_mask without creating a violation.
    A violation is created if there exist a,b in S such that a+x, b+x, a+b are all in S∪{x}."""
    new_mask = S_mask | (1 << x)
    
    # Find all a in S such that a+x is in S∪{x} and a+x <= n
    A = []
    for a in range(1, n + 1):
        if (S_mask >> a) & 1:
            ax = a + x
            if ax <= n and (new_mask >> ax) & 1:
                A.append(a)
    
    # For each pair (a,b) in A, check if a+b is in S∪{x}
    for i in range(len(A)):
        a = A[i]
        for j in range(i+1, len(A)):
            b = A[j]
            ab = a + b
            if ab <= n and (new_mask >> ab) & 1:
                return False
    return True

def _branch_and_bound(S_mask, remaining, current_size, best_size, best_mask, n=128, deadline=None):
    """
    Branch and bound to find maximum valid set.
    S_mask: bitmask of currently selected elements
    remaining: list of elements not yet decided (can be added or not)
    current_size: number of elements in S_mask
    best_size: best size found so far
    best_mask: best mask found so far
    """
    if deadline is not None and time.time() > deadline:
        return best_size, best_mask
    
    if not remaining:
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
        return best_size, best_mask
    
    # Upper bound: current_size + len(remaining)
    if current_size + len(remaining) <= best_size:
        return best_size, best_mask
    
    # Try to add the first element
    x = remaining[0]
    rest = remaining[1:]
    
    # Branch 1: Include x
    if _can_add_element(S_mask, x, n):
        new_mask = S_mask | (1 << x)
        best_size, best_mask = _branch_and_bound(
            new_mask, rest, current_size + 1, best_size, best_mask, n, deadline
        )
    
    # Branch 2: Exclude x
    best_size, best_mask = _branch_and_bound(
        S_mask, rest, current_size, best_size, best_mask, n, deadline
    )
    
    return best_size, best_mask

def _branch_and_bound_iterative(elements, n=128, deadline=None):
    """
    Iterative branch and bound with bitset-based pruning.
    elements: list of candidate elements to consider
    Uses a priority queue to explore most promising branches first.
    """
    import heapq
    
    if deadline is None:
        deadline = time.time() + 5.0
    
    # Sort elements in a good order (heuristic: larger elements first, 
    # as they're less likely to create conflicts)
    elements = sorted(elements, reverse=True)
    
    # State: (negative_upper_bound, S_mask, index, current_size)
    # We use a max-heap on upper bound (min-heap on negative)
    # Upper bound = current_size + (len(elements) - index)
    
    best_size = 0
    best_mask = 0
    
    # Initial state
    heap = []
    heapq.heappush(heap, (-len(elements), 0, 0, 0))
    
    visited = set()
    
    while heap:
        if time.time() > deadline:
            break
        
        neg_ub, S_mask, idx, cur_size = heapq.heappop(heap)
        ub = -neg_ub
        
        if cur_size + (len(elements) - idx) <= best_size:
            continue
        
        if (S_mask, idx) in visited:
            continue
        visited.add((S_mask, idx))
        
        if idx == len(elements):
            if cur_size > best_size:
                best_size = cur_size
                best_mask = S_mask
            continue
        
        x = elements[idx]
        rest_idx = idx + 1
        
        # Branch 1: Include x (if possible)
        if _can_add_element(S_mask, x, n):
            new_mask = S_mask | (1 << x)
            new_ub = cur_size + 1 + (len(elements) - rest_idx)
            if new_ub > best_size:
                heapq.heappush(heap, (-new_ub, new_mask, rest_idx, cur_size + 1))
        
        # Branch 2: Exclude x
        excl_ub = cur_size + (len(elements) - rest_idx)
        if excl_ub > best_size:
            heapq.heappush(heap, (-excl_ub, S_mask, rest_idx, cur_size))
    
    return best_size, best_mask

def _greedy_max_independent_set(elements, n=128):
    """
    Greedy approach: try to add elements in a smart order.
    Start with a good initial set and try to add more.
    """
    S_mask = 0
    S_set = set()
    
    # Try elements in descending order (larger elements are "safer")
    for x in sorted(elements, reverse=True):
        if _can_add_element(S_mask, x, n):
            S_mask |= (1 << x)
            S_set.add(x)
    
    return S_set, S_mask

def _local_search_improve(S_mask, S_set, n=128, deadline=None, max_iter=1000):
    """
    Try to improve the set by removing one element and adding two.
    This is a 1-2 exchange move.
    """
    if deadline is None:
        deadline = time.time() + 5.0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S_set)
        for i, x_remove in enumerate(S_list):
            if time.time() > deadline:
                break
            # Try removing x_remove and adding two elements
            S_without = S_set - {x_remove}
            S_mask_without = S_mask & ~(1 << x_remove)
            
            # Try adding pairs of elements not in S_without
            candidates = [x for x in range(1, n+1) if x not in S_without]
            
            for j in range(len(candidates)):
                if time.time() > deadline:
                    break
                a = candidates[j]
                for k in range(j+1, len(candidates)):
                    b = candidates[k]
                    # Try adding both a and b
                    S_try = S_without | {a, b}
                    if len(S_try) <= len(S_set):
                        continue
                    if _is_valid(S_try):
                        S_set = S_try
                        S_mask = 0
                        for elem in S_set:
                            S_mask |= (1 << elem)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return S_set, S_mask

def construct():
    n = P['n']
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy approach with different orderings
    best_set = set()
    best_size = 0
    
    # Try multiple greedy strategies
    all_elements = list(range(1, n + 1))
    
    # Descending order (larger elements first)
    S, mask = _greedy_max_independent_set(all_elements, n)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Ascending order
    S, mask = _greedy_max_independent_set(sorted(all_elements), n)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Random orderings
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        order = all_elements[:]
        rng.shuffle(order)
        S, mask = _greedy_max_independent_set(order, n)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 2: Branch and bound on a reduced problem
    # Focus on elements that could potentially be added
    # Start with a good base and try to add more
    
    base_set = set(range(64, n + 1))  # 65 elements
    if len(base_set) > best_size:
        best_size = len(base_set)
        best_set = base_set
    
    # Try to add elements from the lower range
    lower_elements = list(range(1, 64))
    
    # Greedy add lower elements to base
    S_mask = 0
    for x in base_set:
        S_mask |= (1 << x)
    S_set = set(base_set)
    
    for x in sorted(lower_elements, reverse=True):
        if _can_add_element(S_mask, x, n):
            S_mask |= (1 << x)
            S_set.add(x)
    
    if len(S_set) > best_size:
        best_size = len(S_set)
        best_set = S_set
    
    # Strategy 3: Local search to improve
    if time.time() < deadline - 10:
        S_mask = 0
        for x in best_set:
            S_mask |= (1 << x)
        S_set, S_mask = _local_search_improve(S_mask, set(best_set), n, deadline - 10, max_iter=500)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = S_set
    
    # Strategy 4: Branch and bound on small subsets
    # Try to find a good set by branching on a small number of key elements
    if time.time() < deadline - 5:
        # Focus on elements in the "middle" range where conflicts are most likely
        key_elements = list(range(30, 100))
        # Use a timeout for B&B
        bb_deadline = time.time() + min(10, deadline - time.time() - 2)
        bb_size, bb_mask = _branch_and_bound_iterative(key_elements, n, bb_deadline)
        
        # Convert mask to set
        bb_set = set()
        for i in range(1, n+1):
            if (bb_mask >> i) & 1:
                bb_set.add(i)
        
        if len(bb_set) > best_size:
            best_size = len(bb_set)
            best_set = bb_set
    
    # Final verification
    if _is_valid(best_set):
        return [int(x) for x in sorted(best_set)]
    else:
        # Fallback to known valid set
        return [int(x) for x in range(63, n + 1)]
# EVOLVE-BLOCK-END
