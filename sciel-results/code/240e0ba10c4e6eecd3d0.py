import math
import time
import random

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    N = 511
    C = 512
    
    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Weights: w[a] = 1/(512 - a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)
    
    # Precompute prime factor masks for each number
    pf = [[] for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            pf[multiple].append(p)
    
    # Build conflict masks using bitsets
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        if a == 1:
            continue
        for b in range(a + 1, N + 1):
            if not pf[a] or not pf[b]:
                continue
            common = False
            for p in pf[a]:
                if p in pf[b]:
                    common = True
                    break
            if common:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    def get_score(selected):
        return sum(w[a] for a in selected)
    
    # Greedy initialization
    def greedy_init(order):
        S = set()
        S.add(1)
        for a in order:
            if a in S:
                continue
            can_add = True
            for b in S:
                if (conflict_mask[b] >> a) & 1:
                    can_add = False
                    break
            if can_add:
                S.add(a)
        return S
    
    orders = [
        list(range(N, 0, -1)),
        list(range(2, N + 1)),
    ]
    
    best_S = None
    best_score = -1
    for order in orders:
        S_greedy = greedy_init(order)
        score = get_score(S_greedy)
        if score > best_score:
            best_score = score
            best_S = S_greedy.copy()
    
    rng = random.Random(42)
    for _ in range(50):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_greedy = greedy_init(order)
        score = get_score(S_greedy)
        if score > best_score:
            best_score = score
            best_S = S_greedy.copy()
    
    # Branch and bound for maximum weight independent set
    # Sort vertices by weight descending for better pruning
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda a: -w[a])
    
    # Precompute for each vertex, which vertices have higher or equal weight
    # and don't conflict with it (for upper bound computation)
    
    best_solution = best_S.copy()
    best_val = best_score
    
    def fractional_knapsack_upper_bound(candidates, conflicts_with_selected):
        """
        Compute upper bound using fractional knapsack relaxation.
        candidates: list of vertex indices that could potentially be added
        conflicts_with_selected: set of vertices that conflict with current selection
        Returns: upper bound on additional weight
        """
        # Filter candidates that don't conflict with selected
        valid = [v for v in candidates if v not in conflicts_with_selected]
        if not valid:
            return 0.0
        
        # Sort by weight/weight ratio (which is 1 here since we're just summing weights)
        # For independent set, the fractional relaxation allows taking fractions
        # Upper bound = sum of all weights in valid set (loose but fast)
        # Better: sort by weight and take greedily (fractional knapsack with capacity = infinity)
        # Actually for this problem, a tighter bound considers that vertices may conflict with each other
        
        # Simple bound: sum of all valid weights
        # Tighter: consider that we can't take all if they conflict
        # Use a greedy coloring-based bound or just sum
        
        # For now, use sum of weights as upper bound (loose but simple)
        total = sum(w[v] for v in valid)
        return total
    
    def branch_and_bound(selected, excluded, remaining, current_weight):
        """
        Recursive branch and bound.
        selected: set of vertices in current solution
        excluded: set of vertices that conflict with selected (can't be added)
        remaining: list of vertices not yet decided
        current_weight: weight of current solution
        """
        nonlocal best_solution, best_val
        
        if time.time() > deadline - 5.0:
            return
        
        if not remaining:
            if current_weight > best_val:
                best_val = current_weight
                best_solution = selected.copy()
            return
        
        # Upper bound: current weight + sum of all remaining weights that don't conflict
        # with selected and aren't excluded
        valid_remaining = [v for v in remaining if v not in excluded]
        upper_bound = current_weight + sum(w[v] for v in valid_remaining)
        
        if upper_bound <= best_val:
            return
        
        # Choose next vertex to branch on (first in remaining list)
        v = remaining[0]
        rest = remaining[1:]
        
        # Option 1: Exclude v
        branch_and_bound(selected, excluded, rest, current_weight)
        
        # Option 2: Include v (if it doesn't conflict with selected)
        if v not in excluded:
            new_selected = selected | {v}
            # Update excluded: add v's neighbors
            new_excluded = excluded | {v}
            for u in range(1, N + 1):
                if (conflict_mask[v] >> u) & 1:
                    new_excluded.add(u)
            branch_and_bound(new_selected, new_excluded, rest, current_weight + w[v])
    
    # Use a smarter approach: iterative deepening with better bounds
    # First, get a good initial solution from greedy
    initial = best_S.copy()
    initial_weight = best_score
    
    # Try to improve using local search first
    all_vertices = list(range(1, N + 1))
    
    def can_add_to_set(S_set, a):
        for b in S_set:
            if (conflict_mask[b] >> a) & 1:
                return False
        return True
    
    S = initial.copy()
    current_score = initial_weight
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        # Try adding any vertex
        for a in all_vertices:
            if a not in S and can_add_to_set(S, a):
                S.add(a)
                current_score += w[a]
                improved = True
        # Try swapping: remove one, add two
        if not improved and len(S) > 0:
            for x in list(S):
                S.remove(x)
                current_score -= w[x]
                # Try to add back best vertices
                best_adds = []
                for a in all_vertices:
                    if a not in S and can_add_to_set(S, a):
                        best_adds.append(a)
                best_adds.sort(key=lambda a: -w[a])
                for a in best_adds[:2]:
                    if can_add_to_set(S, a):
                        S.add(a)
                        current_score += w[a]
                if current_score > initial_weight:
                    initial_weight = current_score
                    initial = S.copy()
                    improved = True
                else:
                    # Revert
                    for a in list(S):
                        if a not in best_adds[:2]:
                            pass
                    # Just keep the improved version if better
                    if current_score < initial_weight:
                        S = initial.copy()
                        current_score = initial_weight
    
    best_S = initial.copy()
    best_score = initial_weight
    
    # Now do branch and bound with time limit
    # Use a more efficient formulation
    # Process vertices in order of decreasing weight
    
    # For the B&B, use a stack-based approach to avoid recursion depth issues
    # State: (selected_set, excluded_set, remaining_list, current_weight)
    
    # Since N=511 is large, pure B&B may not finish in time
    # Instead, use a hybrid: B&B on a reduced problem
    
    # Identify "critical" vertices: those with high weight
    # Focus B&B on the top K vertices by weight
    
    K = 50  # Number of top vertices to consider in B&B
    top_vertices = vertices[:K]
    
    # Precompute conflicts among top vertices
    top_conflicts = {}
    for i, v in enumerate(top_vertices):
        top_conflicts[v] = set()
        for j, u in enumerate(top_vertices):
            if i != j and (conflict_mask[v] >> u) & 1:
                top_conflicts[v].add(u)
    
    # B&B on top vertices only
    def bnb_top(selected, excluded, remaining, current_weight):
        nonlocal best_S, best_score
        
        if time.time() > deadline - 10.0:
            return
        
        if not remaining:
            if current_weight > best_score:
                best_score = current_weight
                best_S = selected.copy()
            return
        
        # Upper bound
        valid = [v for v in remaining if v not in excluded]
        ub = current_weight + sum(w[v] for v in valid)
        if ub <= best_score:
            return
        
        v = remaining[0]
        rest = remaining[1:]
        
        # Exclude v
        bnb_top(selected, excluded, rest, current_weight)
        
        # Include v
        if v not in excluded:
            new_excluded = excluded | {v}
            for u in top_conflicts.get(v, set()):
                new_excluded.add(u)
            bnb_top(selected | {v}, new_excluded, rest, current_weight + w[v])
    
    # Initialize B&B with empty set, no exclusions, all top vertices remaining
    # But we should start from a good initial solution
    # Let's try B&B from scratch on top vertices
    
    bnb_selected = set()
    bnb_excluded = set()
    bnb_remaining = top_vertices[:]
    bnb_top(bnb_selected, bnb_excluded, bnb_remaining, 0.0)
    
    # Combine B&B result with non-top vertices using greedy
    # Add non-top vertices greedily to the B&B solution
    final_S = best_S.copy()
    for a in range(N, 0, -1):
        if a in top_vertices:
            continue
        if a not in final_S and can_add_to_set(final_S, a):
            final_S.add(a)
    
    final_score = get_score(final_S)
    if final_score > best_score:
        best_S = final_S
        best_score = final_score
    
    return list(best_S)
