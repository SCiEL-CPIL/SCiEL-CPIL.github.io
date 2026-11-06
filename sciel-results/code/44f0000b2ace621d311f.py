"""Rewritten program: branch-and-bound maximum independent set on 3-uniform hypergraph."""
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    DEADLINE = 88.0

    n = 128
    universe = list(range(1, n + 1))

    # Precompute LCM table
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = _lcm(i, j)

    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    # Find all bad triples: (a, b, c) with a < b < c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    # For each pair (a,b), find all c > b such that lcm(a,c)==lcm(b,c)==lcm(a,b)
    # Group by lcm value for efficiency
    from collections import defaultdict
    lcm_to_pairs = defaultdict(list)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_to_pairs[l].append((i, j))

    for l, pairs in lcm_to_pairs.items():
        # For this lcm value l, find triples where all three pairwise lcms equal l
        # A triple (a,b,c) has lcm(a,b)=lcm(a,c)=lcm(b,c)=l iff
        # a | l, b | l, c | l, and for each pair, lcm = l
        # i.e., for each pair (x,y): x|l, y|l, and lcm(x,y)=l
        # which means: the set of prime power exponents in x and y together cover l exactly
        candidates = [x for x in range(1, n + 1) if l % x == 0]
        # Filter: x must have lcm with at least one other candidate equal to l
        # Actually just check all triples from candidates
        if len(candidates) >= 3:
            for idx_a in range(len(candidates)):
                a = candidates[idx_a]
                for idx_b in range(idx_a + 1, len(candidates)):
                    b = candidates[idx_b]
                    if _lcm(a, b) != l:
                        continue
                    for idx_c in range(idx_b + 1, len(candidates)):
                        c = candidates[idx_c]
                        if _lcm(a, c) == l and _lcm(b, c) == l:
                            bad_triples.append((a, b, c))

    # Convert to bitsets for fast operations
    # Each element x (1..128) gets a bit index x-1
    # For each element x, conflict_mask[x] is a bitset of elements that cannot coexist with x in a valid set
    # But this is a 3-uniform constraint, not pairwise. We need: for each pair (a,b) that appears in a bad triple,
    # the third element c is forbidden if both a and b are selected.
    
    # Better approach: for branch and bound, we need to check if adding x creates a conflict.
    # A conflict exists if there's a bad triple (a,b,x) where a,b are both already selected.
    # Precompute: for each x, list of pairs (a,b) such that (a,b,x) is a bad triple (with a<b<x or any order)
    
    # Let's restructure: for each x, forbidden_pairs[x] = set of (a,b) with a<b, a!=x, b!=x, 
    # such that {a,b,x} forms a bad triple.
    forbidden_pairs = [set() for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        # The triple is bad if any permutation works, but we defined a<b<c
        # When checking if x can be added: if x=c, then (a,b) is a forbidden pair
        # if x=a, then (b,c) is a forbidden pair
        # if x=b, then (a,c) is a forbidden pair
        forbidden_pairs[c].add((a, b))
        forbidden_pairs[a].add((b, c))
        forbidden_pairs[b].add((a, c))

    # For fast checking, convert forbidden_pairs to bitsets
    # conflict_bitset[x]: bitset where bit i is set if element (i+1) is in some forbidden pair with x
    # Actually we need: given selected set S, can we add x? 
    # We can add x iff for all (a,b) in forbidden_pairs[x], not (a in S and b in S)
    # This is equivalent to: the number of forbidden pairs of x that are fully in S is 0
    
    # Precompute for each x a list of forbidden pairs as (a,b) tuples
    # And also a bitset representation for faster counting
    
    # Let's use a different approach for B&B:
    # Represent the remaining candidate set as a bitset (128 bits = 2 uint64s or just a Python int)
    # When we include x, we need to remove all elements that would create a conflict.
    # But conflicts are pairwise conditional, so this is trickier.
    
    # Alternative: use the standard MIS formulation on a derived graph.
    # Actually, let's just do B&B with careful pruning.
    
    # State: selected set (as a frozenset or list), available candidates (as a set)
    # When we decide to include x, we remove from available:
    #   - x itself
    #   - any y such that there exists a bad triple containing x and y and some z already selected
    # This is getting complex. Let me use a cleaner formulation.
    
    # Cleaner: Think of it as selecting a set S with no bad triple entirely in S.
    # B&B: process elements in some order. At each step, either include or exclude.
    # Pruning: upper bound = current_size + number of remaining candidates that can still be added.
    
    # For efficient conflict checking during B&B:
    # Maintain a set `selected`. When considering adding x:
    #   Check all pairs (a,b) in forbidden_pairs[x]: if a in selected and b in selected, reject.
    #   This is O(|forbidden_pairs[x]|) per check.
    
    # Let's precompute sorted orders and use a good B&B strategy.
    
    # Count constraint degree
    constraint_degree = [0] * (n + 1)
    for x in range(1, n + 1):
        constraint_degree[x] = len(forbidden_pairs[x])
    
    # Order elements by ascending constraint degree (less constrained first) for greedy,
    # but for B&B we might want most constrained first for better pruning.
    
    # Let's implement B&B with bitset acceleration for the "can_add" check.
    # For each x, precompute a list of forbidden pairs.
    # Use a set for selected elements.
    
    best_solution = []
    best_size = 0
    
    def can_add(x, selected_set):
        for (a, b) in forbidden_pairs[x]:
            if a in selected_set and b in selected_set:
                return False
        return True
    
    # Greedy baseline
    def greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            if can_add(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # Get initial solution from greedy
    order_greedy = sorted(universe, key=lambda x: (constraint_degree[x], x))
    init_sol = greedy(order_greedy)
    best_size = len(init_sol)
    best_solution = list(init_sol)
    
    # B&B with memoization-like pruning
    # Use recursive B&B with time checking
    node_count = 0
    
    def upper_bound(selected_count, available_set):
        """Simple upper bound: selected_count + len(available_set)"""
        return selected_count + len(available_set)
    
    # More sophisticated UB: greedy coloring bound on the "conflict graph"
    # For our 3-uniform problem, we can't directly apply graph coloring.
    # Instead, use: UB = selected_count + max number we can still add
    # A safe but loose UB is selected_count + len(available_set).
    # Tighter: iteratively remove elements that conflict with at least one pair in available
    
    def greedy_fill(selected_set, available_list):
        """Greedily fill as many as possible from available_list."""
        result = []
        s = set(selected_set)
        for x in available_list:
            if x not in s and can_add(x, s):
                result.append(x)
                s.add(x)
        return result
    
    # B&B function
    def branch_and_bound(order, selected_list, idx, selected_set, available_bitset):
        nonlocal best_size, best_solution, node_count
        
        if time.time() - start_time > DEADLINE:
            return
        
        node_count += 1
        
        # Upper bound
        remaining_count = 0
        for i in range(idx, len(order)):
            if available_bitset & (1 << order[i]):
                remaining_count += 1
        
        if len(selected_list) + remaining_count <= best_size:
            return
        
        if idx >= len(order):
            if len(selected_list) > best_size:
                best_size = len(selected_list)
                best_solution = list(selected_list)
            return
        
        x = order[idx]
        
        # Option 1: exclude x
        branch_and_bound(order, selected_list, idx + 1, selected_set, available_bitset)
        
        if time.time() - start_time > DEADLINE:
            return
        
        # Option 2: include x (if possible)
        if available_bitset & (1 << x):
            if can_add(x, selected_set):
                new_set = selected_set | {x}
                new_available = available_bitset & ~(1 << x)
                # Remove elements that would create conflicts? 
                # Actually, we don't need to remove them from available; 
                # the can_add check will handle it when we try to add them.
                # But for tighter pruning, we could remove elements y where 
                # there exists z in selected_set such that (y,z) is in forbidden_pairs[some future element]
                # This is complex; skip for now.
                branch_and_bound(order, selected_list + [x], idx + 1, new_set, new_available)
    
    # The naive B&B above is too slow for 128 elements. 
    # Let me use a different strategy: iterative deepening or a smarter heuristic.
    
    # Actually, let's try a different structural approach:
    # Use local search with a better neighborhood (2-swap, 3-swap) and simulated annealing.
    
    def local_search_v2(initial_set, max_iters=50000):
        """Local search with 1-add, 1-remove, 2-swap moves."""
        current = set(initial_set)
        best = set(initial_set)
        best_size_local = len(current)
        T = 2.0  # temperature
        cooling = 0.9999
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            move_type = random.randint(0, 2)
            new_set = set(current)
            
            if move_type == 0:  # 1-add
                candidates = [x for x in universe if x not in current]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add(x, current):
                    new_set.add(x)
                else:
                    continue
            elif move_type == 1:  # 1-remove (only if it helps, i.e., never in maximization unless stuck)
                if not current:
                    continue
                x = random.choice(list(current))
                new_set.remove(x)
            else:  # 2-swap: remove one, add one
                if not current:
                    continue
                to_remove = random.choice(list(current))
                candidates = [x for x in universe if x not in current]
                if not candidates:
                    continue
                to_add = random.choice(candidates)
                new_set.discard(to_remove)
                # Check if to_add can be added
                temp_set = set(new_set)
                if can_add(to_add, temp_set):
                    new_set.add(to_add)
                else:
                    new_set.add(to_remove)  # revert
                    continue
            
            delta = len(new_set) - len(current)
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                current = new_set
                if len(current) > best_size_local:
                    best_size_local = len(current)
                    best = set(current)
            
            T *= cooling
        
        return list(best)
    
    import random
    
    # Run multiple greedy + local search combinations
    all_solutions = []
    
    # Greedy with different orderings
    strategies = [
        lambda: sorted(universe, key=lambda x: (constraint_degree[x], x)),
        lambda: sorted(universe, key=lambda x: (-constraint_degree[x], x)),
        lambda: list(reversed(universe)),
        lambda: sorted(universe, key=lambda x: (x % 2, x)),  # evens first
        lambda: sorted(universe, key=lambda x: (-x.bit_count(), x)),  # fewer bits first
    ]
    
    for i, strat in enumerate(strategies):
        if time.time() - start_time > DEADLINE:
            break
        order = strat()
        sol = greedy(order)
        all_solutions.append(sol)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = list(sol)
    
    # Randomized greedy
    for i in range(30):
        if time.time() - start_time > DEADLINE - 5:
            break
        random.seed(i * 31337 + 1)
        order = list(universe)
        random.shuffle(order)
        sol = greedy(order)
        all_solutions.append(sol)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = list(sol)
    
    # Local search on top solutions
    all_solutions.sort(key=lambda s: -len(s))
    
    for sol in all_solutions[:15]:
        if time.time() - start_time > DEADLINE - 3:
            break
        improved = local_search_v2(sol, max_iters=20000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = list(improved)
    
    # Final polish
    if time.time() - start_time < DEADLINE - 1:
        improved = local_search_v2(best_solution, max_iters=30000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = list(improved)
    
    if not best_solution:
        best_solution = [1]
    
    return best_solution
