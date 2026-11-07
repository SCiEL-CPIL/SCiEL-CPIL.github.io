import time
import random
from collections import defaultdict

N = 128
DEADLINE = 55.0

# Precompute forbidden sets
# A set is forbidden if it contains {a, b, c, a+b, a+c, b+c}
# We represent each forbidden configuration as a frozenset of values.
# Note: a,b,c are distinct. Sums might equal each other or a,b,c.
# The condition is: a in S, b in S, c in S, a+b in S, a+c in S, b+c in S.
# We generate all unique frozensets of values that satisfy this structure.

_forbidden_sets = set()
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        for c in range(b + 1, N + 1):
            s1 = a + b
            s2 = a + c
            s3 = b + c
            if s3 <= N:
                # The set of values involved
                vals = frozenset([a, b, c, s1, s2, s3])
                _forbidden_sets.add(vals)

# Convert to list for indexing
_constraints = list(_forbidden_sets)
_num_constraints = len(_constraints)

# Map each value to the indices of constraints it belongs to
_val_to_cons = defaultdict(list)
for i, c in enumerate(_constraints):
    for v in c:
        _val_to_cons[v].append(i)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_size = 0
    best_set = set()
    
    # Helper to check if a set is valid and update best
    def update_best(s):
        nonlocal best_size, best_set
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
    # Fast validity check for initial seeding
    def is_valid(s):
        s_set = set(s)
        for c in _constraints:
            if c.issubset(s_set):
                return False
        return True

    # Incremental violation counter
    # state: in_set (list of bools), violations (int)
    
    def get_violations(in_set):
        """Calculate total violations from scratch. Used for initialization."""
        v_count = 0
        for c in _constraints:
            all_in = True
            for v in c:
                if not in_set[v]:
                    all_in = False
                    break
            if all_in:
                v_count += 1
        return v_count

    # Simulated Annealing
    # We will run multiple restarts with different random seeds and initial densities.
    
    rng = random.Random(12345)
    
    # Strategy:
    # 1. Try to build a large valid set using a greedy + local search hybrid.
    # 2. Use SA to polish or find better configurations.
    
    # First, let's try a simple greedy to get a baseline.
    # Greedy: Add elements in random order, keep if valid.
    # This is slow if we check validity from scratch.
    # Instead, we use the incremental violation count.
    
    def greedy_random_density(density, seed):
        r = random.Random(seed)
        in_set = [False] * (N + 1)
        violations = 0
        
        # Initialize with random subset
        for v in range(1, N + 1):
            if r.random() < density:
                in_set[v] = True
        
        violations = get_violations(in_set)
        
        # Local search: try to remove elements to reduce violations, then add to increase size
        # Objective: Minimize violations, then Maximize size.
        # Let's just minimize violations first.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing a random element involved in a violation
            # Find any violated constraint
            violated_cons = []
            for i, c in enumerate(_constraints):
                all_in = True
                for v in c:
                    if not in_set[v]:
                        all_in = False
                        break
                if all_in:
                    violated_cons.append(i)
            
            if not violated_cons:
                # No violations, try to add elements
                for v in range(1, N + 1):
                    if not in_set[v]:
                        # Check if adding v creates a violation
                        # Only constraints containing v can become violated
                        new_violations = 0
                        for ci in _val_to_cons[v]:
                            c = _constraints[ci]
                            # Check if all other elements in c are in in_set
                            all_other_in = True
                            for u in c:
                                if u != v and not in_set[u]:
                                    all_other_in = False
                                    break
                            if all_other_in:
                                new_violations += 1
                        if new_violations == 0:
                            in_set[v] = True
                            improved = True
                if improved:
                    continue # Restart loop to check for more additions
                
            if violated_cons:
                # Pick a random violated constraint
                ci = r.choice(violated_cons)
                c = _constraints[ci]
                # Pick a random element from this constraint to remove
                v = r.choice(list(c))
                if in_set[v]:
                    # Remove v
                    in_set[v] = False
                    # Update violations
                    # Any constraint containing v that was violated might become unviolated
                    # Any constraint containing v that was not violated might become violated? No, removing an element cannot create a new violation (subset condition becomes harder to satisfy).
                    # So violations only decrease or stay same.
                    # We just need to recount violations for constraints containing v.
                    # Actually, we can just recalculate total violations? No, that's O(K).
                    # Let's update incrementally.
                    # For each constraint ci' containing v:
                    #   If it was violated (all in set), it is no longer violated.
                    #   If it was not violated, it remains not violated.
                    # So we just need to check which constraints containing v were violated.
                    # But we don't track which are violated individually.
                    # To be safe and fast enough (K is small), we can just recalculate total violations if we do many moves.
                    # Or, we maintain a list of violated constraint indices.
                    
                    # Let's stick to recalculating if we do many moves, or just use a simpler heuristic.
                    # Given K ~ 50k, and N=128, even O(K) per move is 50k ops.
                    # If we do 1000 moves, that's 50M ops. Python might handle 50M simple ops in ~1-2s.
                    # So we can afford O(K) per move for a limited number of moves.
                    
                    # Let's just recalculate violations from scratch if we want to be safe.
                    # But we can do better.
                    pass
                    
        # Convert to set
        s = {i for i in range(1, N + 1) if in_set[i]}
        if is_valid(s):
            update_best(s)
        return s

    # Better approach: Simulated Annealing with incremental updates.
    # State: in_set (list of bools), violations (int)
    # Move: Swap u in S, v not in S.
    # Delta violations:
    #   For each constraint containing u or v:
    #     Check status before and after.
    
    def simulate_annealing(initial_density, seed, time_limit=10.0):
        r = random.Random(seed)
        in_set = [False] * (N + 1)
        for v in range(1, N + 1):
            if r.random() < initial_density:
                in_set[v] = True
        
        violations = get_violations(in_set)
        size = sum(in_set)
        
        best_local_size = 0
        best_local_set = set()
        
        if violations == 0:
            best_local_size = size
            best_local_set = {i for i in range(1, N + 1) if in_set[i]}
            update_best(best_local_set)
            
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        
        start_sa = time.time()
        while time.time() < start_sa + time_limit and T > T_min:
            # Choose move: Swap
            # Pick random u in S, v not in S
            # If S is empty or full, skip
            in_list = [i for i in range(1, N + 1) if in_set[i]]
            out_list = [i for i in range(1, N + 1) if not in_set[i]]
            
            if not in_list or not out_list:
                break
                
            u = r.choice(in_list)
            v = r.choice(out_list)
            
            # Calculate delta violations
            # Constraints affected are those containing u or v.
            # Union of _val_to_cons[u] and _val_to_cons[v]
            affected = set(_val_to_cons[u]) | set(_val_to_cons[v])
            
            delta_viol = 0
            for ci in affected:
                c = _constraints[ci]
                # Check if c is subset of S_before
                # S_before: u in, v not in
                # S_after: u not in, v in
                
                # Count how many elements of c are in S_before
                # And S_after
                # Actually, we just need to know if the count of elements in c changes from 6 (or len(c)) to something else.
                # A constraint is violated if ALL elements are in S.
                
                # Before:
                # If u in c and v not in c:
                #   Before: u is in S. v is not in S.
                #   After: u is not in S. v is in S.
                #   The set of elements in S changes: u removed, v added.
                
                # Let's just check the condition directly for the few affected constraints.
                # Since len(c) <= 6, this is very fast.
                
                # Check if c is subset of S_before
                all_in_before = True
                for w in c:
                    if w == u:
                        # u is in S_before
                        pass
                    elif w == v:
                        # v is not in S_before
                        all_in_before = False
                        break
                    else:
                        if not in_set[w]:
                            all_in_before = False
                            break
                            
                # Check if c is subset of S_after
                all_in_after = True
                for w in c:
                    if w == u:
                        # u is not in S_after
                        all_in_after = False
                        break
                    elif w == v:
                        # v is in S_after
                        pass
                    else:
                        if not in_set[w]:
                            all_in_after = False
                            break
                
                if all_in_before and not all_in_after:
                    delta_viol -= 1
                elif not all_in_before and all_in_after:
                    delta_viol += 1
            
            # Acceptance
            if delta_viol < 0 or r.random() < (2.718281828 ** (-delta_viol / T)):
                # Accept
                in_set[u] = False
                in_set[v] = True
                violations += delta_viol
                
                if violations == 0:
                    new_size = size # Size doesn't change in a swap
                    # Wait, if we swap, size stays same.
                    # We need moves that change size too?
                    # Yes, Add/Remove moves.
                    
                if violations == 0:
                    current_set = {i for i in range(1, N + 1) if in_set[i]}
                    if len(current_set) > best_local_size:
                        best_local_size = len(current_set)
                        best_local_set = current_set
                        update_best(best_local_set)
            
            T *= alpha
            
            # Occasionally try Add/Remove to change size
            # But Swap is good for escaping local minima in validity.
            # To increase size, we need to find "safe" additions.
            
        return best_local_set

    # Run SA with different parameters
    # Since swap doesn't change size, we need a strategy to grow the set.
    # Strategy:
    # 1. Start with small set, grow by adding safe elements.
    # 2. If stuck, use SA (swaps) to find a different valid configuration of same size, then try to grow again.
    # Or, use SA with a cost function that penalizes small size?
    # Cost = violations * 100 - size.
    
    def simulate_annealing_grow(seed, time_limit=20.0):
        r = random.Random(seed)
        in_set = [False] * (N + 1)
        # Start empty
        violations = 0
        size = 0
        
        best_local_size = 0
        best_local_set = set()
        
        T = 10.0
        T_min = 0.1
        alpha = 0.99
        
        start_sa = time.time()
        while time.time() < start_sa + time_limit and T > T_min:
            # Choose move
            # 10% chance: Add random element
            # 10% chance: Remove random element
            # 80% chance: Swap
            
            move_type = r.random()
            
            if move_type < 0.1 and size < N:
                # Add
                candidates = [i for i in range(1, N + 1) if not in_set[i]]
                if candidates:
                    v = r.choice(candidates)
                    # Check delta
                    delta_viol = 0
                    for ci in _val_to_cons[v]:
                        c = _constraints[ci]
                        all_other_in = True
                        for w in c:
                            if w != v and not in_set[w]:
                                all_other_in = False
                                break
                        if all_other_in:
                            delta_viol += 1
                    
                    # Cost change: delta_viol * 100 - 1
                    delta_cost = delta_viol * 100 - 1
                    if delta_cost < 0 or r.random() < (2.718281828 ** (-delta_cost / T)):
                        in_set[v] = True
                        violations += delta_viol
                        size += 1
                        
            elif move_type < 0.2 and size > 0:
                # Remove
                candidates = [i for i in range(1, N + 1) if in_set[i]]
                if candidates:
                    v = r.choice(candidates)
                    # Check delta
                    delta_viol = 0
                    for ci in _val_to_cons[v]:
                        c = _constraints[ci]
                        # Was it violated before?
                        all_in_before = True
                        for w in c:
                            if not in_set[w]:
                                all_in_before = False
                                break
                        # Will it be violated after? (No, because v is removed)
                        if all_in_before:
                            delta_viol -= 1
                    
                    # Cost change: delta_viol * 100 + 1
                    delta_cost = delta_viol * 100 + 1
                    if delta_cost < 0 or r.random() < (2.718281828 ** (-delta_cost / T)):
                        in_set[v] = False
                        violations += delta_viol
                        size -= 1
                        
            else:
                # Swap
                in_list = [i for i in range(1, N + 1) if in_set[i]]
                out_list = [i for i in range(1, N + 1) if not in_set[i]]
                if in_list and out_list:
                    u = r.choice(in_list)
                    v = r.choice(out_list)
                    
                    affected = set(_val_to_cons[u]) | set(_val_to_cons[v])
                    delta_viol = 0
                    for ci in affected:
                        c = _constraints[ci]
                        all_in_before = True
                        for w in c:
                            if w == u:
                                pass
                            elif w == v:
                                all_in_before = False
                                break
                            else:
                                if not in_set[w]:
                                    all_in_before = False
                                    break
                            
                        all_in_after = True
                        for w in c:
                            if w == u:
                                all_in_after = False
                                break
                            elif w == v:
                                pass
                            else:
                                if not in_set[w]:
                                    all_in_after = False
                                    break
                            
                        if all_in_before and not all_in_after:
                            delta_viol -= 1
                        elif not all_in_before and all_in_after:
                            delta_viol += 1
                    
                    # Cost change: delta_viol * 100
                    delta_cost = delta_viol * 100
                    if delta_cost < 0 or r.random() < (2.718281828 ** (-delta_cost / T)):
                        in_set[u] = False
                        in_set[v] = True
                        violations += delta_viol
                        # Size unchanged
            
            if violations == 0:
                current_set = {i for i in range(1, N + 1) if in_set[i]}
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local_set = current_set
                    update_best(best_local_set)
            
            T *= alpha
            
        return best_local_set

    # Run multiple restarts
    seeds = [1, 2, 3, 4, 5, 10, 20, 30, 42, 100, 123, 456, 789, 1000, 2000, 3000]
    
    # First, try a quick greedy to get a decent start
    # Then run SA from there? Or just run SA from scratch.
    
    # Run SA for a few seconds
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        simulate_annealing_grow(seed, time_limit=5.0)
        
    # Final check
    if best_set:
        return [int(x) for x in best_set]
    else:
        # Fallback: upper half
        return list(range(65, 129))
