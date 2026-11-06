import time
import math
import random
import sys

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    Equation: 1/a = 1/b + 1/c => c = ab / (b-a).
    We require b > a and c to be an integer in [1, N] distinct from a and b.
    Note: Since 1/a = 1/b + 1/c, a must be smaller than both b and c.
    """
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    key = frozenset((a, b, c))
                    if key not in seen:
                        seen.add(key)
                        triples.append((a, b, c))
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    N = 180
    
    triples = _precompute_triples(N)
    T = len(triples)
    
    # Build index: element -> list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
        
    # Initialize a valid set using a fast greedy method
    # We use the previous greedy removal logic to get a good starting point
    def greedy_init():
        S = set(range(1, N + 1))
        counts = [3] * T
        active_count = [0] * (N + 1)
        for t_idx in range(T):
            a, b, c = triples[t_idx]
            active_count[a] += 1
            active_count[b] += 1
            active_count[c] += 1
            
        while sum(active_count) > 0:
            max_ac = 0
            candidates = []
            for x in range(1, N + 1):
                if x in S:
                    ac = active_count[x]
                    if ac > max_ac:
                        max_ac = ac
                        candidates = [x]
                    elif ac == max_ac and ac > 0:
                        candidates.append(x)
            if not candidates:
                break
            x_to_remove = random.choice(candidates)
            S.remove(x_to_remove)
            for t_idx in elem_triples[x_to_remove]:
                if counts[t_idx] == 3:
                    a, b, c = triples[t_idx]
                    active_count[a] -= 1
                    active_count[b] -= 1
                    active_count[c] -= 1
                    counts[t_idx] = 2
                elif counts[t_idx] == 2:
                    counts[t_idx] = 1
        return S, counts

    S_init, counts_init = greedy_init()
    
    # Simulated Annealing
    # State: S (set), counts (list of ints)
    # We track best valid solution
    best_set = set(S_init)
    best_size = len(S_init)
    
    # Parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.99
    lambda_penalty = 100.0  # Penalty for active triples
    
    # We will run multiple SA runs with different seeds/starting points if time permits
    # But first, let's try to improve the initial one.
    
    S = set(S_init)
    counts = list(counts_init)
    active_triples_count = 0
    for c in counts:
        if c == 3:
            active_triples_count += 1
            
    current_cost = -len(S) + lambda_penalty * active_triples_count
    
    T = T_start
    steps_per_temp = 1000
    
    while time.time() < deadline - 5.0:
        if T < T_end:
            break
            
        for _ in range(steps_per_temp):
            if time.time() > deadline - 5.0:
                break
                
            # Choose a move
            # 1. Add random element not in S
            # 2. Remove random element in S
            # 3. Swap
            
            move_type = random.random()
            if move_type < 0.3:
                # Add
                not_in_S = [x for x in range(1, N + 1) if x not in S]
                if not not_in_S:
                    continue
                x = random.choice(not_in_S)
                
                # Calculate delta
                delta_size = 1
                delta_active = 0
                for t_idx in elem_triples[x]:
                    if counts[t_idx] == 2:
                        delta_active += 1
                    # We don't update counts yet, just calculate
                # Wait, if we add x, counts[t] increases.
                # If counts[t] was 2, it becomes 3 (active).
                # If counts[t] was 1, it becomes 2.
                # If counts[t] was 0, it becomes 1.
                
                delta_cost = -delta_size + lambda_penalty * delta_active
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    # Accept
                    S.add(x)
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
                        if counts[t_idx] == 3:
                            active_triples_count += 1
                    current_cost += delta_cost
                    if active_triples_count == 0:
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
                            
            elif move_type < 0.6:
                # Remove
                if not S:
                    continue
                x = random.choice(list(S))
                
                delta_size = -1
                delta_active = 0
                for t_idx in elem_triples[x]:
                    if counts[t_idx] == 3:
                        delta_active -= 1
                # If counts[t] was 3, it becomes 2 (inactive).
                # If counts[t] was 2, it becomes 1.
                
                delta_cost = -delta_size + lambda_penalty * delta_active
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    # Accept
                    S.remove(x)
                    for t_idx in elem_triples[x]:
                        counts[t_idx] -= 1
                        if counts[t_idx] == 2:
                            active_triples_count -= 1
                    current_cost += delta_cost
                    if active_triples_count == 0:
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
                            
            else:
                # Swap
                if not S:
                    continue
                not_in_S = [x for x in range(1, N + 1) if x not in S]
                if not not_in_S:
                    continue
                x_out = random.choice(list(S))
                x_in = random.choice(not_in_S)
                
                if x_out == x_in:
                    continue
                    
                # Remove x_out
                delta_active_out = 0
                for t_idx in elem_triples[x_out]:
                    if counts[t_idx] == 3:
                        delta_active_out -= 1
                        
                # Add x_in
                # Note: counts are not updated yet.
                # We need to simulate the state after removal to calculate addition effect?
                # Or calculate net change directly.
                # If x_out and x_in are in the same triple t:
                #   If count[t] == 3: remove x_out -> 2 (inactive, -1), add x_in -> 3 (active, +1). Net 0.
                #   If count[t] == 2: remove x_out -> 1, add x_in -> 2. Net 0.
                #   If count[t] == 1: remove x_out -> 0, add x_in -> 1. Net 0.
                # So for shared triples, net change is 0.
                # For triples with only x_out:
                #   If count == 3: -1.
                # For triples with only x_in:
                #   If count == 2: +1.
                
                # Let's just apply removal, then addition, and track deltas.
                # It's safer.
                
                # Step 1: Remove x_out
                # We'll tentatively update counts and active count, then revert if rejected?
                # Or just calculate delta.
                
                delta_active = 0
                # Effect of removing x_out
                for t_idx in elem_triples[x_out]:
                    if counts[t_idx] == 3:
                        delta_active -= 1
                    # Note: if t also contains x_in, the count drops by 1, but then rises by 1.
                    # The "active" status change:
                    # If count was 3, it becomes 2 (inactive). Then if x_in is added, it becomes 3 (active).
                    # So net change for shared triple is 0.
                    # My delta calculation above:
                    # Remove x_out from triple with count 3: delta -1.
                    # Add x_in to triple with count 2 (after removal): delta +1.
                    # Total 0.
                    # But my "Add x_in" calculation below uses the ORIGINAL counts.
                    # So I need to be careful.
                    
                # Let's do it explicitly:
                # 1. Identify all triples affected.
                # 2. Calculate new count for each.
                # 3. Compare active status.
                
                affected = set(elem_triples[x_out]) | set(elem_triples[x_in])
                delta_active = 0
                for t_idx in affected:
                    old_count = counts[t_idx]
                    new_count = old_count
                    if x_out in triples[t_idx]:
                        new_count -= 1
                    if x_in in triples[t_idx]:
                        new_count += 1
                    # Check if active status changed
                    was_active = (old_count == 3)
                    is_active = (new_count == 3)
                    if was_active and not is_active:
                        delta_active -= 1
                    elif not was_active and is_active:
                        delta_active += 1
                        
                delta_size = 0
                delta_cost = lambda_penalty * delta_active
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    # Accept
                    # Update S
                    S.remove(x_out)
                    S.add(x_in)
                    # Update counts and active_triples_count
                    for t_idx in affected:
                        old_count = counts[t_idx]
                        if x_out in triples[t_idx]:
                            old_count -= 1
                        if x_in in triples[t_idx]:
                            old_count += 1
                        counts[t_idx] = old_count
                        # Update active count based on change
                        # We already calculated delta_active, but we need to update the global counter
                        # Let's just recalculate active_triples_count for affected triples?
                        # Or apply the delta.
                        pass 
                    # It's easier to just update the global active_triples_count by delta_active
                    active_triples_count += delta_active
                    current_cost += delta_cost
                    if active_triples_count == 0:
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
                            
        T *= alpha
        
    # Final validity check
    if not best_set:
        best_set = set(range(90, 181))
        
    valid = True
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            valid = False
            break
            
    if not valid:
        # Fallback repair
        for (a, b, c) in triples:
            if a in best_set and b in best_set and c in best_set:
                best_set.remove(c)
                
    return list(best_set)

# EVOLVE-BLOCK-END
