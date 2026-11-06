"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # Filter out numbers where 2*a is a square (diagonal constraint)
    # If 2*a is a square, a cannot be in the set.
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    # Map valid numbers to indices
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict graph: which valid numbers conflict with which
    # a and b conflict if a + b is a perfect square (a != b)
    # Note: We only store neighbors j > i to save memory/time, or just full sets.
    # Given N~250, full sets are fine.
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                if j != i:
                    conflict[i].add(j)
    
    # Precompute degrees for initial heuristics
    degrees = [len(conflict[i]) for i in range(num_valid)]

    # Function to calculate the number of conflicts in a set
    # We represent the set as a bitmask or a boolean list?
    # With N~250, a boolean list or set of active indices is faster for local search
    # than bit operations in Python if we iterate neighbors.
    
    # Let's use a list of booleans or a set of active vertices.
    # A set of active vertices allows O(1) lookup for membership.
    # To count conflicts efficiently:
    # conflicts(S) = sum_{v in S} (number of neighbors of v in S) / 2
    
    def count_conflicts(active_set):
        # active_set is a set of indices
        c = 0
        for v in active_set:
            for u in conflict[v]:
                if u in active_set:
                    c += 1
        return c // 2

    def is_valid(active_set):
        return count_conflicts(active_set) == 0

    # Heuristic to generate an initial valid set
    # Greedy: sort by degree (ascending), add if no conflict with already added
    def greedy_initial():
        order = sorted(range(num_valid), key=lambda x: degrees[x])
        active = set()
        for v in order:
            # Check if v conflicts with any in active
            ok = True
            for u in conflict[v]:
                if u in active:
                    ok = False
                    break
            if ok:
                active.add(v)
        return active

    # Weighted Min-Conflicts Repair
    # Given a set (possibly invalid), remove vertices to minimize conflicts
    # We prioritize removing vertices with high "cost" (many conflicts)
    def repair_min_conflicts(initial_set, max_iters=1000):
        active = set(initial_set)
        # Calculate initial conflicts per vertex
        # conf_count[v] = number of neighbors of v in active
        conf_count = {v: 0 for v in active}
        for v in active:
            for u in conflict[v]:
                if u in active:
                    conf_count[v] += 1
        
        total_conf = sum(conf_count.values()) // 2
        
        if total_conf == 0:
            return active
        
        for _ in range(max_iters):
            if total_conf == 0:
                break
            
            # Find vertex with max conflicts
            max_v = -1
            max_c = -1
            for v in active:
                if conf_count[v] > max_c:
                    max_c = conf_count[v]
                    max_v = v
            
            if max_v == -1:
                break
                
            # Remove max_v
            active.remove(max_v)
            total_conf -= max_c // 2 # Approximate, will recalc exactly below? 
            # Actually, let's just recalculate affected counts
            # When removing v:
            # For each neighbor u of v:
            #   conf_count[u] decreases by 1
            #   total_conf decreases by 1 (edge v-u removed)
            
            # Wait, my total_conf update above was wrong. 
            # Let's do it properly.
            # Remove v from active.
            # Decrement conf_count for all neighbors of v that are in active.
            for u in conflict[max_v]:
                if u in active:
                    conf_count[u] -= 1
            # The number of conflicts removed is the number of neighbors of max_v in active.
            # That is exactly conf_count[max_v] BEFORE removal.
            # So total_conf -= conf_count[max_v]
            # But we already updated conf_count? No, we update neighbors.
            # Let's re-structure.
            
            # Reset logic for clarity and correctness in the loop below
            pass 

        # Let's implement a simpler, robust repair:
        # While conflicts exist, find a conflicting pair, remove the one with higher degree in current subgraph.
        def repair_simple(active_set):
            active = set(active_set)
            changed = True
            while changed:
                changed = False
                # Find a conflict
                for v in active:
                    for u in conflict[v]:
                        if u in active:
                            # Conflict found between v and u
                            # Remove one with higher degree in active
                            deg_v = sum(1 for x in conflict[v] if x in active)
                            deg_u = sum(1 for x in conflict[u] if x in active)
                            
                            if deg_v >= deg_u:
                                active.remove(v)
                            else:
                                active.remove(u)
                            changed = True
                            break
                    if changed:
                        break
            return active

        return repair_simple(initial_set)

    # Simulated Annealing / Hill Climbing on Valid Sets
    # Move: Remove one vertex, or Remove one and Add one (if valid)
    # To ensure validity, we only consider moves that result in a valid set.
    # A valid set is an independent set.
    # Adding a vertex v to a valid set S is valid iff no neighbor of v is in S.
    # Removing a vertex v from a valid set S is always valid.
    
    def neighbor_count(v, active):
        return sum(1 for u in conflict[v] if u in active)

    def try_add(v, active):
        # Returns True if v can be added to active
        for u in conflict[v]:
            if u in active:
                return False
        return True

    def hill_climb(initial_set, time_limit=5.0):
        active = set(initial_set)
        best_size = len(active)
        best_set = set(active)
        
        # Temperature for Simulated Annealing (if we allowed invalid sets)
        # But let's stick to valid set moves for simplicity and robustness.
        # Move types:
        # 1. Remove random vertex (size -1)
        # 2. Remove vertex v, Add vertex w (size 0) - "Swap"
        # 3. Add vertex w (size +1) - "Improve"
        
        # We want to maximize size.
        # So we prioritize "Add" moves. If none available, try "Swap" to escape local optimum.
        
        t_start = time.time()
        temp = 1.0
        cooling = 0.999
        
        while time.time() - t_start < time_limit:
            # Try to add a vertex
            # Candidate vertices: those not in active and no neighbors in active
            # We can precompute candidates? Or just iterate all?
            # N is ~250, so iterating all is fast.
            
            added = False
            # Shuffle to avoid bias
            candidates = list(range(num_valid))
            random.shuffle(candidates)
            
            for v in candidates:
                if v not in active and try_add(v, active):
                    active.add(v)
                    added = True
                    break
                    
            if added:
                if len(active) > best_size:
                    best_size = len(active)
                    best_set = set(active)
                continue # Try to add more
            
            # If no add possible, try a swap
            # Remove v, try to add w
            # Heuristic: Remove v with low "potential" (few non-neighbors?) 
            # Or just random.
            
            # To make swaps effective, we should remove a vertex that "blocks" many others.
            # But let's try random removal first.
            
            if active:
                v_remove = random.choice(list(active))
                active.remove(v_remove)
                
                # Now try to add something
                # Maybe add the vertex we just removed? No, that's undo.
                # Try to add a different vertex.
                
                candidates = list(range(num_valid))
                random.shuffle(candidates)
                
                swap_success = False
                for w in candidates:
                    if w == v_remove:
                        continue
                    if w not in active and try_add(w, active):
                        active.add(w)
                        swap_success = True
                        break
                        
                if not swap_success:
                    # Put it back? Or keep it removed?
                    # If we can't find a replacement, keeping it removed reduces size.
                    # But maybe it opens up space for a future add?
                    # Let's put it back to maintain size, unless we want to explore smaller sets.
                    # Actually, if we can't add anything, we are in a local max.
                    # Let's try a random removal without immediate replacement to "jump"
                    # But we must maintain validity.
                    
                    # Let's just try to add the removed one back if we didn't replace it,
                    # OR keep it removed if we think it helps.
                    # Better: If no swap found, revert to previous state (undo the removal)
                    # and try a different removal.
                    active.add(v_remove)
                    
                # Acceptance logic for Swaps (size unchanged)
                # In pure hill climbing, we only accept if size increases.
                # Swaps don't increase size. So we need Simulated Annealing to accept swaps.
                # Let's implement a simple SA loop instead of pure HC.

        return best_set

    # Let's implement a proper Simulated Annealing that allows temporary size decrease
    # to escape local optima, but heavily biases towards size increase.
    
    def simulated_annealing(initial_set, time_limit=20.0):
        active = set(initial_set)
        best_set = set(active)
        best_size = len(active)
        
        t_start = time.time()
        temp = 10.0
        cooling = 0.99
        
        # Precompute potential add candidates? 
        # Dynamic calculation is fine for N=250.
        
        while time.time() - t_start < time_limit:
            # Generate a neighbor state
            # 1. Random Add (if valid)
            # 2. Random Remove
            # 3. Random Swap (Remove one, Add one)
            
            move_type = random.random()
            
            new_active = set(active)
            
            if move_type < 0.5:
                # Try Add
                # Pick random candidate
                # To speed up, we can filter candidates first?
                # Let's just pick random and check.
                v = random.randint(0, num_valid - 1)
                if v not in active and try_add(v, active):
                    new_active.add(v)
                else:
                    # If add fails, try Remove
                    if active:
                        v = random.choice(list(active))
                        new_active.remove(v)
                    else:
                        continue
                        
            elif move_type < 0.8:
                # Try Remove
                if active:
                    v = random.choice(list(active))
                    new_active.remove(v)
                else:
                    continue
                    
            else:
                # Try Swap
                if not active:
                    continue
                v_remove = random.choice(list(active))
                new_active.remove(v_remove)
                
                # Try to add a random vertex
                v_add = random.randint(0, num_valid - 1)
                if v_add != v_remove and v_add not in active and try_add(v_add, active):
                    new_active.add(v_add)
                else:
                    # If swap fails, revert to remove only? 
                    # Or just treat as remove?
                    # Let's treat as remove if add fails, to keep it simple.
                    # Actually, if add fails, the state is just "removed v_remove".
                    # That is a valid state.
                    pass

            size_old = len(active)
            size_new = len(new_active)
            delta = size_new - size_old
            
            if delta > 0:
                # Always accept improvements
                active = new_active
                if size_new > best_size:
                    best_size = size_new
                    best_set = set(active)
            else:
                # Accept with probability exp(delta / temp)
                # delta is negative or 0
                if delta == 0:
                    # Accept swaps with high probability to explore
                    if random.random() < 0.5:
                        active = new_active
                else:
                    # delta < 0
                    prob = math.exp(delta / temp)
                    if random.random() < prob:
                        active = new_active
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01

        return best_set

    # Main Loop
    # 1. Get a good initial solution via Greedy
    # 2. Run SA for a while
    # 3. Restart with different random seeds/initializations if time permits
    
    best_global = set()
    best_global_size = 0
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Generate initial solution
        # Sometimes random, sometimes greedy
        if restarts % 3 == 0:
            # Random start
            init_set = set(random.sample(range(num_valid), min(num_valid, random.randint(50, 150))))
            init_set = repair_simple(init_set)
        else:
            # Greedy start
            init_set = greedy_initial()
        
        # Run SA
        # Time budget per SA run: 2 seconds
        sa_result = simulated_annealing(init_set, time_limit=2.0)
        
        if len(sa_result) > best_global_size:
            best_global_size = len(sa_result)
            best_global = set(sa_result)
        
        # Check time
        if time.time() > deadline - 5:
            break

    # Final check: Try to add any remaining vertices to best_global
    # (Greedy improvement on the best found)
    final_set = set(best_global)
    improved = True
    while improved:
        improved = False
        for v in range(num_valid):
            if v not in final_set and try_add(v, final_set):
                final_set.add(v)
                improved = True

    if len(final_set) > best_global_size:
        best_global = final_set
        best_global_size = len(final_set)

    # Convert to list of numbers
    result = [valid_numbers[v] for v in best_global]
    
    if not result:
        # Fallback
        result = [1]
        
    return sorted(result)

# Helper function defined inside construct scope for clarity, 
# but repair_simple was used before definition in simulated_annealing? 
# No, repair_simple is defined inside repair_min_conflicts? 
# Wait, in the code above, repair_simple is defined INSIDE repair_min_conflicts.
# But simulated_annealing calls repair_simple? No, it doesn't.
# greedy_initial calls nothing external.
# The initial call in main loop: init_set = repair_simple(init_set)
# This will fail if repair_simple is not in scope.
# Let's refactor: Define repair_simple at the top level of construct.

def construct():
    start_time = time.time()
    deadline = 58.0
    
    n = 256
    max_sum = 2 * n
    
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                if j != i:
                    conflict[i].add(j)
    
    degrees = [len(conflict[i]) for i in range(num_valid)]

    def repair_simple(active_set):
        active = set(active_set)
        changed = True
        while changed:
            changed = False
            for v in active:
                for u in conflict[v]:
                    if u in active:
                        deg_v = sum(1 for x in conflict[v] if x in active)
                        deg_u = sum(1 for x in conflict[u] if x in active)
                        if deg_v >= deg_u:
                            active.remove(v)
                        else:
                            active.remove(u)
                        changed = True
                        break
                if changed:
                    break
        return active

    def greedy_initial():
        order = sorted(range(num_valid), key=lambda x: degrees[x])
        active = set()
        for v in order:
            ok = True
            for u in conflict[v]:
                if u in active:
                    ok = False
                    break
            if ok:
                active.add(v)
        return active

    def try_add(v, active):
        for u in conflict[v]:
            if u in active:
                return False
        return True

    def simulated_annealing(initial_set, time_limit=2.0):
        active = set(initial_set)
        best_set = set(active)
        best_size = len(active)
        
        t_start = time.time()
        temp = 10.0
        cooling = 0.99
        
        while time.time() - t_start < time_limit:
            move_type = random.random()
            new_active = set(active)
            
            if move_type < 0.5:
                v = random.randint(0, num_valid - 1)
                if v not in active and try_add(v, active):
                    new_active.add(v)
                else:
                    if active:
                        v = random.choice(list(active))
                        new_active.remove(v)
                    else:
                        continue
            elif move_type < 0.8:
                if active:
                    v = random.choice(list(active))
                    new_active.remove(v)
                else:
                    continue
            else:
                if not active:
                    continue
                v_remove = random.choice(list(active))
                new_active.remove(v_remove)
                v_add = random.randint(0, num_valid - 1)
                if v_add != v_remove and v_add not in active and try_add(v_add, active):
                    new_active.add(v_add)

            size_old = len(active)
            size_new = len(new_active)
            delta = size_new - size_old
            
            if delta > 0:
                active = new_active
                if size_new > best_size:
                    best_size = size_new
                    best_set = set(active)
            else:
                if delta == 0:
                    if random.random() < 0.5:
                        active = new_active
                else:
                    prob = math.exp(delta / temp)
                    if random.random() < prob:
                        active = new_active
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01

        return best_set

    best_global = set()
    best_global_size = 0
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        if restarts % 3 == 0:
            init_set = set(random.sample(range(num_valid), min(num_valid, random.randint(50, 150))))
            init_set = repair_simple(init_set)
        else:
            init_set = greedy_initial()
        
        sa_result = simulated_annealing(init_set, time_limit=2.0)
        
        if len(sa_result) > best_global_size:
            best_global_size = len(sa_result)
            best_global = set(sa_result)
        
        if time.time() > deadline - 5:
            break

    final_set = set(best_global)
    improved = True
    while improved:
        improved = False
        for v in range(num_valid):
            if v not in final_set and try_add(v, final_set):
                final_set.add(v)
                improved = True

    if len(final_set) > best_global_size:
        best_global = final_set
        best_global_size = len(final_set)

    result = [valid_numbers[v] for v in best_global]
    
    if not result:
        result = [1]
        
    return sorted(result)
# EVOLVE-BLOCK-END
