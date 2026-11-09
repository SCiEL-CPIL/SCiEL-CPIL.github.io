"""Combined program: Greedy/Local Search structure with Simulated Annealing refinement."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit (or 90s if specified, but 55 is safe for 60s eval)
    
    n_max = 180
    
    # 1. Precompute forbidden triples and conflict lists
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    # Let a < b < c.
    # u = b-a, v = c-a. u*v = a^2.
    # b = a+u, c = a+v.
    # We need b <= n_max and c <= n_max.
    
    # conflicts[x] = list of (y, z) such that {x, y, z} is a forbidden triple.
    # Note: The triple is unordered in the set, but the equation is specific.
    # However, the condition "1/a = 1/b + 1/c" implies a is the smallest?
    # 1/a = 1/b + 1/c => 1/a > 1/b and 1/a > 1/c => a < b and a < c.
    # So a is always the smallest element in the triple.
    # If we have a set S, a violation exists if there are a,b,c in S such that a<b<c and (b-a)(c-a)=a^2.
    
    conflicts = [[] for _ in range(n_max + 1)]
    
    for a in range(1, n_max + 1):
        a_sq = a * a
        # Iterate divisors u of a_sq
        # u < v implies u < sqrt(a_sq)
        for u in range(1, int(math.isqrt(a_sq)) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u == v:
                    continue # b == c, not distinct
                # b = a + u, c = a + v
                # Since u < v, b < c. And a < b < c.
                b = a + u
                c = a + v
                if b <= n_max and c <= n_max:
                    # Add to conflicts for a, b, c
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))
    
    # Helper to calculate change in violations if we flip element x
    # S is a list of 0/1
    def delta_violations(x, S):
        count = 0
        for y, z in conflicts[x]:
            if S[y] and S[z]:
                count += 1
        return count

    def is_valid_addition(S, x):
        # Can we add x without creating a violation?
        # A violation is created if there exists (y,z) in conflicts[x] such that y,z in S.
        # Note: conflicts[x] contains pairs (y,z) where {x,y,z} is a triple.
        # If we add x, we complete the triple if y and z are already in S.
        for y, z in conflicts[x]:
            if S[y] and S[z]:
                return False
        return True

    def greedy_build(order):
        S = [0] * (n_max + 1)
        for x in order:
            if is_valid_addition(S, x):
                S[x] = 1
        return S

    def local_search_polish(S, max_iter=200):
        """
        Deterministic local search: 
        1. Try to add any valid element.
        2. Try 1-swap (remove one, add one) to increase size or maintain size if it helps future adds? 
           Actually, standard local search for max independent set:
           - Try to add.
           - If stuck, try swap: remove x, add y if y was blocked by x.
        """
        S = S[:]
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() - start_time > deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Add
            for x in range(1, n_max + 1):
                if not S[x] and is_valid_addition(S, x):
                    S[x] = 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Swap
            # Try removing x and adding y
            # To be efficient, only consider y that are currently blocked.
            # A simple approach: try all (x, y) pairs where x in S, y not in S.
            # This is O(N^2 * deg). N=180, so 32400 * small. Acceptable.
            for x in range(1, n_max + 1):
                if not S[x]:
                    continue
                S[x] = 0
                for y in range(1, n_max + 1):
                    if S[y]:
                        continue
                    if is_valid_addition(S, y):
                        S[y] = 1
                        improved = True
                        break
                if not improved:
                    S[x] = 1
            if improved:
                continue
                
            # 3. 2-swap (Remove x1, x2, Add y1, y2)
            # This is expensive. Only do if time permits.
            if time.time() - start_time > deadline - 5.0:
                break
                
            S_list = [i for i in range(1, n_max + 1) if S[i]]
            found_swap = False
            for i in range(len(S_list)):
                if found_swap:
                    break
                for j in range(i + 1, len(S_list)):
                    x1, x2 = S_list[i], S_list[j]
                    S[x1] = 0
                    S[x2] = 0
                    # Try to add two elements
                    added_count = 0
                    temp_added = []
                    for y in range(1, n_max + 1):
                        if S[y]:
                            continue
                        if is_valid_addition(S, y):
                            S[y] = 1
                            temp_added.append(y)
                            added_count += 1
                            if added_count == 2:
                                break
                    if added_count == 2:
                        found_swap = True
                        improved = True
                        # Keep the new state
                    else:
                        # Revert
                        for y in temp_added:
                            S[y] = 0
                        S[x1] = 1
                        S[x2] = 1
            if improved:
                continue

        return S

    def simulated_annealing(initial_S, T_start=10.0, T_end=0.01, alpha=0.9999):
        S = initial_S[:]
        current_size = sum(S)
        
        # Calculate initial violations
        current_violations = 0
        # We can compute this by summing delta_violations for all x? No, that counts each triple 3 times.
        # Let's just count triples directly for initial state if needed, 
        # but for SA we only need deltas.
        # We need to track current_violations to evaluate new states.
        
        # Efficient violation count for initial state:
        # Iterate all triples? We didn't store the list of triples, only conflicts.
        # We can reconstruct violations by:
        # For each x in S, count pairs (y,z) in conflicts[x] where y,z in S and x < y < z? 
        # No, conflicts[x] has (y,z) for the triple. 
        # To count total violations, we can sum over all x in S: 
        # count of (y,z) in conflicts[x] such that y,z in S.
        # Each violation {a,b,c} with a<b<c is counted:
        # In conflicts[a]: (b,c) -> counted if b,c in S.
        # In conflicts[b]: (a,c) -> counted if a,c in S.
        # In conflicts[c]: (a,b) -> counted if a,b in S.
        # So summing over all x gives 3 * (number of violations).
        
        total_count = 0
        for x in range(1, n_max + 1):
            if S[x]:
                total_count += delta_violations(x, S)
        current_violations = total_count // 3
        
        best_S = S[:]
        best_size = current_size
        best_violations = current_violations
        
        T = T_start
        while T > T_end:
            if time.time() - start_time > deadline:
                break
            
            # Pick random x
            x = random.randint(1, n_max)
            
            old_state = S[x]
            # Calculate delta
            v_delta = delta_violations(x, S)
            
            if old_state == 0:
                # Adding x
                # New violations increase by v_delta
                new_violations = current_violations + v_delta
                new_size = current_size + 1
            else:
                # Removing x
                # New violations decrease by v_delta
                new_violations = current_violations - v_delta
                new_size = current_size - 1
            
            # Objective: Maximize size, Minimize violations.
            # Score = size - P * violations
            P = 100
            score_delta = (new_size - current_size) - P * (new_violations - current_violations)
            
            if score_delta >= 0 or (T > 0 and random.random() < math.exp(score_delta / T)):
                S[x] = 1 - old_state
                current_size = new_size
                current_violations = new_violations
                
                # Update best
                # We prefer feasible solutions (violations == 0)
                # If new is feasible and better size, update.
                # If current best is infeasible, and new is feasible, update.
                # If both feasible, max size.
                # If both infeasible, max size (though we want to avoid this).
                
                if new_violations == 0:
                    if best_violations != 0 or new_size > best_size:
                        best_size = new_size
                        best_violations = 0
                        best_S = S[:]
                else:
                    # Infeasible. Only update if it's "better" in some sense?
                    # Usually we want to keep the best feasible.
                    # If we have no feasible yet, keep the one with min violations/max size.
                    if best_violations != 0:
                         if new_violations < best_violations or (new_violations == best_violations and new_size > best_size):
                             best_violations = new_violations
                             best_size = new_size
                             best_S = S[:]
            
            T *= alpha
        
        return best_S, best_size

    # Main Loop
    best_set = set()
    best_size = 0
    
    # 1. Get a good initial guess via Greedy
    orders = [
        list(range(1, n_max + 1)),
        list(range(n_max, 0, -1)),
        sorted(range(1, n_max + 1), key=lambda x: len(conflicts[x]))
    ]
    
    initial_states = []
    for order in orders:
        S_greedy = greedy_build(order)
        S_polished = local_search_polish(S_greedy, max_iter=50)
        initial_states.append(S_polished)
        if sum(S_polished) > best_size:
            best_size = sum(S_polished)
            best_set = {i for i in range(1, n_max + 1) if S_polished[i]}
            
    # 2. Run Simulated Annealing from different starts
    # Use multiple random seeds and starting points
    random.seed(42)
    
    # Start from greedy results
    for S_init in initial_states:
        S_best, sz = simulated_annealing(S_init, T_start=5.0, T_end=0.01, alpha=0.9998)
        if sz > best_size:
            best_size = sz
            best_set = {i for i in range(1, n_max + 1) if S_best[i]}
            
    # Start from random states
    for _ in range(20):
        if time.time() - start_time > deadline - 5.0:
            break
        S_rand = [0] * (n_max + 1)
        for i in range(1, n_max + 1):
            if random.random() < 0.5:
                S_rand[i] = 1
        # Quick polish to remove obvious violations? No, SA handles it.
        # But starting with high violations might be bad.
        # Let's start with a slightly biased random set
        S_rand = [0] * (n_max + 1)
        for i in range(1, n_max + 1):
            if random.random() < 0.6:
                S_rand[i] = 1
        
        S_best, sz = simulated_annealing(S_rand, T_start=10.0, T_end=0.01, alpha=0.9995)
        if sz > best_size:
            best_size = sz
            best_set = {i for i in range(1, n_max + 1) if S_best[i]}

    # 3. Final Polish on the best set found
    if best_set:
        S_final_bool = [0] * (n_max + 1)
        for i in best_set:
            S_final_bool[i] = 1
        S_final = local_search_polish(S_final_bool, max_iter=500)
        final_set = {i for i in range(1, n_max + 1) if S_final[i]}
        if len(final_set) > best_size:
            best_set = final_set
            best_size = len(final_set)
            
    # Ensure non-empty
    if not best_set:
        best_set = {180}

    return sorted(best_set)
