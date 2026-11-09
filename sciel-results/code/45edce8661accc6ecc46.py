"""Rewritten program using Max-SAT / simulated annealing approach."""
import random
import time

def construct():
    # We want a nonempty subset S of [1, 180] with no three distinct elements a, b, c
    # such that 1/a = 1/b + 1/c. Maximize |S|.
    
    # First, enumerate all forbidden triples (a, b, c) with a, b, c distinct in [1, 180]
    # such that 1/a = 1/b + 1/c.
    # Note: 1/a = 1/b + 1/c  <=>  c = a*b/(a+b-a*c) ... wait, let's derive properly.
    # 1/a = 1/b + 1/c => 1/a = (b+c)/(bc) => bc = a(b+c) => bc - ab - ac = 0 => bc - ab - ac + a^2 = a^2 => (b-a)(c-a) = a^2
    # So for each a, we need (b-a)(c-a) = a^2 with b != c, b != a, c != a, all in [1, 180].
    # Let u = b - a, v = c - a. Then u*v = a^2, u != 0, v != 0, u != v (since b != c means u != v... wait, b != c means b-a != c-a so u != v).
    # Also b = a + u in [1, 180], c = a + v in [1, 180].
    
    # Build set of forbidden triples as frozensets of {a, b, c}
    forbidden = set()
    for a in range(1, 181):
        a2 = a * a
        # Find all divisors u of a^2 such that u != 0
        # For each divisor u, v = a^2 // u, and we need u * v == a^2
        # b = a + u, c = a + v
        # We need 1 <= b <= 180, 1 <= c <= 180, b != a, c != a, b != c
        # u can be negative or positive
        # Let's iterate over all possible u values where a^2 is divisible by u
        # Actually, let's iterate over possible b and c directly for efficiency
        for b in range(1, 181):
            if b == a:
                continue
            # 1/a = 1/b + 1/c => 1/c = 1/a - 1/b = (b-a)/(ab) => c = ab/(b-a)
            # c must be integer, positive, in [1,180], c != a, c != b
            if b - a == 0:
                continue
            num = a * b
            den = b - a
            if den == 0:
                continue
            if num % den != 0:
                continue
            c = num // den
            if c < 1 or c > 180:
                continue
            if c == a or c == b:
                continue
            # We have a valid triple (a, b, c)
            # Store as frozenset to avoid duplicates
            triple = frozenset((a, b, c))
            forbidden.add(triple)
    
    # Now we have the set of all forbidden triples.
    # We want to find a maximum-size subset S of [1, 180] that contains no forbidden triple.
    # This is equivalent to finding a maximum independent set in a 3-uniform hypergraph.
    # 
    # Approach: Simulated Annealing on the boolean vector x[1..180] where x[i]=1 if i is in S.
    # Objective: maximize sum(x[i]) - penalty * (number of forbidden triples fully contained in S)
    # We'll use a penalty that is large enough to ensure feasibility in the end, but we can also
    # just maximize |S| subject to no forbidden triples.
    # 
    # Simpler: Use simulated annealing where the state is a set S. 
    # Moves: add a random element not in S, remove a random element from S, or swap.
    # Energy: -|S| + lambda * (number of violated triples)
    # We want to minimize energy, so we want large |S| and few violated triples.
    # At the end, we ensure no violated triples.
    
    # Let's precompute for each element, the list of forbidden triples it participates in.
    n = 180
    elements = list(range(1, n+1))
    
    # Map element to list of triple indices
    elem_to_triples = {i: [] for i in range(1, n+1)}
    triple_list = list(forbidden)
    for idx, tri in enumerate(triple_list):
        for e in tri:
            elem_to_triples[e].append(idx)
    
    # Number of triples
    num_triples = len(triple_list)
    
    # Helper to calculate energy
    # We want to minimize: penalty * violations - |S|
    # Let's use a dynamic penalty or a high fixed one.
    # If we use a high penalty, the SA will try to reach 0 violations quickly.
    
    def run_sa(initial_S, time_limit=5.0):
        """
        Runs simulated annealing starting from initial_S.
        Returns the best feasible set found during the run.
        """
        if not initial_S:
            initial_S = set()
        
        current_S = set(initial_S)
        
        # Precompute violated triples for current state
        # To speed up, we can maintain a count of how many elements of each triple are in S.
        # triple_counts[idx] = number of elements of triple_list[idx] in current_S
        triple_counts = [0] * num_triples
        for e in current_S:
            for idx in elem_to_triples[e]:
                triple_counts[idx] += 1
        
        # A triple is violated if count == 3
        current_violations = sum(1 for c in triple_counts if c == 3)
        current_size = len(current_S)
        
        # Energy function
        # We want to maximize size, minimize violations.
        # Score = size - penalty * violations
        # We want to maximize Score.
        # Let's define Delta Score for moves.
        penalty = 100 # High enough to prioritize feasibility
        
        def get_score(size, viol):
            return size - penalty * viol
        
        current_score = get_score(current_size, current_violations)
        best_S = set(current_S)
        best_score = current_score
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Choose a move
            # 1. Add
            # 2. Remove
            # 3. Swap (Remove one, Add one)
            
            move_type = random.random()
            
            if move_type < 0.3 and len(current_S) < n:
                # Add
                candidates = [e for e in elements if e not in current_S]
                if not candidates:
                    continue
                e_add = random.choice(candidates)
                
                # Calculate delta
                # Adding e_add increases size by 1.
                # It may increase violations for triples containing e_add.
                # For each triple idx containing e_add:
                #   if triple_counts[idx] == 2, it becomes 3 (violation +1)
                #   if triple_counts[idx] == 3, it stays 3 (violation 0 change, but count 4? No, max 3 distinct)
                #   Actually, triple_counts counts distinct elements in S.
                #   If e_add is not in S, adding it increases count by 1.
                
                delta_viol = 0
                for idx in elem_to_triples[e_add]:
                    if triple_counts[idx] == 2:
                        delta_viol += 1
                    # If it was 3, it's already violated. Adding a 4th element to a 3-set?
                    # No, triples are sets of 3 distinct elements.
                    # So if count was 3, all 3 elements are in S. e_add is not in S, so e_add is not in the triple.
                    # Wait, elem_to_triples[e_add] lists triples containing e_add.
                    # So if e_add is in the triple, and count was 3, that means the other 2 are in S?
                    # No, count is number of elements of the triple in S.
                    # If e_add is in the triple, and count was 3, then all 3 elements of the triple are in S.
                    # But e_add is not in S (we are adding it). Contradiction?
                    # Ah, triple_counts[idx] counts elements of triple_list[idx] that are in current_S.
                    # If e_add is in triple_list[idx], and e_add is NOT in current_S, then triple_counts[idx] <= 2.
                    # So triple_counts[idx] can only be 0, 1, or 2 before adding e_add.
                    # If it is 2, adding e_add makes it 3 -> violation.
                    # If it is < 2, no violation.
                
                new_size = current_size + 1
                new_viol = current_violations + delta_viol
                new_score = get_score(new_size, new_viol)
                delta_score = new_score - current_score
                
                if delta_score >= 0 or random.random() < (2.718281828 ** (delta_score / T)):
                    # Accept
                    current_S.add(e_add)
                    for idx in elem_to_triples[e_add]:
                        triple_counts[idx] += 1
                        if triple_counts[idx] == 3:
                            pass # Already counted in delta_viol logic? 
                            # Wait, I calculated delta_viol based on old counts.
                            # I just need to update counts and violations.
                            # My delta_viol logic: if old count was 2, new is 3. Violation increases by 1.
                            # So current_violations += delta_viol is correct.
                    current_size = new_size
                    current_violations = new_viol
                    current_score = new_score
                    
                    if current_violations == 0 and current_score > best_score:
                        best_S = set(current_S)
                        best_score = current_score
                        
            elif move_type < 0.6 and current_S:
                # Remove
                e_rem = random.choice(list(current_S))
                
                # Removing e_rem decreases size by 1.
                # It may decrease violations.
                # For each triple idx containing e_rem:
                #   if triple_counts[idx] == 3, it becomes 2 (violation -1)
                #   if triple_counts[idx] < 3, no change.
                
                delta_viol = 0
                for idx in elem_to_triples[e_rem]:
                    if triple_counts[idx] == 3:
                        delta_viol -= 1
                
                new_size = current_size - 1
                new_viol = current_violations + delta_viol
                new_score = get_score(new_size, new_viol)
                delta_score = new_score - current_score
                
                if delta_score >= 0 or random.random() < (2.718281828 ** (delta_score / T)):
                    # Accept
                    current_S.discard(e_rem)
                    for idx in elem_to_triples[e_rem]:
                        triple_counts[idx] -= 1
                    current_size = new_size
                    current_violations = new_viol
                    current_score = new_score
                    
                    if current_violations == 0 and current_score > best_score:
                        best_S = set(current_S)
                        best_score = current_score
                        
            else:
                # Swap: Remove one from S, Add one not in S
                if current_S and len(current_S) < n:
                    e_rem = random.choice(list(current_S))
                    candidates = [e for e in elements if e not in current_S]
                    if candidates:
                        e_add = random.choice(candidates)
                        
                        # Calculate delta for removing e_rem and adding e_add
                        # This is tricky because the sets of triples overlap.
                        # It's safer to just do Add then Remove or vice versa, or calculate net change.
                        # Let's calculate net change in violations.
                        # This requires checking triples that contain e_rem or e_add.
                        
                        # To keep it simple and robust, we can just do a "Remove then Add" or "Add then Remove" 
                        # but that's two moves. 
                        # Alternatively, we can just rely on Add/Remove. Swaps are good for escaping local optima 
                        # where size is stuck.
                        # Let's implement a proper swap delta calculation.
                        
                        # Tuples of (triple_idx, count_change)
                        # We need to be careful if a triple contains both e_rem and e_add.
                        
                        affected_triples = set()
                        for idx in elem_to_triples[e_rem]:
                            affected_triples.add(idx)
                        for idx in elem_to_triples[e_add]:
                            affected_triples.add(idx)
                        
                        delta_viol = 0
                        for idx in affected_triples:
                            old_count = triple_counts[idx]
                            # Determine new count
                            # e_rem is in S, e_add is not in S.
                            # If e_rem is in triple idx, count decreases by 1.
                            # If e_add is in triple idx, count increases by 1.
                            
                            has_rem = e_rem in triple_list[idx]
                            has_add = e_add in triple_list[idx]
                            
                            new_count = old_count
                            if has_rem:
                                new_count -= 1
                            if has_add:
                                new_count += 1
                            
                            old_viol = 1 if old_count == 3 else 0
                            new_viol = 1 if new_count == 3 else 0
                            delta_viol += (new_viol - old_viol)
                        
                        new_size = current_size # Size unchanged
                        new_viol = current_violations + delta_viol
                        new_score = get_score(new_size, new_viol)
                        delta_score = new_score - current_score
                        
                        if delta_score >= 0 or random.random() < (2.718281828 ** (delta_score / T)):
                            # Accept
                            current_S.discard(e_rem)
                            current_S.add(e_add)
                            
                            # Update counts
                            for idx in elem_to_triples[e_rem]:
                                triple_counts[idx] -= 1
                            for idx in elem_to_triples[e_add]:
                                triple_counts[idx] += 1
                                
                            current_violations = new_viol
                            current_score = new_score
                            
                            if current_violations == 0 and current_score > best_score:
                                best_S = set(current_S)
                                best_score = current_score
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_S

    # Initial heuristic: Greedy Max Independent Set
    # Start with all elements, remove those causing most conflicts.
    S_greedy = set(elements)
    violated = set(range(num_triples))
    
    # Count violations per element
    elem_viol_count = {e: 0 for e in S_greedy}
    for idx in violated:
        for e in triple_list[idx]:
            elem_viol_count[e] += 1
            
    changed = True
    while changed:
        changed = False
        max_count = 0
        max_elem = None
        for e in S_greedy:
            if elem_viol_count[e] > max_count:
                max_count = elem_viol_count[e]
                max_elem = e
        
        if max_count == 0:
            break
            
        S_greedy.discard(max_elem)
        changed = True
        
        # Update counts
        for idx in elem_to_triples[max_elem]:
            tri = triple_list[idx]
            if all(e in S_greedy for e in tri):
                # Still violated
                pass
            else:
                # No longer violated
                # Subtract 1 from count of all elements in this triple
                for e in tri:
                    if e in S_greedy:
                        elem_viol_count[e] -= 1
                # We need to check if it was actually violated before.
                # My logic above is slightly flawed for incremental updates.
                # Let's just recompute or use a simpler greedy.
                
    # Simpler Greedy: Randomized Greedy
    # Order elements randomly, add if no conflict with already added.
    def greedy_random():
        S = set()
        random.shuffle(elements)
        # To make it faster, we can precompute conflicts
        # But with 180 elements, O(N^3) is fine? 180^3 is ~6M.
        # Let's just check triples.
        for e in elements:
            # Check if adding e creates a forbidden triple
            conflict = False
            for idx in elem_to_triples[e]:
                tri = triple_list[idx]
                if all(x in S for x in tri):
                    conflict = True
                    break
            if not conflict:
                S.add(e)
        return S

    best_overall = set()
    best_size = 0
    
    # Run multiple SA instances with different starts
    # Start 1: Greedy Random
    start_time = time.time()
    limit = 55.0
    
    while time.time() - start_time < limit:
        # Generate a few random greedy starts
        S_start = greedy_random()
        
        # Run SA
        S_result = run_sa(S_start, time_limit=5.0)
        
        if len(S_result) > best_size:
            best_size = len(S_result)
            best_overall = S_result
            
        # Also try starting from empty? No, greedy is better.
        # Try starting from the previous best?
        if best_overall:
            S_start2 = set(best_overall)
            # Perturb it
            for _ in range(5):
                if S_start2:
                    e = random.choice(list(S_start2))
                    S_start2.discard(e)
                else:
                    break
                candidates = [e for e in elements if e not in S_start2]
                if candidates:
                    S_start2.add(random.choice(candidates))
                    
            S_result2 = run_sa(S_start2, time_limit=5.0)
            if len(S_result2) > best_size:
                best_size = len(S_result2)
                best_overall = S_result2

    if not best_overall:
        best_overall = {180}
        
    return list(best_overall)
