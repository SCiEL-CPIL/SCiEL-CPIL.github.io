import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    n = 128
    numbers = list(range(1, n + 1))

    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    # 3. Build conflict structure for fast validity checks
    # For each element x, store the set of pairs (u,v) such that {x,u,v} is a bad triple
    conflict = {i: set() for i in range(1, n + 1)}
    for a, b, c in bad_triples:
        conflict[a].add((b, c))
        conflict[b].add((a, c))
        conflict[c].add((a, b))

    # 4. Simulated Annealing Setup
    # State: A set S of selected elements.
    # Objective: Maximize |S| while minimizing the number of violated constraints (bad triples fully contained in S).
    # Energy function: E(S) = -|S| + P * (# violated triples in S)
    # We want to minimize E.
    
    # To speed up energy calculation during SA moves, we maintain a count of violations.
    # When adding x: new violations = number of pairs (u,v) in conflict[x] such that u,v in S.
    # When removing x: removed violations = number of pairs (u,v) in conflict[x] such that u,v in S.
    
    # Initial state: Greedy construction
    # Start with empty set, add elements in a random order if valid.
    # Or start with all elements and remove those causing most conflicts?
    # Let's try starting with a random subset of size ~60 and refining.
    
    best_set = set()
    best_energy = float('inf')
    
    def get_violations_count(S):
        """Count bad triples fully contained in S."""
        count = 0
        for a, b, c in bad_triples:
            if a in S and b in S and c in S:
                count += 1
        return count

    def energy(S):
        return -len(S) + 10.0 * get_violations_count(S)

    # Helper to check if adding x to S creates any new violation
    # A new violation is created if there exists (u,v) in conflict[x] such that u in S and v in S.
    def creates_violation(S, x):
        for u, v in conflict[x]:
            if u in S and v in S:
                return True
        return False

    # Helper to calculate the change in violation count if we add x to S
    def delta_violations_add(S, x):
        count = 0
        for u, v in conflict[x]:
            if u in S and v in S:
                count += 1
        return count

    # Helper to calculate the change in violation count if we remove x from S
    def delta_violations_remove(S, x):
        count = 0
        for u, v in conflict[x]:
            if u in S and v in S:
                count += 1
        return count

    # Initial Heuristic: Greedy Randomized
    # Try a few different random greedy starts
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(numbers)
        S = set()
        for x in numbers:
            if not creates_violation(S, x):
                S.add(x)
        
        e = energy(S)
        if e < best_energy:
            best_energy = e
            best_set = set(S)

    # Simulated Annealing
    # We will run SA from the best greedy solution found, and also from a few random valid solutions.
    # Move: 
    # 1. Add a random element x not in S.
    # 2. Remove a random element x in S.
    # 3. Swap: Remove x in S, Add y not in S.
    
    # To make SA effective, we need to be able to evaluate the energy change quickly.
    # Maintaining a full count of violations is O(T) where T is number of triples.
    # T is roughly a few hundred. This is fast enough.
    
    # Let's optimize the violation counting.
    # Instead of iterating all triples, we can maintain a dictionary of "active" triples?
    # No, T is small enough (< 1000 likely). Let's just recompute or use delta.
    
    # Let's use delta updates for speed.
    # current_violations = get_violations_count(S)
    
    def sa_run(initial_S, steps=50000):
        S = set(initial_S)
        current_violations = get_violations_count(S)
        current_energy = -len(S) + 10.0 * current_violations
        
        local_best_S = set(S)
        local_best_energy = current_energy
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        for step in range(steps):
            if time.time() > deadline - 2:
                break
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element
                candidates = [x for x in numbers if x not in S]
                if not candidates:
                    continue
                x = random.choice(candidates)
                
                # Calculate delta
                dv = delta_violations_add(S, x)
                # Energy change: -1 (size +1) + 10 * dv
                delta_E = -1.0 + 10.0 * dv
                
                if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                    S.add(x)
                    current_violations += dv
                    current_energy += delta_E
            elif move_type < 0.8:
                # Remove a random element
                if not S:
                    continue
                x = random.choice(list(S))
                
                # Calculate delta
                dv = delta_violations_remove(S, x)
                # Energy change: +1 (size -1) - 10 * dv
                delta_E = 1.0 - 10.0 * dv
                
                if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                    S.remove(x)
                    current_violations -= dv
                    current_energy += delta_E
            else:
                # Swap
                if not S:
                    continue
                x = random.choice(list(S))
                candidates = [y for y in numbers if y not in S]
                if not candidates:
                    continue
                y = random.choice(candidates)
                
                # Remove x, Add y
                # Order matters for intermediate state? No, we can compute net delta.
                # But we must ensure we don't have x==y (impossible by definition)
                
                # Delta for removing x
                dv_rem = delta_violations_remove(S, x)
                # Delta for adding y to (S - {x})
                # We need to check conflicts of y with S \ {x}
                # Note: If x is in conflict[y], it might affect the count.
                # Let's just simulate:
                S_temp = S - {x}
                dv_add = delta_violations_add(S_temp, y)
                
                # Total delta in violations: dv_add - dv_rem
                # Total delta in size: 0
                delta_E = 10.0 * (dv_add - dv_rem)
                
                if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                    S.remove(x)
                    S.add(y)
                    current_violations += dv_add - dv_rem
                    current_energy += delta_E
            
            if current_energy < local_best_energy:
                local_best_energy = current_energy
                local_best_S = set(S)
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return local_best_S, local_best_energy

    # Run SA from best greedy
    if time.time() < deadline - 5:
        sa_S, sa_E = sa_run(best_set, steps=100000)
        if sa_E < best_energy:
            best_energy = sa_E
            best_set = sa_S

    # Run SA from a few random valid sets
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        random.shuffle(numbers)
        S_rand = set()
        for x in numbers:
            if not creates_violation(S_rand, x):
                S_rand.add(x)
        
        sa_S, sa_E = sa_run(S_rand, steps=50000)
        if sa_E < best_energy:
            best_energy = sa_E
            best_set = sa_S

    # Final cleanup: Ensure the best_set is valid.
    # SA might leave us in an invalid state if we didn't find a valid one with lower energy.
    # But energy includes penalty. If best_set has violations, we should try to repair it.
    # However, the task requires a VALID set.
    # If best_set is invalid, we must project it to the nearest valid set or discard.
    # Given the penalty weight 10, valid sets (E = -|S|) are preferred over invalid ones (E = -|S| + 10*k)
    # unless |S| is much larger.
    # Let's check validity.
    
    def is_valid_set(S):
        for a, b, c in bad_triples:
            if a in S and b in S and c in S:
                return False
        return True

    if not is_valid_set(best_set):
        # Repair: Remove elements involved in violations until valid.
        # This might reduce size.
        # Better: Use the greedy "add back" strategy on a subset.
        # Let's try to find the largest valid subset of best_set.
        # Or just run the greedy repair on best_set.
        
        # Simple repair:
        S_repair = set(best_set)
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            # Find any violation
            viol_found = False
            for a, b, c in bad_triples:
                if a in S_repair and b in S_repair and c in S_repair:
                    # Remove the one with highest conflict count? Or random?
                    # Remove 'a' for now
                    S_repair.remove(a)
                    changed = True
                    viol_found = True
                    break
            if not viol_found:
                break
        
        if is_valid_set(S_repair):
            best_set = S_repair
        else:
            # Fallback to the best valid set found during SA or greedy
            # We didn't track the best *valid* set separately if the energy was lower for invalid.
            # Let's just take the largest valid set from our history.
            # Actually, the greedy starts were valid.
            # Let's re-run a quick greedy to ensure we have a valid set.
            random.shuffle(numbers)
            S_greedy = set()
            for x in numbers:
                if not creates_violation(S_greedy, x):
                    S_greedy.add(x)
            if len(S_greedy) > len(best_set):
                best_set = S_greedy

    # One final local search polish on the valid best_set
    if time.time() < deadline - 2 and is_valid_set(best_set):
        current = set(best_set)
        # Add phase
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for x in numbers:
                if x not in current and not creates_violation(current, x):
                    current.add(x)
                    changed = True
        
        # Swap phase
        for _ in range(1000):
            if time.time() > deadline - 1:
                break
            elems = list(current)
            if not elems:
                break
            out_elem = random.choice(elems)
            trial = set(current)
            trial.discard(out_elem)
            # Try to add best replacement
            best_in = None
            for in_elem in numbers:
                if in_elem in trial:
                    continue
                if not creates_violation(trial, in_elem):
                    trial2 = set(trial)
                    trial2.add(in_elem)
                    # Try to add more
                    local_changed = True
                    while local_changed and time.time() < deadline - 1:
                        local_changed = False
                        for y in numbers:
                            if y not in trial2 and not creates_violation(trial2, y):
                                trial2.add(y)
                                local_changed = True
                    if len(trial2) > len(current):
                        best_in = in_elem
                        current = trial2
                        break
        
        best_set = current

    # Ensure non-empty
    if not best_set:
        best_set = {1}

    return sorted(list(best_set))
