import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Find all bad triples
    bad_triples = []
    bad_triple_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
                    bad_triple_set.add((a, b, c))
    
    # Build conflict structure
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
        
    def can_add(x, S):
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                return False
        return True

    def delta_add(x, S):
        delta = 0
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                delta += 1
        return delta

    def delta_remove(x, S):
        delta = 0
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                delta += 1
        return delta

    def calc_energy(S):
        # Only used for initialization
        e = 0
        S_list = sorted(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j+1, len(S_list)):
                    c = S_list[k]
                    if (a, b, c) in bad_triple_set:
                        e += 1
        return e

    best_set = set()
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    def polish(S):
        changed = True
        while changed:
            changed = False
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    changed = True
        return S

    def greedy_init():
        S = set()
        order = sorted(elements, key=lambda x: len(conflict_pairs[x]))
        for x in order:
            if can_add(x, S):
                S.add(x)
        return polish(S)

    def simulated_annealing(start_set, time_limit):
        S = set(start_set)
        current_energy = calc_energy(S)
        
        T = 1.0
        cooling = 0.999
        min_T = 1e-5
        
        while time.time() < time_limit and T > min_T:
            move_type = random.random()
            
            if move_type < 0.3:
                # Add
                candidates = [x for x in elements if x not in S]
                if not candidates: continue
                x = random.choice(candidates)
                d = delta_add(x, S)
                new_E = current_energy + d
                
                if new_E == 0:
                    S.add(x)
                    current_energy = new_E
                elif new_E < current_energy:
                    S.add(x)
                    current_energy = new_E
                else:
                    # Penalty: 100 * Energy
                    delta_cost = 100 * d
                    if random.random() < pow(2.71828, -delta_cost / T):
                        S.add(x)
                        current_energy = new_E

            elif move_type < 0.6:
                # Remove
                if not S: continue
                x = random.choice(list(S))
                d = delta_remove(x, S)
                new_E = current_energy - d
                
                if d > 0:
                    S.remove(x)
                    current_energy = new_E
                else:
                    if random.random() < 0.01:
                        S.remove(x)
                        current_energy = new_E

            elif move_type < 0.9:
                # Swap
                if not S: continue
                x = random.choice(list(S))
                candidates = [y for y in elements if y not in S]
                if not candidates: continue
                y = random.choice(candidates)
                
                S_temp = S - {x}
                d_rem = delta_remove(x, S)
                d_add = delta_add(y, S_temp)
                delta_E = d_add - d_rem
                new_E = current_energy + delta_E
                
                if delta_E < 0:
                    S = S_temp | {y}
                    current_energy = new_E
                elif delta_E == 0:
                    if random.random() < 0.5:
                        S = S_temp | {y}
                        current_energy = new_E
                else:
                    if random.random() < pow(2.71828, -100 * delta_E / T):
                        S = S_temp | {y}
                        current_energy = new_E

            else:
                # 2-Swap
                if len(S) < 2: continue
                x1, x2 = random.sample(list(S), 2)
                candidates = [y for y in elements if y not in S]
                if len(candidates) < 2: continue
                y1, y2 = random.sample(candidates, 2)
                
                S_temp = S - {x1, x2}
                d_rem1 = delta_remove(x1, S)
                # Note: x2 is still in S for d_rem1? No, delta_remove(x1, S) counts triples with x1 and two others in S.
                # If x2 is in S, and {x1, x2, z} is bad, it is counted.
                # When we remove x1 and x2, we remove all triples involving x1 or x2.
                # It is safer to calculate energy of S_temp + {y1, y2} vs S?
                # Or just calculate delta properly.
                
                # Let's just use a simple check:
                # Energy(S) = E
                # Energy(S_temp) = E - (triples with x1) - (triples with x2) + (triples with x1 and x2 counted twice?)
                # This is complex.
                
                # Alternative: Calculate energy of new set directly if it's small?
                # No, too slow.
                
                # Let's skip 2-swap for now to ensure correctness and speed, rely on restarts.
                pass

            T *= cooling
            
            if current_energy == 0:
                S = polish(S)
                update_best(S)

        update_best(S)
        return S

    random.seed(42)
    
    # 1. Greedy
    S0 = greedy_init()
    update_best(S0)
    
    # 2. SA from Greedy
    if time.time() < deadline - 5:
        simulated_annealing(S0, deadline - 5)
        
    # 3. Perturbed Restarts
    # Use remaining time
    while time.time() < deadline - 2:
        # Perturb best set
        if best_set:
            S_start = set(best_set)
            # Remove 5-10 random elements
            k = random.randint(5, 10)
            if len(S_start) > k:
                to_remove = random.sample(list(S_start), k)
                S_start -= set(to_remove)
        else:
            S_start = set(random.sample(elements, 50))
            
        # Run short SA
        t_start = time.time()
        simulated_annealing(S_start, min(deadline - 2, t_start + 1.0))
        
    # Final Polish
    if time.time() < deadline - 1 and best_set:
        S_best = set(best_set)
        S_best = polish(S_best)
        update_best(S_best)

    if not best_set:
        best_set = {1}
        
    return list(best_set)
