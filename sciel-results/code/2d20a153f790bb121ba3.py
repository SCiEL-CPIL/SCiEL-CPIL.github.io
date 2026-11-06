"""Direct structural approach for equal-LCM-triple-free subset of [1,128]."""
import time
import random
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
    
    # Build conflict structure: for each element, which pairs with it form bad triples
    # conflict_pairs[x] = list of (y, z) such that (x, y, z) is a bad triple
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
        
    def get_conflict_delta_add(x, S):
        """Calculate increase in conflict count if x is added to S."""
        delta = 0
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                delta += 1
        return delta

    def get_conflict_delta_remove(x, S):
        """Calculate decrease in conflict count if x is removed from S."""
        delta = 0
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                delta += 1
        return delta

    def count_conflicts(S):
        """Calculate number of bad triples in S."""
        count = 0
        for x in S:
            for (y, z) in conflict_pairs[x]:
                if y in S and z in S:
                    count += 1
        return count // 3

    best_set = set()
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    def greedy_fill(S, order):
        for x in order:
            if x not in S:
                if get_conflict_delta_add(x, S) == 0:
                    S.add(x)
        return S

    def local_search(S, time_limit):
        """Iterative improvement: remove elements from bad triples, then refill."""
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Find all bad triples in S
            current_bad = []
            for x in S:
                for (y, z) in conflict_pairs[x]:
                    if y in S and z in S:
                        current_bad.append((x, y, z))
            
            if not current_bad:
                break
            
            # For each bad triple, find the element whose removal breaks the most triples
            # Count how many bad triples each element participates in
            triple_count = {}
            for (a, b, c) in current_bad:
                triple_count[a] = triple_count.get(a, 0) + 1
                triple_count[b] = triple_count.get(b, 0) + 1
                triple_count[c] = triple_count.get(c, 0) + 1
            
            # Remove the element with highest participation
            if triple_count:
                worst = max(triple_count, key=lambda k: triple_count[k])
                if worst in S:
                    S.remove(worst)
                    improved = True
        
        # Refill greedily
        for x in elements:
            if x not in S and get_conflict_delta_add(x, S) == 0:
                S.add(x)
        
        return S

    def simulated_annealing(start_set, time_limit):
        S = set(start_set)
        current_energy = count_conflicts(S)
        
        T = 1.0
        cooling = 0.999
        min_T = 1e-4
        
        while time.time() < time_limit and T > min_T:
            move_type = random.random()
            if move_type < 0.4:
                # Add random element
                candidates = [x for x in elements if x not in S]
                if not candidates: continue
                x = random.choice(candidates)
                delta_E = get_conflict_delta_add(x, S)
                new_E = current_energy + delta_E
                
                if new_E < current_energy:
                    S.add(x)
                    current_energy = new_E
                elif new_E == current_energy:
                    S.add(x)
                    current_energy = new_E
                else:
                    if random.random() < pow(2.71828, -delta_E / T):
                        S.add(x)
                        current_energy = new_E
                        
            elif move_type < 0.8:
                # Remove random element
                if not S: continue
                x = random.choice(list(S))
                delta_E = -get_conflict_delta_remove(x, S)
                new_E = current_energy + delta_E
                
                if new_E < current_energy:
                    S.remove(x)
                    current_energy = new_E
                # Do not accept neutral or worse removals as they reduce size without helping energy
                
            else:
                # Swap
                if not S: continue
                x = random.choice(list(S))
                candidates = [y for y in elements if y not in S]
                if not candidates: continue
                y = random.choice(candidates)
                
                S_temp = S - {x}
                d_rem = -get_conflict_delta_remove(x, S)
                d_add = get_conflict_delta_add(y, S_temp)
                delta_E = d_rem + d_add
                new_E = current_energy + delta_E
                
                if new_E < current_energy:
                    S = S_temp | {y}
                    current_energy = new_E
                elif new_E == current_energy:
                    if random.random() < 0.5:
                        S = S_temp | {y}
                        current_energy = new_E
                else:
                    if random.random() < pow(2.71828, -delta_E / T):
                        S = S_temp | {y}
                        current_energy = new_E
                        
            T *= cooling
            
            if current_energy == 0:
                changed = True
                while changed:
                    changed = False
                    for x in elements:
                        if x not in S and get_conflict_delta_add(x, S) == 0:
                            S.add(x)
                            changed = True
                update_best(S)
                
        update_best(S)
        return S

    random.seed(42)
    
    # Initial set: Greedy
    S0 = set()
    order = sorted(elements, key=lambda x: len(conflict_pairs[x]))
    greedy_fill(S0, order)
    update_best(S0)
    
    if time.time() < deadline - 10:
        simulated_annealing(S0, deadline - 10)
        
    for _ in range(50):
        if time.time() > deadline - 5:
            break
            
        size = random.randint(30, 60)
        S_start = set(random.sample(elements, size))
        
        simulated_annealing(S_start, deadline - 5)
        
    if time.time() < deadline - 2:
        S_best = set(best_set)
        for x in elements:
            if x not in S_best and get_conflict_delta_add(x, S_best) == 0:
                S_best.add(x)
        update_best(S_best)
        
    if not best_set:
        best_set = {1}
        
    return list(best_set)
