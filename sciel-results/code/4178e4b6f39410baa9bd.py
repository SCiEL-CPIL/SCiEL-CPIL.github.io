import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 180
    elements = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples
    raw_triples = set()
    for a in range(1, N + 1):
        a2 = a * a
        limit = int(math.isqrt(a2))
        for u in range(1, limit + 1):
            if a2 % u == 0:
                v = a2 // u
                if u == v:
                    continue
                b = a + u
                c = a + v
                if b <= N and c <= N:
                    t = tuple(sorted((a, b, c)))
                    raw_triples.add(t)

    triples = list(raw_triples)
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(idx)
            
    num_triples = len(triples)
    
    # 2. Greedy Initialization with better tie-breaking
    def run_greedy(strategy='random'):
        active_elements = set(elements)
        active_triples = set(range(num_triples))
        conflict_count = {i: 0 for i in range(1, N + 1)}
        
        for idx in active_triples:
            for x in triples[idx]:
                conflict_count[x] += 1
        
        while True:
            # Find element with max conflict count
            max_conflict = -1
            candidates = []
            for x in active_elements:
                if conflict_count[x] > max_conflict:
                    max_conflict = conflict_count[x]
                    candidates = [x]
                elif conflict_count[x] == max_conflict:
                    candidates.append(x)
            
            if max_conflict == 0:
                break
            
            # Select victim based on strategy
            if strategy == 'random' and len(candidates) > 1:
                victim = random.choice(candidates)
            elif strategy == 'min':
                victim = min(candidates)
            elif strategy == 'max':
                victim = max(candidates)
            elif strategy == 'degree':
                # Pick element with highest degree in conflict graph
                # Approximate by conflict_count * (number of active neighbors)
                best_victim = None
                best_score = -1
                for x in candidates:
                    # Count how many active elements are in conflicts with x
                    deg = 0
                    for idx in elem_to_triples[x]:
                        if idx in active_triples:
                            a, b, c = triples[idx]
                            for y in (a, b, c):
                                if y != x and y in active_elements:
                                    deg += 1
                    score = conflict_count[x] * 10 + deg
                    if score > best_score:
                        best_score = score
                        best_victim = x
                victim = best_victim
            else:
                victim = min(candidates)

            active_elements.remove(victim)
            
            # Update active_triples and conflict_counts
            for idx in elem_to_triples[victim]:
                if idx in active_triples:
                    active_triples.remove(idx)
                    for x in triples[idx]:
                        if x != victim and x in active_elements:
                            conflict_count[x] -= 1
        
        return active_elements

    # 3. Simulated Annealing with incremental violation tracking
    def simulate_annealing(initial_set, time_budget):
        # State: list of booleans, index i corresponds to element i+1
        present = [False] * N
        for x in initial_set:
            present[x-1] = True
        
        # Precompute for each element, which triples it participates in
        elem_triples = {i: [] for i in range(1, N + 1)}
        for idx, (a, b, c) in enumerate(triples):
            elem_triples[a].append(idx)
            elem_triples[b].append(idx)
            elem_triples[c].append(idx)
        
        # Track violation count for each triple
        triple_violation = [0] * num_triples  # 1 if all three present, 0 otherwise
        total_violations = 0
        
        # Initialize violation counts
        for idx, (a, b, c) in enumerate(triples):
            if present[a-1] and present[b-1] and present[c-1]:
                triple_violation[idx] = 1
                total_violations += 1
        
        def get_violations():
            return total_violations
        
        def get_size():
            return sum(present)
        
        def calc_score():
            return get_size() - 1000 * get_violations()
        
        current_score = calc_score()
        best_score = current_score
        best_present = present[:]
        
        T = 10.0
        cooling = 0.999
        start_sa = time.time()
        
        while time.time() - start_sa < time_budget:
            # Pick random index to flip
            idx = random.randint(0, N - 1)
            elem = idx + 1
            old_val = present[idx]
            present[idx] = not old_val
            
            # Update violations incrementally
            old_violations = total_violations
            for t_idx in elem_triples[elem]:
                a, b, c = triples[t_idx]
                was_violation = triple_violation[t_idx]
                is_violation = present[a-1] and present[b-1] and present[c-1]
                if was_violation and not is_violation:
                    total_violations -= 1
                    triple_violation[t_idx] = 0
                elif not was_violation and is_violation:
                    total_violations += 1
                    triple_violation[t_idx] = 1
            
            new_score = calc_score()
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_present = present[:]
            else:
                # Accept with probability exp(delta / T)
                if T > 0.01 and random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    # Revert
                    present[idx] = old_val
                    for t_idx in elem_triples[elem]:
                        a, b, c = triples[t_idx]
                        was_violation = triple_violation[t_idx]
                        is_violation = present[a-1] and present[b-1] and present[c-1]
                        if was_violation and not is_violation:
                            total_violations -= 1
                            triple_violation[t_idx] = 0
                        elif not was_violation and is_violation:
                            total_violations += 1
                            triple_violation[t_idx] = 1
            
            T *= cooling
            if T < 0.01:
                break
                
        return best_present

    # 4. Smart cleanup: remove minimum elements to fix all violations
    def smart_cleanup(initial_set):
        active = set(initial_set)
        
        # Find all violations
        violations = []
        for idx, (a, b, c) in enumerate(triples):
            if a in active and b in active and c in active:
                violations.append((a, b, c))
        
        if not violations:
            return active
        
        # Greedily remove elements to fix violations
        # For each element, count how many violations it participates in
        elem_viol_count = {x: 0 for x in active}
        for (a, b, c) in violations:
            elem_viol_count[a] += 1
            elem_viol_count[b] += 1
            elem_viol_count[c] += 1
        
        while violations:
            # Find element with max violation count
            max_count = -1
            candidates = []
            for x in active:
                if elem_viol_count[x] > max_count:
                    max_count = elem_viol_count[x]
                    candidates = [x]
                elif elem_viol_count[x] == max_count:
                    candidates.append(x)
            
            if max_count == 0:
                break
            
            # Remove element with max violation count (tie-break: smallest)
            victim = min(candidates)
            active.remove(victim)
            
            # Update violations
            new_violations = []
            for (a, b, c) in violations:
                if a != victim and b != victim and c != victim:
                    new_violations.append((a, b, c))
            violations = new_violations
            
            # Update violation counts
            for x in active:
                elem_viol_count[x] = 0
            for (a, b, c) in violations:
                elem_viol_count[a] += 1
                elem_viol_count[b] += 1
                elem_viol_count[c] += 1
        
        return active

    # Strategy:
    # 1. Run multiple greedy passes with different strategies
    # 2. Run SA on top few seeds
    # 3. Smart cleanup at the end
    
    best_set = set()
    best_size = 0
    greedy_results = []
    
    # Run greedy with different strategies
    strategies = ['random', 'min', 'max', 'degree']
    for strat in strategies:
        for i in range(25):
            if time.time() > deadline - 25:
                break
            s = run_greedy(strategy=strat)
            greedy_results.append((len(s), s))
            if len(s) > best_size:
                best_size = len(s)
                best_set = s
    
    # Sort by size, take top 3 for SA
    greedy_results.sort(key=lambda x: -x[0])
    top_seeds = [s for _, s in greedy_results[:3]]
    
    # Run SA on top seeds
    time_elapsed = time.time() - start_time
    if time_elapsed < deadline - 5:
        sa_budget = (deadline - time_elapsed - 5) / len(top_seeds)
        for seed in top_seeds:
            if time.time() > deadline - 5:
                break
            best_present = simulate_annealing(seed, sa_budget)
            sa_set = {i+1 for i, val in enumerate(best_present) if val}
            # Clean up if needed
            sa_set = smart_cleanup(sa_set)
            if len(sa_set) > best_size:
                best_size = len(sa_set)
                best_set = sa_set
    
    # Final smart cleanup
    final_set = smart_cleanup(best_set)

    return list(final_set)

# EVOLVE-BLOCK-END
