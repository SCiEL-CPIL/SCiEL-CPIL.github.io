import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin for 90s wall-clock
    
    max_val = 128
    
    # 1. Precompute LCMs
    lcm_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # 2. Identify all conflicting triples
    # A triple (a, b, c) with a < b < c is conflicting if
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    conflicting_triples = []
    conflict_map = {}
    
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            l_ab = lcm_table[a][b]
            for c in range(b + 1, max_val + 1):
                l_ac = lcm_table[a][c]
                if l_ac != l_ab:
                    continue
                l_bc = lcm_table[b][c]
                if l_bc == l_ab:
                    conflicting_triples.append((a, b, c))
                    if (a, b) not in conflict_map:
                        conflict_map[(a, b)] = set()
                    conflict_map[(a, b)].add(c)
                    if (a, c) not in conflict_map:
                        conflict_map[(a, c)] = set()
                    conflict_map[(a, c)].add(b)
                    if (b, c) not in conflict_map:
                        conflict_map[(b, c)] = set()
                    conflict_map[(b, c)].add(a)
    
    # 3. Calculate conflict count for each number
    conflict_count = [0] * (max_val + 1)
    for a, b, c in conflicting_triples:
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1
    
    # 4. Fast check function using precomputed data
    # For a given set S (as a boolean array), check if x can be added
    def make_checker():
        # Precompute for each x, the list of (a, b) pairs in S that would conflict
        # Actually, let's use a different approach: for each x, precompute which pairs (a,b)
        # in the conflict triples would cause a problem
        # conflict_triples_with_x[x] = list of (a, b) such that {a,b,x} is a conflicting triple
        conflict_triples_with_x = [[] for _ in range(max_val + 1)]
        for a, b, c in conflicting_triples:
            conflict_triples_with_x[a].append((b, c))
            conflict_triples_with_x[b].append((a, c))
            conflict_triples_with_x[c].append((a, b))
        
        def check_add(x, in_set):
            # in_set is a list or array where in_set[i] = True if i is in the set
            for a, b in conflict_triples_with_x[x]:
                if in_set[a] and in_set[b]:
                    return False
            return True
        
        return check_add, conflict_triples_with_x
    
    check_add, conflict_triples_with_x = make_checker()
    
    # 5. Greedy construction with multiple orderings
    def greedy_construct(order, in_set=None):
        if in_set is None:
            in_set = [False] * (max_val + 1)
        for x in order:
            if not in_set[x] and check_add(x, in_set):
                in_set[x] = True
        return in_set
    
    order = list(range(1, max_val + 1))
    order.sort(key=lambda x: conflict_count[x])
    
    best_in_set = greedy_construct(order)
    best_size = sum(best_in_set)
    
    # Try random orderings
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        random.shuffle(order)
        s = greedy_construct(order)
        sz = sum(s)
        if sz > best_size:
            best_size = sz
            best_in_set = s
    
    # 6. Local search: WalkSAT-style with clause weighting
    # Each conflicting triple is a clause that must NOT be fully satisfied
    # We want to maximize the number of included elements while keeping all clauses unsatisfied
    
    def local_search(start_in_set, max_steps=5000):
        in_set = list(start_in_set)
        current_size = sum(in_set)
        best_local = list(in_set)
        best_local_size = current_size
        
        # Compute initial clause satisfaction
        # A clause (triple) is "satisfied" (bad) if all three are in the set
        # We want to minimize the number of satisfied clauses
        
        # Precompute for each element, which clauses it belongs to
        clauses_for_element = [[] for _ in range(max_val + 1)]
        for idx, (a, b, c) in enumerate(conflicting_triples):
            clauses_for_element[a].append(idx)
            clauses_for_element[b].append(idx)
            clauses_for_element[c].append(idx)
        
        # Track which clauses are currently satisfied
        clause_satisfied = [False] * len(conflicting_triples)
        for idx, (a, b, c) in enumerate(conflicting_triples):
            if in_set[a] and in_set[b] and in_set[c]:
                clause_satisfied[idx] = True
        
        unsatisfied_count = len(conflicting_triples) - sum(clause_satisfied)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Try to add an element
            candidates_to_add = [x for x in range(1, max_val + 1) if not in_set[x]]
            if candidates_to_add:
                random.shuffle(candidates_to_add)
                for x in candidates_to_add:
                    # Check if adding x would create any new satisfied clause
                    would_create = False
                    for clause_idx in clauses_for_element[x]:
                        a, b, c = conflicting_triples[clause_idx]
                        # x is one of a, b, c; check if the other two are in set
                        if a == x:
                            if in_set[b] and in_set[c]:
                                would_create = True
                                break
                        elif b == x:
                            if in_set[a] and in_set[c]:
                                would_create = True
                                break
                        else:
                            if in_set[a] and in_set[b]:
                                would_create = True
                                break
                    if not would_create:
                        in_set[x] = True
                        current_size += 1
                        # Update clause satisfaction
                        for clause_idx in clauses_for_element[x]:
                            a, b, c = conflicting_triples[clause_idx]
                            if not clause_satisfied[clause_idx]:
                                if a == x:
                                    if in_set[b] and in_set[c]:
                                        clause_satisfied[clause_idx] = True
                                elif b == x:
                                    if in_set[a] and in_set[c]:
                                        clause_satisfied[clause_idx] = True
                                else:
                                    if in_set[a] and in_set[b]:
                                        clause_satisfied[clause_idx] = True
                        if current_size > best_local_size:
                            best_local_size = current_size
                            best_local = list(in_set)
                        break
            
            # Try to remove an element that's in a satisfied clause
            satisfied_clauses = [i for i, s in enumerate(clause_satisfied) if s]
            if satisfied_clauses:
                # Pick a random satisfied clause and remove one of its elements
                clause_idx = random.choice(satisfied_clauses)
                a, b, c = conflicting_triples[clause_idx]
                # Choose which element to remove - prefer the one with most conflicts
                candidates = [a, b, c]
                random.shuffle(candidates)
                for x in candidates:
                    if in_set[x]:
                        in_set[x] = False
                        current_size -= 1
                        # Update clause satisfaction
                        for clause_idx2 in clauses_for_element[x]:
                            if clause_satisfied[clause_idx2]:
                                aa, bb, cc = conflicting_triples[clause_idx2]
                                if not (in_set[aa] and in_set[bb] and in_set[cc]):
                                    clause_satisfied[clause_idx2] = False
                        break
            
            # Try swaps: remove one element, add another
            if current_size > 0 and random.random() < 0.3:
                # Pick a random element to remove
                in_elements = [x for x in range(1, max_val + 1) if in_set[x]]
                if in_elements:
                    x_remove = random.choice(in_elements)
                    # Try to add a random element not in set
                    out_elements = [x for x in range(1, max_val + 1) if not in_set[x]]
                    if out_elements:
                        x_add = random.choice(out_elements)
                        # Check if this swap is valid
                        # Temporarily remove x_remove and check if x_add can be added
                        in_set[x_remove] = False
                        can_add = True
                        for clause_idx in clauses_for_element[x_add]:
                            a, b, c = conflicting_triples[clause_idx]
                            if a == x_add:
                                if in_set[b] and in_set[c]:
                                    can_add = False
                                    break
                            elif b == x_add:
                                if in_set[a] and in_set[c]:
                                    can_add = False
                                    break
                            else:
                                if in_set[a] and in_set[b]:
                                    can_add = False
                                    break
                        if can_add:
                            in_set[x_add] = True
                            # Update clause satisfaction for both
                            for clause_idx in clauses_for_element[x_remove]:
                                if clause_satisfied[clause_idx]:
                                    aa, bb, cc = conflicting_triples[clause_idx]
                                    if not (in_set[aa] and in_set[bb] and in_set[cc]):
                                        clause_satisfied[clause_idx] = False
                            for clause_idx in clauses_for_element[x_add]:
                                if not clause_satisfied[clause_idx]:
                                    aa, bb, cc = conflicting_triples[clause_idx]
                                    if in_set[aa] and in_set[bb] and in_set[cc]:
                                        clause_satisfied[clause_idx] = True
                        else:
                            in_set[x_remove] = True
        
        return best_local, best_local_size
    
    # Run local search from multiple starting points
    for restart in range(15):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            start_set = list(best_in_set)
        else:
            random.shuffle(order)
            start_set = greedy_construct(order)
        
        result, result_size = local_search(start_set, max_steps=3000)
        if result_size > best_size:
            best_size = result_size
            best_in_set = result
    
    # 7. Final greedy add pass
    in_set = list(best_in_set)
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for x in range(1, max_val + 1):
            if not in_set[x] and check_add(x, in_set):
                in_set[x] = True
                improved = True
    
    final_size = sum(in_set)
    if final_size > best_size:
        best_size = final_size
        best_in_set = in_set
    
    # Convert to list
    result = [x for x in range(1, max_val + 1) if best_in_set[x]]
    
    return result
