import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560
    
    # Hardcoded 15 matchings of indices 0-5
    matchings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    # Build K(2,2,2) list: each K is a frozenset of 8 triple indices
    # Also build triple_to_ks: for each triple, list of K indices
    triple_to_ks = [[] for _ in range(num_triples)]
    all_ks = []  # list of frozensets of 8 triple indices
    
    for six_points in combinations(range(n), 6):
        for matching in matchings:
            p1 = (six_points[matching[0][0]], six_points[matching[0][1]])
            p2 = (six_points[matching[1][0]], six_points[matching[1][1]])
            p3 = (six_points[matching[2][0]], six_points[matching[2][1]])
            
            t_indices = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        t = tuple(sorted((x1, x2, x3)))
                        idx = triple_to_idx[t]
                        t_indices.append(idx)
            
            k_idx = len(all_ks)
            all_ks.append(frozenset(t_indices))
            
            for ti in t_indices:
                triple_to_ks[ti].append(k_idx)
    
    num_ks = len(all_ks)
    
    # Greedy with set-based violation tracking
    def greedy_set_based():
        in_set = [False] * num_triples
        k_count = [0] * num_ks  # how many of the 8 triples are in the set
        solution_count = 0
        available = set(range(num_triples))
        
        # Precompute degree for each triple (number of K's it belongs to)
        # Use a randomized order initially
        order = list(range(num_triples))
        random.shuffle(order)
        
        for idx in order:
            if idx not in available:
                continue
            
            # Check if adding idx would complete any K
            can_add = True
            for k_idx in triple_to_ks[idx]:
                if k_count[k_idx] == 7:
                    can_add = False
                    break
            
            if can_add:
                in_set[idx] = True
                solution_count += 1
                for k_idx in triple_to_ks[idx]:
                    k_count[k_idx] += 1
                
                # Remove all triples that would complete a K if added
                # A triple j is now blocked if there exists a K containing both idx and j
                # where k_count[K] would become 8 (i.e., currently 7 after adding idx)
                for k_idx in triple_to_ks[idx]:
                    if k_count[k_idx] == 7:
                        # All other triples in this K are now blocked
                        for j in all_ks[k_idx]:
                            if j != idx and j in available:
                                available.discard(j)
        
        return in_set, solution_count
    
    # Local search: kick phase
    def local_search(in_set, solution_count, time_limit):
        k_count = [0] * num_ks
        for i in range(num_triples):
            if in_set[i]:
                for k_idx in triple_to_ks[i]:
                    k_count[k_idx] += 1
        
        best_count = solution_count
        best_in_set = in_set[:]
        
        while time.time() < time_limit:
            # Pick a random triple to remove (kick)
            if best_count < 1:
                break
            removed = random.randrange(num_triples)
            if not best_in_set[removed]:
                continue
            
            # Remove it
            best_in_set[removed] = False
            for k_idx in triple_to_ks[removed]:
                k_count[k_idx] -= 1
            current_count = best_count - 1
            
            # Try to greedily add back triples
            available = [i for i in range(num_triples) if not best_in_set[i]]
            random.shuffle(available)
            
            for idx in available:
                if time.time() > time_limit:
                    break
                can_add = True
                for k_idx in triple_to_ks[idx]:
                    if k_count[k_idx] == 7:
                        can_add = False
                        break
                if can_add:
                    best_in_set[idx] = True
                    current_count += 1
                    for k_idx in triple_to_ks[idx]:
                        k_count[k_idx] += 1
            
            if current_count > best_count:
                best_count = current_count
                # Keep the improved solution
            else:
                # Revert to best
                pass  # We already have best_in_set modified; need to track properly
        
        return best_in_set, best_count
    
    # Simulated annealing style local search
    def local_search_sa(in_set, solution_count, time_limit):
        k_count = [0] * num_ks
        for i in range(num_triples):
            if in_set[i]:
                for k_idx in triple_to_ks[i]:
                    k_count[k_idx] += 1
        
        best_count = solution_count
        best_in_set = in_set[:]
        current_in_set = in_set[:]
        current_count = solution_count
        
        temperature = 5.0
        cooling = 0.999
        
        while time.time() < time_limit:
            # Pick a random move: add or remove a triple
            idx = random.randrange(num_triples)
            
            if current_in_set[idx]:
                # Remove
                current_in_set[idx] = False
                for k_idx in triple_to_ks[idx]:
                    k_count[k_idx] -= 1
                current_count -= 1
            else:
                # Try to add
                can_add = True
                for k_idx in triple_to_ks[idx]:
                    if k_count[k_idx] == 7:
                        can_add = False
                        break
                if not can_add:
                    continue
                current_in_set[idx] = True
                for k_idx in triple_to_ks[idx]:
                    k_count[k_idx] += 1
                current_count += 1
            
            # Accept if better or with probability exp(-delta/T)
            delta = best_count - current_count  # positive means worse
            if delta <= 0:
                if current_count > best_count:
                    best_count = current_count
                    best_in_set = current_in_set[:]
            else:
                import math
                if random.random() < math.exp(-delta / max(temperature, 0.01)):
                    pass  # accept worse move
                else:
                    # Revert
                    if current_in_set[idx]:
                        current_in_set[idx] = False
                        for k_idx in triple_to_ks[idx]:
                            k_count[k_idx] -= 1
                        current_count -= 1
                    else:
                        current_in_set[idx] = True
                        for k_idx in triple_to_ks[idx]:
                            k_count[k_idx] += 1
                        current_count += 1
            
            temperature *= cooling
        
        return best_in_set, best_count
    
    best_in_set = [False] * num_triples
    best_count = 0
    
    # Phase 1: Multiple greedy restarts
    while time.time() < deadline - 40.0:
        in_set, count = greedy_set_based()
        if count > best_count:
            best_count = count
            best_in_set = in_set
    
    # Phase 2: Local search on best solution
    if time.time() < deadline - 5.0:
        in_set, count = local_search_sa(best_in_set, best_count, deadline - 5.0)
        if count > best_count:
            best_count = count
            best_in_set = in_set
    
    # Convert to result
    result = []
    for i in range(num_triples):
        if best_in_set[i]:
            result.append(list(all_triples[i]))
    
    return result
