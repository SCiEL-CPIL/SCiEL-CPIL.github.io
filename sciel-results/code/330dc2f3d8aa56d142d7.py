import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) clauses
    def get_matchings(verts):
        matchings = []
        for i in range(1, 6):
            p0 = (verts[0], verts[i])
            rest = [v for j, v in enumerate(verts) if j != 0 and j != i]
            p1a, p1b, p2a, p2b = rest
            matchings.append([p0, (p1a, p1b), (p2a, p2b)])
            matchings.append([p0, (p1a, p2a), (p1b, p2b)])
            matchings.append([p0, (p1a, p2b), (p1b, p2a)])
        return matchings

    clauses = []
    for verts in combinations(range(n), 6):
        for matching in get_matchings(verts):
            p1, p2, p3 = matching
            transversals = []
            for u in p1:
                for v in p2:
                    for w in p3:
                        transversals.append(frozenset((u, v, w)))
            clause_indices = [triple_to_idx[t] for t in transversals]
            clauses.append(clause_indices)
            
    var_to_clauses = [[] for _ in range(num_triples)]
    for c_idx, clause in enumerate(clauses):
        for v in clause:
            var_to_clauses[v].append(c_idx)
    
    def solve_local_search(seed):
        random.seed(seed)
        x = [1] * num_triples
        
        clause_sums = [sum(x[i] for i in clause) for clause in clauses]
        violated = set(i for i, s in enumerate(clause_sums) if s == 8)
        
        if not violated:
            return x
        
        # To pick the best variable, we need to know for each variable in violated clauses,
        # how many violated clauses it belongs to.
        # We can maintain a counter `var_violation_count`.
        # Initially, compute it.
        var_violation_count = [0] * num_triples
        for c_idx in violated:
            for v in clauses[c_idx]:
                var_violation_count[v] += 1
        
        while violated:
            if time.time() > deadline - 2:
                break
            
            # Find variable with max var_violation_count among those in violated clauses
            # Optimization: We can just iterate through all variables? No, 560 is small.
            # But we only care about variables in violated clauses.
            # Let's find the max.
            max_count = -1
            best_var = -1
            # Iterate through violated clauses, and their variables
            # To avoid duplicates, we can just check all 560 variables?
            # 560 checks is very fast.
            for v in range(num_triples):
                if x[v] == 1 and var_violation_count[v] > max_count:
                    max_count = var_violation_count[v]
                    best_var = v
            
            if best_var == -1:
                # Should not happen if violated is not empty
                break
                
            # Flip best_var
            x[best_var] = 0
            
            # Update clause sums and violated set
            for c_idx in var_to_clauses[best_var]:
                clause_sums[c_idx] -= 1
                if clause_sums[c_idx] == 7:
                    # This clause was violated (sum 8) and is now satisfied
                    if c_idx in violated:
                        violated.discard(c_idx)
                        # Decrement violation count for all variables in this clause
                        for v in clauses[c_idx]:
                            var_violation_count[v] -= 1
                # If clause_sums[c_idx] was < 8, it doesn't affect violated set or counts (since we only count violated)
                # Wait, if clause_sums[c_idx] was 7, it wasn't in violated.
                # If it was 8, it was in violated.
                # So we only care if it transitions 8 -> 7.
        
        return x

    best_score = 0
    best_x = None
    
    # Try multiple seeds
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        x = solve_local_search(seed)
        score = sum(x)
        
        if score > best_score:
            best_score = score
            best_x = x
    
    if best_x is None:
        return []
        
    result = []
    for i, val in enumerate(best_x):
        if val == 1:
            result.append(list(all_triples[i]))
            
    return result
