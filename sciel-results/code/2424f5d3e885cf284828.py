import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) clauses
    # A clause is a set of 8 triples that must not all be present.
    # Defined by 6 distinct vertices and a partition into 3 pairs.
    
    clauses = []
    # To avoid duplicates, we can just iterate over all 6-sets and all matchings.
    # Number of 6-sets: C(16,6) = 8008.
    # Number of matchings on 6 vertices: 15.
    # Total clauses: 8008 * 15 = 120,120.
    # Each clause has 8 triples.
    # This is manageable.
    
    def get_matchings(verts):
        # verts is a tuple of 6 unique ints
        matchings = []
        # Fix vertex 0. Pair it with one of the other 5.
        for i in range(1, 6):
            p0 = (verts[0], verts[i])
            rest = [v for j, v in enumerate(verts) if j != 0 and j != i]
            # rest has 4 vertices. Pair them.
            # 3 ways to pair 4 vertices: (0,1)-(2,3), (0,2)-(1,3), (0,3)-(1,2)
            # relative to rest indices
            matchings.append([p0, (rest[0], rest[1]), (rest[2], rest[3])])
            matchings.append([p0, (rest[0], rest[2]), (rest[1], rest[3])])
            matchings.append([p0, (rest[0], rest[3]), (rest[1], rest[2])])
        return matchings

    for verts in combinations(range(n), 6):
        for matching in get_matchings(verts):
            p1, p2, p3 = matching
            transversals = []
            for u in p1:
                for v in p2:
                    for w in p3:
                        # Use tuple for consistency
                        t = tuple(sorted((u, v, w)))
                        transversals.append(t)
            clause_indices = [triple_to_idx[t] for t in transversals]
            clauses.append(clause_indices)
            
    var_to_clauses = [[] for _ in range(num_triples)]
    for c_idx, clause in enumerate(clauses):
        for v in clause:
            var_to_clauses[v].append(c_idx)
    
    best_score = 0
    best_x = None
    num_restarts = 500 # High number of restarts as per knowledge
    
    for seed in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        random.seed(seed)
        x = [1] * num_triples
        
        # Initialize clause sums
        clause_sums = [8] * len(clauses)
        violated = set(range(len(clauses)))
        
        # Precompute violation count per variable for speed?
        # var_violation_count[v] = number of violated clauses containing v
        var_violation_count = [0] * num_triples
        for c_idx in violated:
            for v in clauses[c_idx]:
                var_violation_count[v] += 1

        # Local Search: Remove triples to satisfy constraints
        # We use a heuristic: remove the triple in a violated clause that is in the most other violated clauses.
        # With some probability, choose randomly to escape local optima.
        
        max_steps = 10000 # Cap steps per restart to ensure restarts happen
        steps = 0
        
        while violated and steps < max_steps:
            if time.time() > deadline - 1:
                break
            
            # Pick a violated clause
            c_idx = random.choice(list(violated))
            
            # Pick a triple to remove from this clause
            # Heuristic: highest var_violation_count
            # Randomness: 10% chance to pick random
            if random.random() < 0.1:
                v = random.choice(clauses[c_idx])
            else:
                # Find v in clauses[c_idx] with max var_violation_count
                best_v = -1
                max_count = -1
                for v in clauses[c_idx]:
                    if var_violation_count[v] > max_count:
                        max_count = var_violation_count[v]
                        best_v = v
                v = best_v
            
            if x[v] == 1:
                x[v] = 0
                # Update clause sums and violation counts
                for c in var_to_clauses[v]:
                    clause_sums[c] -= 1
                    if clause_sums[c] == 7:
                        # Clause c is no longer violated
                        if c in violated:
                            violated.discard(c)
                            # Update var_violation_count for all vars in c
                            for u in clauses[c]:
                                var_violation_count[u] -= 1
            steps += 1
            
        if violated:
            # Failed to satisfy constraints in this restart
            continue
            
        # Polish Phase: Try to add back removed triples
        # This is crucial for maximizing the count.
        # We iterate through all triples that are currently 0.
        # To make it faster, we can just iterate all triples.
        # If adding a triple doesn't cause any clause sum to exceed 7 (i.e., become 8), we add it.
        
        # Note: Adding a triple increases clause_sums for clauses containing it.
        # We must ensure no clause becomes 8.
        
        # We can do this in random order to find different optima.
        removed_indices = [i for i in range(num_triples) if x[i] == 0]
        random.shuffle(removed_indices)
        
        for v in removed_indices:
            if time.time() > deadline - 1:
                break
            # Check if we can add v
            # We can add v if for all clauses c containing v, clause_sums[c] < 7
            # i.e., clause_sums[c] <= 6
            can_add = True
            for c in var_to_clauses[v]:
                if clause_sums[c] >= 7:
                    can_add = False
                    break
            
            if can_add:
                x[v] = 1
                for c in var_to_clauses[v]:
                    clause_sums[c] += 1
                    # Note: clause_sums[c] becomes 7. It is not violated yet.
                    # But if we add another triple to this clause later, it will become 8.
                    # The check `clause_sums[c] >= 7` prevents this.
                
        score = sum(x)
        
        if score > best_score:
            best_score = score
            best_x = x[:]

    if best_x is None:
        return []
        
    result = []
    for i, val in enumerate(best_x):
        if val == 1:
            result.append(list(all_triples[i]))
            
    return result
