"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    
    # Generate all C(16,3) = 560 triples
    all_triples = list(combinations(range(n), 3))
    
    # For each triple, find all K(2,2,2)s it belongs to
    # A K(2,2,2) is defined by 3 disjoint pairs: (a1,a2), (b1,b2), (c1,c2)
    # The 8 transversal triples are {ai, bj, ck} for all i,j,k in {0,1}
    
    # Precompute: for each triple, list of (k222_id, position) 
    # where position indicates which of the 8 slots it fills
    # Better: for each triple, store the list of K(2,2,2) IDs it belongs to
    
    # First, enumerate all K(2,2,2)s
    # A K(2,2,2) is determined by choosing 3 disjoint pairs from 16 points
    # Number of ways: choose 6 points, then partition into 3 pairs
    # C(16,6) * 15 (number of perfect matchings on 6 points)
    
    # For each K(2,2,2), store the 8 triples that form it
    k222_triples = {}  # k222_id -> list of 8 triples
    triple_to_k222s = {t: [] for t in all_triples}  # triple -> list of k222_ids
    
    k222_id = 0
    for six_points in combinations(range(n), 6):
        # Generate all perfect matchings of these 6 points
        # There are 15 perfect matchings on 6 labeled points
        a, b, c, d, e, f = six_points
        matchings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for m in matchings:
            (p1, p2), (q1, q2), (r1, r2) = m
            # 8 transversal triples
            triples_in_k222 = [
                (p1, q1, r1),
                (p1, q1, r2),
                (p1, q2, r1),
                (p1, q2, r2),
                (p2, q1, r1),
                (p2, q1, r2),
                (p2, q2, r1),
                (p2, q2, r2),
            ]
            # Sort each triple for consistency
            triples_in_k222 = [tuple(sorted(t)) for t in triples_in_k222]
            
            # Check if all 8 are distinct (they should be since pairs are disjoint)
            if len(set(triples_in_k222)) == 8:
                k222_triples[k222_id] = triples_in_k222
                for t in triples_in_k222:
                    if t in triple_to_k222s:
                        triple_to_k222s[t].append(k222_id)
                k222_id += 1
    
    # Greedy: add triples in an order that minimizes K(2,2,2) conflicts
    # Use a randomized greedy with multiple restarts
    
    best_result = None
    best_count = 0
    
    # Track for each K(2,2,2) how many of its 8 triples are in the current set
    # When adding a triple, check if any K(2,2,2) it belongs to would reach 8
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919)
        
        # Shuffle triples
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        # Count how many of each K(2,2,2)'s triples are present
        k222_count = [0] * k222_id
        
        selected = set()
        
        for t in shuffled:
            if time.time() > deadline - 5:
                break
            
            # Check if adding t would complete any K(2,2,2)
            can_add = True
            for kid in triple_to_k222s[t]:
                if k222_count[kid] >= 7:
                    can_add = False
                    break
            
            if can_add:
                selected.add(t)
                for kid in triple_to_k222s[t]:
                    k222_count[kid] += 1
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = list(selected)
    
    if best_result is None:
        # Fallback: star at point 0
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return [[a, b, c] for a, b, c in best_result]
# EVOLVE-BLOCK-END
