import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    all_points = list(range(n))
    
    # Generate all possible triples and index them
    all_triples = []
    triple_to_idx = {}
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                t = (a, b, c)
                triple_to_idx[t] = len(all_triples)
                all_triples.append(t)
    
    total_triples = len(all_triples)
    
    # For each triple, find all other triples that with it could form a K(2,2,2)
    # Actually, let's precompute: for each pair of triples, do they "conflict"?
    # Two triples conflict if there exists a K(2,2,2) where both are transversals.
    
    # Better approach: for each triple t, and for each possible "complement" set of 3 points,
    # check if all 8 transversals would be present.
    
    # Let's use a different strategy: greedy with efficient checking
    # When we add a triple, we mark which "K(2,2,2) patterns" are now "one away"
    
    # Actually, let's just use a greedy approach with a faster validity check
    # Key insight: for triple (a,b,c), a K(2,2,2) containing it requires choosing
    # a' from the 13 other points, b' from remaining 12, c' from remaining 11
    # But we can prune: for each a', we need (a',b,c) to exist, etc.
    
    def check_valid_fast(triple, current_set):
        a, b, c = triple
        # For each possible a', check if (a',b,c) is in set
        # If yes, then for each b', check if (a',b',c) is in set
        # If yes, then for each c', check if all remaining are in set
        
        candidates_a_prime = []
        for a_prime in range(n):
            if a_prime in (a, b, c):
                continue
            t5 = tuple(sorted((a_prime, b, c)))
            if t5 in current_set:
                candidates_a_prime.append(a_prime)
        
        for a_prime in candidates_a_prime:
            candidates_b_prime = []
            for b_prime in range(n):
                if b_prime in (a, b, c, a_prime):
                    continue
                t7 = tuple(sorted((a_prime, b_prime, c)))
                if t7 in current_set:
                    candidates_b_prime.append(b_prime)
            
            for b_prime in candidates_b_prime:
                candidates_c_prime = []
                for c_prime in range(n):
                    if c_prime in (a, b, c, a_prime, b_prime):
                        continue
                    t2 = tuple(sorted((a, b, c_prime)))
                    if t2 in current_set:
                        candidates_c_prime.append(c_prime)
                
                for c_prime in candidates_c_prime:
                    # Check remaining 4 transversals
                    t3 = tuple(sorted((a, b_prime, c)))
                    t4 = tuple(sorted((a, b_prime, c_prime)))
                    t6 = tuple(sorted((a_prime, b, c_prime)))
                    t8 = tuple(sorted((a_prime, b_prime, c_prime)))
                    
                    if (t3 in current_set and t4 in current_set and 
                        t6 in current_set and t8 in current_set):
                        return False
        
        return True
    
    best_set = set()
    best_size = 0
    
    # Multiple random restarts with greedy + local search
    for trial in range(100):
        if time.time() > deadline - 10.0:
            break
        
        order = all_triples[:]
        random.shuffle(order)
        
        current_set = set()
        
        for t in order:
            if time.time() > deadline - 5.0:
                break
            if check_valid_fast(t, current_set):
                current_set.add(t)
        
        # Local search: try to add any missing triple
        if time.time() > deadline - 10.0:
            break
        
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
            improved = False
            for t in all_triples:
                if t in current_set:
                    continue
                if check_valid_fast(t, current_set):
                    current_set.add(t)
                    improved = True
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    result = [list(t) for t in best_set]
    return result
