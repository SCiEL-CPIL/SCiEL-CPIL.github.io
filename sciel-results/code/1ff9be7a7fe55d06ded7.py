"""
Maximize number of triples in a 3-uniform hypergraph on 16 vertices
such that no three disjoint pairs have all 8 transversal triples present.

Strategy:
1. Precompute all K(2,2,2) configurations (each is a set of 8 triples).
2. Use local search (WalkSAT-style) with multi-swap moves to maximize
   the number of triples while keeping all K(2,2,2) constraints satisfied.
3. Use degree-based greedy initialization for diverse starting points.
4. Time-bounded with multiple restarts.
"""

import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s (evaluator kills at 150s)
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_total = len(all_triples)  # 560
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs of vertices.
    # The 8 transversal triples are formed by picking one vertex from each pair.
    # We need to enumerate all ways to partition 6 vertices into 3 pairs,
    # then for each such partition, the 8 transversals are the triples.
    
    # Actually, a K(2,2,2) is a complete 3-partite 3-uniform hypergraph with parts of size 2.
    # It's defined by choosing 6 distinct vertices and partitioning them into 3 pairs.
    # The 8 edges are all triples picking one from each pair.
    
    # Let's enumerate: choose 6 vertices from 16, then partition into 3 pairs.
    # Number of ways to partition 6 labeled items into 3 unordered pairs: 15
    # So total K(2,2,2) configs = C(16,6) * 15 = 8008 * 15 = 120120
    
    # For efficiency, precompute for each triple which K(2,2,2) configs it belongs to.
    # But 120120 configs * 8 triples each = ~960K memberships, manageable.
    
    # Alternative approach: for local search, we need to quickly check if adding/removing
    # a triple creates/violates a K(2,2,2). 
    # 
    # Key insight: A triple t = {a,b,c} participates in a K(2,2,2) if there exist 
    # three disjoint pairs whose transversals include t. 
    # 
    # Let's think differently. For a given set of triples S, a violation exists iff
    # there are 3 disjoint pairs (A,B,C) such that all 8 transversals are in S.
    # 
    # For local search, we can maintain for each K(2,2,2) config how many of its 8 triples
    # are currently in S. A violation occurs when any counter reaches 8.
    # 
    # But storing 120120 counters and updating them on each add/remove might be slow.
    # Let's estimate: each triple belongs to how many K(2,2,2) configs?
    # A triple {a,b,c}: we need to find 3 disjoint pairs whose transversals include {a,b,c}.
    # The triple picks one from each pair. So one pair contains a, another contains b, 
    # another contains c. The other elements of the pairs must be from {0..15} \ {a,b,c}.
    # Pair 1: {a, x} where x not in {a,b,c}
    # Pair 2: {b, y} where y not in {a,b,c,x}
    # Pair 3: {c, z} where z not in {a,b,c,x,y}
    # So number of K(2,2,2) containing a given triple = 13 * 12 * 11 = 1716
    # That's a lot. Total memberships = 560 * 1716 = ~960K, which matches.
    
    # Let's precompute: for each triple index, list of K(2,2,2) config indices it belongs to.
    
    # First, enumerate all K(2,2,2) configs
    k222_configs = []  # list of frozensets of 8 triple indices
    triple_to_k222 = [[] for _ in range(num_total)]
    
    for six_vertices in itertools.combinations(range(N), 6):
        # Partition 6 vertices into 3 pairs
        # There are 15 ways
        a, b, c, d, e, f = six_vertices
        partitions = [
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
        for p1, p2, p3 in partitions:
            # Generate 8 transversals
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            # Convert to indices
            trans_indices = frozenset(triple_to_idx[t] for t in transversals)
            k222_configs.append(trans_indices)
            config_idx = len(k222_configs) - 1
            for ti in trans_indices:
                triple_to_k222[ti].append(config_idx)
    
    num_k222 = len(k222_configs)
    
    # Local search state
    # present[i] = True if triple i is in our set
    # k222_count[j] = number of triples from k222_configs[j] that are present
    
    def time_check():
        return time.time() >= deadline
    
    best_result = []
    best_count = 0
    
    # Greedy initialization
    def greedy_init(rng):
        present = [False] * num_total
        k222_count = [0] * num_k222
        count = 0
        
        # Randomize order of triples
        order = list(range(num_total))
        rng.shuffle(order)
        
        for ti in order:
            # Check if adding this triple would create a violation
            can_add = True
            for ci in triple_to_k222[ti]:
                if k222_count[ci] == 7:
                    can_add = False
                    break
            if can_add:
                present[ti] = True
                count += 1
                for ci in triple_to_k222[ti]:
                    k222_count[ci] += 1
        return present, k222_count, count
    
    # Local search with safe moves: Add-only and 1-for-1 swaps
    def local_search(present, k222_count, count, rng, max_steps=200000):
        steps = 0
        no_improve = 0
        max_no_improve = 10000  # Allow more swaps to escape local maxima
        
        while steps < max_steps and no_improve < max_no_improve:
            if time_check():
                break
            
            moved = False
            
            # 1. Try to add a triple (improves count)
            for _ in range(50):
                ti = rng.randint(0, num_total - 1)
                if present[ti]:
                    continue
                can_add = True
                for ci in triple_to_k222[ti]:
                    if k222_count[ci] == 7:
                        can_add = False
                        break
                if can_add:
                    present[ti] = True
                    count += 1
                    for ci in triple_to_k222[ti]:
                        k222_count[ci] += 1
                    moved = True
                    no_improve = 0
                    break
            
            if moved:
                steps += 1
                continue

            # 2. Try a 1-for-1 swap (neutral count, changes state)
            # This helps escape local minima where no addition is possible.
            swap_done = False
            for _ in range(50):
                ti_remove = rng.randint(0, num_total - 1)
                if not present[ti_remove]:
                    continue
                
                # Remove
                for ci in triple_to_k222[ti_remove]:
                    k222_count[ci] -= 1
                present[ti_remove] = False
                count -= 1
                
                # Try to add a different triple
                found_add = False
                ti_add = -1
                for _ in range(100):
                    ti_candidate = rng.randint(0, num_total - 1)
                    if present[ti_candidate]:
                        continue
                    can_add = True
                    for ci in triple_to_k222[ti_candidate]:
                        if k222_count[ci] == 7:
                            can_add = False
                            break
                    if can_add:
                        ti_add = ti_candidate
                        found_add = True
                        break
                
                if found_add:
                    # Accept swap
                    present[ti_add] = True
                    count += 1
                    for ci in triple_to_k222[ti_add]:
                        k222_count[ci] += 1
                    swap_done = True
                    no_improve = 0
                else:
                    # Revert removal
                    for ci in triple_to_k222[ti_remove]:
                        k222_count[ci] += 1
                    present[ti_remove] = True
                    count += 1
            
            if swap_done:
                steps += 1
                continue

            # If we couldn't add or swap, we are stuck.
            no_improve += 1
            steps += 1
        
        return present, count
    
    # Main loop with multiple restarts
    rng = random.Random(42)
    
    for restart in range(200):
        if time_check():
            break
        
        # Greedy init with randomization
        present, k222_count, count = greedy_init(rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        # Local search
        present, count = local_search(present, k222_count, count, rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        if time_check():
            break
    
    if not best_result:
        # Fallback: return the seed solution
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result
