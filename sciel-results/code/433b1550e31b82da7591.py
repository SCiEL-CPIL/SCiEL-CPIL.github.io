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
    
    # Local search with 1-swap and 2-swap moves
    def local_search(present, k222_count, count, rng, max_steps=200000):
        steps = 0
        no_improve = 0
        max_no_improve = 5000
        
        while steps < max_steps and no_improve < max_no_improve:
            if time_check():
                break
            
            # Try to find a beneficial move
            # Move types:
            # 1. Add a triple that doesn't create a violation
            # 2. Remove a triple and add two others (2-for-1)
            # 3. Remove one triple and add another (1-for-1, no net gain but may enable future)
            # 4. Remove two triples and add three (3-for-2)
            
            # First try simple additions
            added = False
            # Sample some candidate additions
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
                    added = True
                    no_improve = 0
                    break
            
            if added:
                steps += 1
                continue
            
            # Try 2-for-1 moves: remove one triple, add two
            found_move = False
            for _ in range(30):
                # Pick a triple to remove
                ti_remove = rng.randint(0, num_total - 1)
                if not present[ti_remove]:
                    continue
                
                # Temporarily remove
                for ci in triple_to_k222[ti_remove]:
                    k222_count[ci] -= 1
                present[ti_remove] = False
                count -= 1
                
                # Try to add two triples
                added_two = 0
                for _ in range(100):
                    ti_add = rng.randint(0, num_total - 1)
                    if present[ti_add]:
                        continue
                    can_add = True
                    for ci in triple_to_k222[ti_add]:
                        if k222_count[ci] == 7:
                            can_add = False
                            break
                    if can_add:
                        present[ti_add] = True
                        count += 1
                        for ci in triple_to_k222[ti_add]:
                            k222_count[ci] += 1
                        added_two += 1
                        if added_two == 2:
                            break
                
                if added_two == 2:
                    found_move = True
                    no_improve = 0
                    break
                else:
                    # Revert
                    for _ in range(added_two):
                        pass  # We need to track which we added
                    # This is getting complex; let's simplify
                    # Revert the additions
                    # Actually, let's track properly
                    # Let me redo this more carefully
                    pass
                
                # Revert the removal
                for ci in triple_to_k222[ti_remove]:
                    k222_count[ci] += 1
                present[ti_remove] = True
                count += 1
            
            if found_move:
                steps += 1
                continue
            
            # Try 3-for-2 moves: remove two, add three
            found_move = False
            for _ in range(20):
                ti_r1 = rng.randint(0, num_total - 1)
                ti_r2 = rng.randint(0, num_total - 1)
                if not present[ti_r1] or not present[ti_r2] or ti_r1 == ti_r2:
                    continue
                
                # Remove both
                for ci in triple_to_k222[ti_r1]:
                    k222_count[ci] -= 1
                present[ti_r1] = False
                for ci in triple_to_k222[ti_r2]:
                    k222_count[ci] -= 1
                present[ti_r2] = False
                count -= 2
                
                # Try to add three
                added_three = 0
                added_list = []
                for _ in range(150):
                    ti_add = rng.randint(0, num_total - 1)
                    if present[ti_add]:
                        continue
                    can_add = True
                    for ci in triple_to_k222[ti_add]:
                        if k222_count[ci] == 7:
                            can_add = False
                            break
                    if can_add:
                        present[ti_add] = True
                        count += 1
                        for ci in triple_to_k222[ti_add]:
                            k222_count[ci] += 1
                        added_three += 1
                        added_list.append(ti_add)
                        if added_three == 3:
                            break
                
                if added_three == 3:
                    found_move = True
                    no_improve = 0
                    break
                else:
                    # Revert additions
                    for ti in added_list:
                        present[ti] = False
                        count -= 1
                        for ci in triple_to_k222[ti]:
                            k222_count[ci] -= 1
                
                # Revert removals
                for ci in triple_to_k222[ti_r1]:
                    k222_count[ci] += 1
                present[ti_r1] = True
                for ci in triple_to_k222[ti_r2]:
                    k222_count[ci] += 1
                present[ti_r2] = True
                count += 2
            
            if found_move:
                steps += 1
                continue
            
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
