"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    best_result = None
    best_score = float('inf')
    
    # Strategy: 
    # 1. Generate candidate value sets that are "collision-friendly".
    #    Small values create more sum/product collisions.
    #    Arithmetic Progressions (AP) create many sum collisions (e.g., 1+4=2+3).
    #    However, APs create many distinct products if spread out.
    #    Let's try sets with small numbers, potentially with duplicates allowed? 
    #    No, "fourteen distinct positive integers".
    #    So we must pick 14 distinct ints.
    #    To minimize distinct sums, we want the set to be "additively degenerate".
    #    Small consecutive integers: {1..14} have sums in [3..27]. 25 distinct sums.
    #    Can we do better?
    #    If we use {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 
    #    sums are 3..27. Count = 25.
    #    Products: 1*2=2 ... 13*14=182. Many distinct products.
    #    
    #    Alternative: Use a set that minimizes product variety.
    #    If values are close, products are close but distinct.
    #    If values are powers of 2, products are distinct powers (bad for product count, good for sum count? No, sums of powers of 2 are unique in binary -> 2^14 sums, very bad).
    #    
    #    Let's analyze the trade-off.
    #    Max distinct sums for 14 distinct ints is at most (max+min-1) if they form an AP.
    #    For {1..14}, sums = 25.
    #    For {1,2,4,8,...}, sums = 2^14 (huge).
    #    So APs are good for sums.
    #    
    #    What about products for AP {1..14}?
    #    Products range from 2 to 182.
    #    How many distinct products?
    #    Let's check a few.
    #    1*2=2, 1*3=3, ..., 1*14=14.
    #    2*2 not allowed (distinct vertices).
    #    2*3=6, 2*4=8, 2*5=10...
    #    3*4=12, 3*5=15...
    #    There will be many collisions (e.g., 2*6=12, 3*4=12).
    #    But likely > 25 distinct products.
    #    
    #    Let's try to minimize the MAX of the two counts.
    #    If we can get sums down to 25, can we get products down to 25?
    #    Or can we get both to ~30?
    #    
    #    Let's try a few specific value sets and optimize edges for them.
    
    candidate_value_sets = []
    
    # Set 1: Consecutive integers 1..14
    candidate_value_sets.append(list(range(1, 15)))
    
    # Set 2: Consecutive integers 2..15
    candidate_value_sets.append(list(range(2, 16)))
    
    # Set 3: Small integers with a gap to break some product patterns?
    # Try {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} again but maybe shift?
    # Shifting doesn't change sum counts, only product counts increase.
    # So 1..14 is better for products than 2..15.
    
    # Set 4: Try to cluster values to increase product collisions.
    # e.g. {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    # What if we use {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20}?
    # This breaks the AP sum property. Sums will be more distinct.
    # AP is best for minimizing sum distinctness.
    
    # Let's stick with APs of small numbers.
    # Try different starting points?
    # {1..14} is best for products (smaller values).
    # Try {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    
    # Are there non-AP sets with fewer distinct sums?
    # For a set of size n, the minimum number of distinct sums is 2n-3 (for AP).
    # So 2*14-3 = 25 is the theoretical minimum for distinct sums.
    # So we cannot go below 25 distinct sums.
    # Therefore, the answer is at least 25.
    # Can we achieve 25 distinct sums AND 25 distinct products?
    # If yes, score is 25.
    # If products are > 25, score is determined by products.
    
    # Let's evaluate {1..14} with optimal edges.
    # To minimize distinct products, we should pick edges that reuse product values.
    # This is a graph problem: given a set of values, find 35 edges that minimize distinct products.
    
    # Heuristic for Edge Selection:
    # 1. Calculate all possible products for all pairs.
    # 2. Group pairs by product value.
    # 3. We need to select 35 pairs.
    # 4. To minimize distinct products, we want to pick as many pairs as possible from the same product group.
    # 5. However, we also care about sums.
    # 6. Let's prioritize minimizing the max(count_sums, count_prods).
    
    # Since min sums is 25, let's see if we can keep prods <= 25.
    # If not, we minimize prods.
    
    # Let's implement a greedy edge selector for a fixed value set.
    
    def optimize_edges(values):
        n = len(values)
        all_pairs = list(combinations(range(n), 2))
        
        # Calculate product for each pair
        pair_products = {}
        for i, j in all_pairs:
            p = values[i] * values[j]
            if p not in pair_products:
                pair_products[p] = []
            pair_products[p].append((i, j))
        
        # Sort product groups by size (larger groups first) to maximize reuse
        # But we also need to consider sums.
        # Let's try a greedy approach that balances both.
        
        # Strategy: 
        # We have a budget of "distinct sums" and "distinct products".
        # Start with empty edge set.
        # At each step, pick the edge that adds the "least" new distinct values to the union of sums and prods.
        # Specifically, minimize (1 if new_sum else 0) + (1 if new_prod else 0).
        # If tie, prefer edges that allow more future reuse? Hard to know.
        
        # Simpler:
        # 1. Sort all pairs by product frequency (descending).
        # 2. Sort by sum frequency? 
        # Let's try a local search / greedy hybrid.
        
        # Greedy:
        # Maintain current set of sums and prods.
        # Candidate edges: all not yet selected.
        # For each candidate, calculate cost = (1 if sum not in current_sums else 0) + (1 if prod not in current_prods else 0).
        # Pick min cost. If tie, pick one that is part of a larger product group?
        
        # This is O(m * E) which is 35 * 91 = 3185 ops per step, 35 steps. Very fast.
        
        current_edges = []
        current_sums = set()
        current_prods = set()
        used_pairs = set()
        
        for _ in range(m):
            best_pair = None
            best_cost = float('inf')
            best_tie_breaker = 0
            
            for i, j in all_pairs:
                if (i, j) in used_pairs:
                    continue
                
                s = values[i] + values[j]
                p = values[i] * values[j]
                
                cost = 0
                if s not in current_sums:
                    cost += 1
                if p not in current_prods:
                    cost += 1
                
                # Tie breaker: prefer pairs that have many other pairs with the same product
                # This encourages clustering around high-frequency products
                prod_count = len(pair_products[p])
                
                if cost < best_cost or (cost == best_cost and prod_count > best_tie_breaker):
                    best_cost = cost
                    best_pair = (i, j)
                    best_tie_breaker = prod_count
            
            if best_pair is None:
                break
            
            used_pairs.add(best_pair)
            current_edges.append(best_pair)
            current_sums.add(values[best_pair[0]] + values[best_pair[1]])
            current_prods.add(values[best_pair[0]] * values[best_pair[1]])
        
        return current_edges

    # Try a few value sets
    # 1. {1..14}
    # 2. {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} (same)
    # 3. Maybe a set with more product collisions?
    #    e.g. {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    #    What if we replace 14 with something that creates more product collisions?
    #    14*1=14, 14*2=28, 14*3=42...
    #    12*2=24, 12*3=36...
    #    13*2=26, 13*3=39...
    #    11*2=22, 11*3=33...
    #    10*2=20, 10*3=30...
    #    9*2=18, 9*3=27...
    #    8*2=16, 8*3=24 (collision with 12*2)
    #    7*2=14 (collision with 14*1)
    #    6*2=12 (collision with 3*4)
    #    5*2=10 (collision with 2*5)
    #    4*2=8 (collision with 1*8)
    #    3*2=6 (collision with 1*6)
    #    2*2 not allowed.
    
    #    It seems {1..14} has good collision potential.
    
    # Let's also try a random search around {1..14} to see if we can improve.
    # But first, let's just run the greedy on {1..14} and a few variations.
    
    value_sets_to_try = [
        list(range(1, 15)),
        list(range(2, 16)), # likely worse
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20], # break AP
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15], # break AP
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16], # break AP
    ]
    
    # Add some random sets
    for _ in range(50):
        if time.time() > deadline:
            break
        # Random set of 14 distinct ints from 1..20
        vals = random.sample(range(1, 21), 14)
        vals.sort()
        value_sets_to_try.append(vals)
        
    for vals in value_sets_to_try:
        if time.time() > deadline:
            break
        
        edges = optimize_edges(vals)
        score = _raw_value(vals, edges)
        
        if score < best_score:
            best_score = score
            best_result = {'values': vals, 'edges': [list(e) for e in edges]}
            
            # If we hit the theoretical minimum for sums (25) and products are also low,
            # we might be near optimal.
            if best_score <= 25:
                break

    # If no result found (shouldn't happen), fallback
    if best_result is None:
        vals = list(range(1, 15))
        edges = optimize_edges(vals)
        best_result = {'values': vals, 'edges': [list(e) for e in edges]}
        
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
