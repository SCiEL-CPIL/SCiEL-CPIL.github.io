import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples for fast violation checking
    # A 4-AP is (a, a+d, a+2d, a+3d) with a>=1, a+3d<=N
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples for counting
    # A 3-AP is (a, a+d, a+2d) with a>=1, a+2d<=N
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
    
    # Helper: count 3-APs in a set
    def count_3aps(S):
        count = 0
        for a, b, c in three_ap_tuples:
            if a in S and b in S and c in S:
                count += 1
        return count
        
    # Helper: check if adding element x to set S creates any 4-AP
    # Returns True if no 4-AP is created
    def can_add_without_4ap(S, x):
        for a, b, c, d in four_ap_tuples:
            if x in (a, b, c, d):
                if a in S and b in S and c in S and d in S:
                    return False
        return True
        
    # Helper: check if removing element x from set S breaks any 4-AP that exists
    # We need to count how many 4-APs contain x
    def count_4aps_containing(S, x):
        count = 0
        for a, b, c, d in four_ap_tuples:
            if x in (a, b, c, d):
                if a in S and b in S and c in S and d in S:
                    count += 1
        return count

    # --- Strategy: Greedy Density Construction ---
    # 1. Start with all elements.
    # 2. Iteratively remove the element that is in the most 4-APs.
    # 3. If tie, remove the one that is in the fewest 3-APs (to preserve 3-APs).
    
    # First, let's try a different seed: Randomized Greedy
    # We want to maximize 3-APs while avoiding 4-APs.
    # A good heuristic is to start with a large set and prune 4-APs.
    
    # Let's try a "Greedy Removal" approach:
    # Start with A = [1..N].
    # While A contains a 4-AP, remove one element from it.
    # Which element to remove? The one that breaks the most 4-APs?
    # Or the one that is least critical for 3-APs?
    
    # Let's try a hybrid:
    # 1. Generate a random permutation of [1..N].
    # 2. Iterate through the permutation.
    # 3. If adding the element to the current set does not create a 4-AP, add it.
    # This is a standard greedy set packing/covering approach.
    
    # To improve on the seed (which is a specific greedy), we can use a "Look-ahead" greedy.
    # Or, we can use the "Density" heuristic:
    # Calculate for each x, the number of 4-APs it participates in (if all were present).
    # Calculate for each x, the number of 3-APs it participates in.
    # Sort by (4-AP count / 3-AP count) descending? No, we want to remove high 4-AP/low 3-AP.
    
    # Let's implement a fast local search starting from a good greedy seed.
    # The current seed is:
    # selected = set()
    # for x in range(1, N + 1):
    #     if all(not all(x - j * dd in selected for j in (1, 2, 3)) for dd in range(1, (x - 1) // 3 + 1)):
    #         selected.add(x)
    
    # This seed is actually quite good. Let's enhance the search.
    
    # Precompute adjacency for 4-APs: for each x, list of 4-APs containing x.
    # This allows O(1) or O(k) check for 4-AP violation instead of scanning all 4-APs.
    
    four_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        for x in (a, b, c, d):
            four_ap_neighbors[x].append(idx)
            
    # Precompute adjacency for 3-APs: for each x, list of 3-APs containing x.
    three_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(three_ap_tuples):
        for x in (a, b, c):
            three_ap_neighbors[x].append(idx)
    
    # Start with the greedy seed
    selected = set()
    for x in range(1, N + 1):
        if all(not all(x - j * dd in selected for j in (1, 2, 3)) for dd in range(1, (x - 1) // 3 + 1)):
            selected.add(x)
    
    current_score = count_3aps(selected)
    best_set = set(selected)
    best_score = current_score
    
    # Fast 4-AP check for a set
    def has_4ap(S):
        for idx in range(len(four_ap_tuples)):
            a, b, c, d = four_ap_tuples[idx]
            if a in S and b in S and c in S and d in S:
                return True
        return False

    # Optimized Local Search:
    # We maintain the set S.
    # Move: Swap x in S with y not in S.
    # Condition: S - {x} + {y} must not contain a 4-AP.
    # Objective: Maximize count_3aps(S - {x} + {y}).
    
    # To speed up, we can precompute the "cost" of removing x and adding y.
    # Delta_3ap = (3-APs gained by adding y) - (3-APs lost by removing x) + (interaction terms)
    # This is complex. Let's stick to a fast evaluation but with a better move generator.
    
    # Better Move Generator:
    # 1. Identify "bad" elements: those in many 4-APs (potential conflicts) or low 3-AP contribution.
    # 2. Identify "good" candidates: those not in S, that are in many 3-APs with current S.
    
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Generate a random move: swap, add, or remove
        move_type = random.random()
        
        if move_type < 0.6:
            # Swap: remove one element, add another
            if len(selected) < 2:
                T *= cooling
                continue
            x_remove = random.choice(list(selected))
            candidates = [x for x in range(1, N + 1) if x not in selected]
            if not candidates:
                T *= cooling
                continue
            x_add = random.choice(candidates)
            
            # Check if adding x_add would create a 4-AP (after removing x_remove)
            temp_set = set(selected)
            temp_set.discard(x_remove)
            temp_set.add(x_add)
            
            # Check 4-AP violation
            has_4ap = False
            for idx in four_ap_neighbors[x_add]:
                a, b, c, d = four_ap_tuples[idx]
                if a in temp_set and b in temp_set and c in temp_set and d in temp_set:
                    has_4ap = True
                    break
            if not has_4ap:
                # Also check if removing x_remove breaks a 4-AP? No, removing can't create one.
                # But we must ensure the rest of the set is valid. 
                # Since S was valid, S-{x} is valid. Adding y might create one.
                # The check above is sufficient.
                
                new_score = count_3aps(temp_set)
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(delta / T):
                    selected = temp_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(selected)
        
        elif move_type < 0.8:
            # Add an element
            candidates = [x for x in range(1, N + 1) if x not in selected]
            if not candidates:
                T *= cooling
                continue
            x_add = random.choice(candidates)
            
            temp_set = set(selected)
            temp_set.add(x_add)
            
            # Check 4-AP violation
            has_4ap = False
            for idx in four_ap_neighbors[x_add]:
                a, b, c, d = four_ap_tuples[idx]
                if a in temp_set and b in temp_set and c in temp_set and d in temp_set:
                    has_4ap = True
                    break
            
            if not has_4ap:
                new_score = count_3aps(temp_set)
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(delta / T):
                    selected = temp_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(selected)
        
        else:
            # Remove an element
            if len(selected) < 2:
                T *= cooling
                continue
            x_remove = random.choice(list(selected))
            
            temp_set = set(selected)
            temp_set.discard(x_remove)
            
            new_score = count_3aps(temp_set)
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                selected = temp_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(selected)
        
        T *= cooling
        
    return sorted(best_set)
