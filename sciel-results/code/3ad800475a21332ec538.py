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
                    # This would be the 4th element completing a 4-AP
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
    
    # Start with the greedy seed
    selected = set()
    for x in range(1, N + 1):
        if all(not all(x - j * dd in selected for j in (1, 2, 3)) for dd in range(1, (x - 1) // 3 + 1)):
            selected.add(x)
    
    current_score = count_3aps(selected)
    best_set = set(selected)
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Generate a random move: swap, add, or remove
        move_type = random.random()
        
        if move_type < 0.5:
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
            for a, b, c, d in four_ap_tuples:
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
        
        elif move_type < 0.75:
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
            for a, b, c, d in four_ap_tuples:
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
