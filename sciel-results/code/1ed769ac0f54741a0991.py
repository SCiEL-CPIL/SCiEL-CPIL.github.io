import time
import random
from itertools import combinations

# Global deadline
START_TIME = time.time()
TIME_LIMIT = 85.0  # Safety margin for 90s limit

def precompute_triple_sums(numbers):
    """Precompute all triple sums for the given numbers."""
    triple_sums = {}
    n = len(numbers)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = numbers[i] + numbers[j] + numbers[k]
                triple_sums.setdefault(s, []).append((i, j, k))
    return triple_sums

def can_add_fast(S_list, x, triple_sum_map):
    """
    Check if x can be added to valid set S_list.
    Uses precomputed triple sums for speed.
    """
    n = len(S_list)
    
    # Check 1: x divides sum of 3 distinct in S_list
    # We need to check if any triple sum is divisible by x
    # Instead of iterating all triples, we can check sums modulo x
    # But that's still O(n^3) in worst case. Let's use a different approach.
    
    # For each triple (i,j,k), check if (S_list[i]+S_list[j]+S_list[k]) % x == 0
    # This is O(n^3) but with small constant.
    # Alternative: precompute for each x the set of residues that would conflict.
    # Since x can be up to 192, and sums up to 576, we can precompute.
    
    # Actually, let's just do the direct check but with early termination.
    for i in range(n):
        si = S_list[i]
        for j in range(i+1, n):
            sj = S_list[j]
            for k in range(j+1, n):
                if (si + sj + S_list[k]) % x == 0:
                    return False
    
    # Check 2: a in S_list divides sum of x and 2 distinct in S_list \ {a}
    for i, a in enumerate(S_list):
        if n < 3:
            continue
        for j in range(n):
            if j == i:
                continue
            sj = S_list[j]
            for k in range(j+1, n):
                if k == i:
                    continue
                if (x + sj + S_list[k]) % a == 0:
                    return False
    
    return True

def count_violations(S_list, x):
    """Count how many violations x would cause if added to S_list."""
    n = len(S_list)
    violations = 0
    
    # Check 1: x divides sum of 3 distinct in S_list
    for i in range(n):
        si = S_list[i]
        for j in range(i+1, n):
            sj = S_list[j]
            for k in range(j+1, n):
                if (si + sj + S_list[k]) % x == 0:
                    violations += 1
    
    # Check 2: a in S_list divides sum of x and 2 distinct in S_list \ {a}
    for i, a in enumerate(S_list):
        if n < 3:
            continue
        for j in range(n):
            if j == i:
                continue
            sj = S_list[j]
            for k in range(j+1, n):
                if k == i:
                    continue
                if (x + sj + S_list[k]) % a == 0:
                    violations += 1
    
    return violations

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_set = set()
    best_size = 0
    
    numbers = list(range(1, 193))
    
    while time.time() - START_TIME < TIME_LIMIT:
        current_set = set()
        current_list = []
        
        # Phase 1: Greedy add large numbers first (they're less likely to divide sums)
        large_numbers = [x for x in numbers if x > 96]
        random.shuffle(large_numbers)
        for x in large_numbers:
            if can_add_fast(current_list, x, None):
                current_set.add(x)
                current_list.append(x)
        
        # Phase 2: Greedy add smaller numbers
        small_numbers = [x for x in numbers if x <= 96 and x not in current_set]
        random.shuffle(small_numbers)
        for x in small_numbers:
            if can_add_fast(current_list, x, None):
                current_set.add(x)
                current_list.append(x)
        
        # Phase 3: Repair - remove elements that cause the most violations
        improved = True
        while improved and time.time() - START_TIME < TIME_LIMIT:
            improved = False
            if not current_list:
                break
            
            # Find the element whose removal allows the most additions
            best_removal = -1
            best_gain = 0
            best_temp_list = None
            
            for i, removed in enumerate(current_list):
                if time.time() - START_TIME >= TIME_LIMIT:
                    break
                
                temp_list = [x for x in current_list if x != removed]
                temp_set = set(temp_list)
                
                # Try to add back elements
                remaining = [x for x in numbers if x not in temp_set]
                random.shuffle(remaining)
                
                added_count = 0
                for x in remaining:
                    if can_add_fast(temp_list, x, None):
                        temp_list.append(x)
                        temp_set.add(x)
                        added_count += 1
                        if added_count > best_gain:
                            best_gain = added_count
                            best_removal = i
                            best_temp_list = list(temp_list)
                        if added_count >= 2:  # Net gain of at least 1
                            break
                
                # Check if this removal gives net positive gain
                if added_count > 0:  # We removed 1, added added_count, net = added_count - 1
                    if added_count - 1 > best_gain - 1:  # Compare net gains
                        best_gain = added_count
                        best_removal = i
                        best_temp_list = list(temp_list)
            
            if best_removal >= 0 and best_gain >= 2:
                current_list = best_temp_list
                current_set = set(current_list)
                improved = True
        
        # Phase 4: Local search - try swaps
        improved = True
        while improved and time.time() - START_TIME < TIME_LIMIT:
            improved = False
            for i in range(len(current_list)):
                if time.time() - START_TIME >= TIME_LIMIT:
                    break
                removed = current_list[i]
                temp_list = [x for x in current_list if x != removed]
                temp_set = set(temp_list)
                
                remaining = [x for x in numbers if x not in temp_set]
                random.shuffle(remaining)
                
                added_count = 0
                for x in remaining:
                    if can_add_fast(temp_list, x, None):
                        temp_list.append(x)
                        temp_set.add(x)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 2:
                    current_list = temp_list
                    current_set = temp_set
                    improved = True
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    return list(best_set)
