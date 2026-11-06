import time
import random

def construct():
    """
    Find a large subset of [1,256] such that for every distinct a,b in the subset,
    (a*b) % (a+b) != 0.
    """
    start_time = time.time()
    deadline = start_time + 70.0  # Safety margin
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict graph using bitmasks for fast operations
    # conflict_mask[x] is a bitmask where bit y is set if x and y conflict
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute conflict count for each element against all others
    # conflict_count[x][y] = 1 if conflict, 0 otherwise (already in mask)
    
    def is_valid_set(S):
        """Check if set S is valid (no conflicts)."""
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if conflict_mask[a] & (1 << b):
                    return False
        return True
    
    def count_conflicts(S, x):
        """Count how many elements in S conflict with x using bitmask."""
        # Build bitmask of S
        mask = 0
        for y in S:
            mask |= (1 << y)
        # Count bits in conflict_mask[x] & mask
        return bin(conflict_mask[x] & mask).count('1')
    
    def count_conflicts_mask(x, S_mask):
        """Count conflicts using precomputed mask."""
        return bin(conflict_mask[x] & S_mask).count('1')
    
    def build_mask(S):
        """Build bitmask from set."""
        mask = 0
        for y in S:
            mask |= (1 << y)
        return mask
    
    # Initial set: all odd numbers
    odds = set(range(1, N + 1, 2))
    if not is_valid_set(odds):
        current_set = set()
        for x in sorted(odds):
            if count_conflicts(current_set, x) == 0:
                current_set.add(x)
    else:
        current_set = set(odds)
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Greedy addition of even numbers, sorted by conflict count
    evens = list(range(2, N + 1, 2))
    evens.sort(key=lambda x: count_conflicts(current_set, x))
    for x in evens:
        if count_conflicts(current_set, x) == 0:
            current_set.add(x)
    
    if len(current_set) > best_size:
        best_set = set(current_set)
        best_size = len(best_set)
    
    # Local search with multiple strategies
    random.seed(42)
    unselected = set(all_nums) - current_set
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any unselected element with zero conflicts
        candidates = list(unselected)
        random.shuffle(candidates)
        for x in candidates:
            if count_conflicts(current_set, x) == 0:
                current_set.add(x)
                unselected.remove(x)
                improved = True
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                break
        
        if time.time() > deadline:
            break
        
        # 2. Remove-one-and-greedily-add-many
        s_list = list(current_set)
        random.shuffle(s_list)
        
        for x in s_list:
            if time.time() > deadline:
                break
            
            S_temp = set(current_set)
            S_temp.remove(x)
            S_temp_mask = build_mask(S_temp)
            
            # Sort unselected by conflict count with S_temp
            u_list = list(unselected)
            u_list.sort(key=lambda y: bin(conflict_mask[y] & S_temp_mask).count('1'))
            
            for y in u_list:
                if (conflict_mask[y] & S_temp_mask) == 0:
                    S_temp.add(y)
                    S_temp_mask |= (1 << y)
            
            if len(S_temp) > len(current_set):
                current_set = S_temp
                unselected = set(all_nums) - current_set
                improved = True
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                break
        
        # 3. Try 2-for-1 swaps: remove two elements, try to add three
        if time.time() > deadline:
            break
        s_list2 = list(current_set)
        random.shuffle(s_list2)
        found_swap = False
        for i in range(len(s_list2)):
            if time.time() > deadline:
                break
            x1 = s_list2[i]
            for j in range(i + 1, len(s_list2)):
                if time.time() > deadline:
                    break
                x2 = s_list2[j]
                S_temp = set(current_set)
                S_temp.remove(x1)
                S_temp.remove(x2)
                S_temp_mask = build_mask(S_temp)
                
                u_list = list(unselected)
                u_list.sort(key=lambda y: bin(conflict_mask[y] & S_temp_mask).count('1'))
                
                added = 0
                for y in u_list:
                    if (conflict_mask[y] & S_temp_mask) == 0:
                        S_temp.add(y)
                        S_temp_mask |= (1 << y)
                        added += 1
                
                if len(S_temp) > len(current_set):
                    current_set = S_temp
                    unselected = set(all_nums) - current_set
                    improved = True
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
                    found_swap = True
                    break
            if found_swap or time.time() > deadline:
                break
        if found_swap:
            continue
    
    # Final verification
    if not is_valid_set(best_set):
        best_set = set(range(1, N + 1, 2))
    
    return list(best_set)

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            if (a*b) % (a+b) == 0:
                valid = False
                print(f"Conflict: {a}, {b}")
    print(f"Valid: {valid}")
