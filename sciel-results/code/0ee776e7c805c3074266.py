import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s margin
    
    # Max possible sum for 60 elements of max 256
    MAX_N = 60
    MAX_VAL = 256
    MAX_SUM = MAX_N * MAX_VAL
    
    # Helper to count conflicts (number of sum values shared by different cardinalities)
    # bitsets is a list of ints, bitsets[k] has bit s set if sum s is possible with size k
    def count_conflicts(bitsets):
        conflicts = 0
        n = len(bitsets)
        for i in range(n):
            if bitsets[i] == 0:
                continue
            for j in range(i + 1, n):
                if bitsets[j] == 0:
                    continue
                # Bitwise AND and count set bits
                # Python int.bit_count() is available in 3.10+
                # Fallback: bin(x).count('1')
                try:
                    conflicts += (bitsets[i] & bitsets[j]).bit_count()
                except AttributeError:
                    conflicts += bin(bitsets[i] & bitsets[j]).count('1')
        return conflicts

    # Update bitsets by adding an element x
    # bitsets is list of ints. bitsets[k] corresponds to size k.
    # We need to update in reverse order of k to avoid using updated values
    def add_element(bitsets, x):
        # bitsets[0] is always 1 (sum 0)
        # New bitsets will have length len(bitsets) + 1
        new_bitsets = [0] * (len(bitsets) + 1)
        new_bitsets[0] = 1
        for k in range(1, len(bitsets) + 1):
            # Sums of size k in new set come from:
            # 1. Sums of size k in old set (not using x)
            # 2. Sums of size k-1 in old set, shifted by x (using x)
            new_bitsets[k] = bitsets[k-1] | ((bitsets[k-1] if k-1 < len(bitsets) else 0) << x)
            # Wait, bitsets[k-1] is the old set of size k-1.
            # The term "not using x" is the old bitsets[k].
            # The term "using x" is old bitsets[k-1] << x.
            # So new_bitsets[k] = old_bitsets[k] | (old_bitsets[k-1] << x)
            
            # Correction:
            # new_bitsets[k] = (bitsets[k] if k < len(bitsets) else 0) | ((bitsets[k-1] if k-1 < len(bitsets) else 0) << x)
            
            # Let's rewrite cleanly
        new_bitsets = [0] * (len(bitsets) + 1)
        new_bitsets[0] = 1
        for k in range(1, len(bitsets) + 1):
            old_k = bitsets[k] if k < len(bitsets) else 0
            old_k_minus_1 = bitsets[k-1] if k-1 < len(bitsets) else 0
            new_bitsets[k] = old_k | (old_k_minus_1 << x)
        return new_bitsets

    # Remove an element x from the set represented by bitsets?
    # This is hard because bitsets don't tell us which elements formed the sums.
    # So we must recalculate from scratch if we remove.
    # Recalculation cost is O(N * MAX_SUM) with bitset shifts?
    # No, recalculation from scratch for a set C of size N:
    # Start with bitsets = [1, 0, ...]
    # For each x in C: bitsets = add_element(bitsets, x)
    # This is O(N^2 * (cost of shift/or)).
    # N=60. 3600 operations. Very fast.
    
    def compute_bitsets_from_set(C):
        bitsets = [1]  # size 0
        for x in C:
            bitsets = add_element(bitsets, x)
        return bitsets

    # Main search loop
    best_size = 0
    best_set = []
    
    # Initialize with a random set
    # Start with a small valid set if possible, or just random
    # Let's start with 20 random elements
    current_set = set(random.sample(range(1, 257), 20))
    
    # We will maintain the set as a list for easy iteration and removal
    current_list = list(current_set)
    
    # Initial bitsets
    bitsets = compute_bitsets_from_set(current_list)
    conflicts = count_conflicts(bitsets)
    
    # Track best
    if conflicts == 0:
        best_size = len(current_list)
        best_set = list(current_list)
        
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() < deadline:
        if conflicts == 0:
            # Try to expand
            # Pick a random unused element
            unused = [i for i in range(1, 257) if i not in current_set]
            if not unused:
                break
            x = random.choice(unused)
            
            # Test addition
            test_bitsets = add_element(bitsets, x)
            test_conflicts = count_conflicts(test_bitsets)
            
            if test_conflicts == 0:
                # Accept
                current_list.append(x)
                current_set.add(x)
                bitsets = test_bitsets
                conflicts = 0
                no_improvement_count = 0
                
                if len(current_list) > best_size:
                    best_size = len(current_list)
                    best_set = list(current_list)
            else:
                # Reject
                no_improvement_count += 1
                if no_improvement_count > 100:
                    # Try a swap
                    # Remove random element, add random other
                    if current_list:
                        idx = random.randint(0, len(current_list) - 1)
                        old_x = current_list[idx]
                        # Remove old_x: recompute bitsets
                        # To save time, we can just recompute from scratch
                        current_list.pop(idx)
                        current_set.remove(old_x)
                        
                        # Try to add a new one
                        unused = [i for i in range(1, 257) if i not in current_set]
                        if unused:
                            new_x = random.choice(unused)
                            test_list = current_list + [new_x]
                            test_bitsets = compute_bitsets_from_set(test_list)
                            test_conflicts = count_conflicts(test_bitsets)
                            
                            if test_conflicts == 0:
                                current_list = test_list
                                current_set.add(new_x)
                                bitsets = test_bitsets
                                conflicts = 0
                                no_improvement_count = 0
                                if len(current_list) > best_size:
                                    best_size = len(current_list)
                                    best_set = list(current_list)
                            else:
                                # Keep the set without old_x? Or put it back?
                                # If we keep it without, size decreases.
                                # Let's put it back if we can't improve.
                                current_list.append(old_x)
                                current_set.add(old_x)
                                bitsets = compute_bitsets_from_set(current_list)
                                conflicts = count_conflicts(bitsets)
                                no_improvement_count += 1
                        else:
                            # No unused, put back
                            current_list.append(old_x)
                            current_set.add(old_x)
                            bitsets = compute_bitsets_from_set(current_list)
                            conflicts = count_conflicts(bitsets)
                            no_improvement_count += 1
        else:
            # Conflicts > 0, try to reduce
            # Try removing a random element
            idx = random.randint(0, len(current_list) - 1)
            old_x = current_list[idx]
            
            # Compute bitsets without old_x
            # Reconstruct list without old_x
            test_list = current_list[:idx] + current_list[idx+1:]
            test_bitsets = compute_bitsets_from_set(test_list)
            test_conflicts = count_conflicts(test_bitsets)
            
            if test_conflicts < conflicts:
                # Accept removal
                current_list = test_list
                current_set.remove(old_x)
                bitsets = test_bitsets
                conflicts = test_conflicts
                no_improvement_count = 0
                
                # Update best if valid
                if conflicts == 0 and len(current_list) > best_size:
                    best_size = len(current_list)
                    best_set = list(current_list)
            else:
                # Reject
                no_improvement_count += 1
                
                # If stuck, do a random restart or random swap
                if no_improvement_count > 50:
                    # Random Swap
                    if current_list:
                        idx = random.randint(0, len(current_list) - 1)
                        old_x = current_list[idx]
                        unused = [i for i in range(1, 257) if i not in current_set]
                        if unused:
                            new_x = random.choice(unused)
                            current_list[idx] = new_x
                            current_set.remove(old_x)
                            current_set.add(new_x)
                            bitsets = compute_bitsets_from_set(current_list)
                            conflicts = count_conflicts(bitsets)
                            no_improvement_count = 0
                            
                            if conflicts == 0 and len(current_list) > best_size:
                                best_size = len(current_list)
                                best_set = list(current_list)
                        else:
                            no_improvement_count = 0
                    else:
                        # Set is empty, restart
                        current_list = random.sample(range(1, 257), 10)
                        current_set = set(current_list)
                        bitsets = compute_bitsets_from_set(current_list)
                        conflicts = count_conflicts(bitsets)
                        no_improvement_count = 0

    # Final check and return
    # Ensure types are correct
    # The task requires a list of ints
    if not best_set:
        # Fallback: return a small valid set
        # 1, 2, 4... is valid
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set

if __name__ == "__main__":
    # Test
    s = construct()
    print(len(s))
    print(s)
