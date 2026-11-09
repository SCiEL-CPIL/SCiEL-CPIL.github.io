import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    def check_divisibility(sums_set, new_sum):
        # Check if new_sum divides any sum in sums_set or vice versa
        for s in sums_set:
            if s == new_sum:
                return False # Duplicate
            if new_sum > s:
                if new_sum % s == 0:
                    return False
            else:
                if s % new_sum == 0:
                    return False
        return True

    def try_add(sums_set, elements, x):
        # sums_set is the set of current subset sums
        # elements is the list of current elements (not strictly needed if we have sums_set)
        
        # New sums will be:
        # 1. x
        # 2. s + x for all s in sums_set
        
        # First, check if x is already in sums_set
        if x in sums_set:
            return None, None
            
        # Generate new sums
        new_sums = set()
        new_sums.add(x)
        
        for s in sums_set:
            ns = s + x
            if ns in sums_set:
                return None, None
            if ns in new_sums:
                return None, None
            new_sums.add(ns)
            
        # Check divisibility
        # We need to check all pairs in (sums_set U new_sums)
        # But we can be smarter.
        # Any new sum must not divide any old sum, and any old sum must not divide any new sum.
        # And new sums must not divide each other.
        
        # 1. New sums vs Old sums
        for ns in new_sums:
            if not check_divisibility(sums_set, ns):
                return None, None
                
        # 2. New sums vs New sums
        new_sums_list = list(new_sums)
        for i in range(len(new_sums_list)):
            a = new_sums_list[i]
            for j in range(i + 1, len(new_sums_list)):
                b = new_sums_list[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    return None, None
                    
        # If we get here, it's valid
        # Return updated sums_set
        updated_sums = sums_set.union(new_sums)
        return updated_sums, elements + [x]

    best_set = []
    best_size = 0
    
    candidates = list(range(1, 257))
    
    while time.time() < deadline:
        random.shuffle(candidates)
        
        # Try descending order first?
        # Let's just use the shuffled order.
        current_sums = set()
        current_elements = []
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            new_sums, new_elements = try_add(current_sums, current_elements, x)
            if new_elements is not None:
                current_sums = new_sums
                current_elements = new_elements
                
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best_set = current_elements[:]
            
    return [int(x) for x in best_set]
