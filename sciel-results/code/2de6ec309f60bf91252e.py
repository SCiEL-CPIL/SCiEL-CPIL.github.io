import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(current_sums, new_sums):
        for ns in new_sums:
            for os in current_sums:
                if ns == os: continue
                if os > ns:
                    if os % ns == 0: return False
                else:
                    if ns % os == 0: return False
        new_list = list(new_sums)
        for i in range(len(new_list)):
            a = new_list[i]
            for j in range(i+1, len(new_list)):
                b = new_list[j]
                if a > b: a, b = b, a
                if b % a == 0: return False
        return True

    def get_sums(elements):
        sums = set()
        for mask in range(1, 1 << len(elements)):
            s = 0
            for i in range(len(elements)):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def try_build():
        current_set = []
        current_sums = set()
        
        # Try mixing ranges?
        # Start with large, add small?
        # Let's just shuffle all 1..256
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            new_sums = {x}
            for s in current_sums:
                new_sums.add(s + x)
            
            if is_valid_addition(current_sums, new_sums):
                current_set.append(x)
                current_sums.update(new_sums)
        
        return current_set

    # Run search
    while time.time() < deadline:
        result = try_build()
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
        # If we have a good set, try to improve it by adding more?
        # The greedy already tries all candidates in random order.
        # So it's already maximal for that random order.
        
    # Final polish: try to add any missing number to best_set
    if best_set:
        best_sums = get_sums(best_set)
        existing = set(best_set)
        for x in range(1, 257):
            if x in existing: continue
            if time.time() > deadline: break
            new_sums = {x}
            for s in best_sums:
                new_sums.add(s + x)
            if is_valid_addition(best_sums, new_sums):
                best_set.append(x)
                best_sums.update(new_sums)
                existing.add(x)
                
    return best_set
