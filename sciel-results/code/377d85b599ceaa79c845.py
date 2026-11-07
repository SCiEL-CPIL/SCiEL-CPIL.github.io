import time
import sys
import random

def _validate_full(elements):
    if not elements:
        return True
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    if len(set(elements)) != len(elements):
        return False
    sums = set([0])
    for x in elements:
        new_sums = set(s + x for s in sums)
        if sums & new_sums:
            return False
        sums |= new_sums
    non_empty_sums = sorted([s for s in sums if s > 0])
    n = len(non_empty_sums)
    if n == 0:
        return True
    max_sum = non_empty_sums[-1]
    for i in range(n):
        a = non_empty_sums[i]
        if a > max_sum // 2:
            break
        for j in range(i + 1, n):
            b = non_empty_sums[j]
            if b % a == 0:
                return False
    return True

def construct():
    deadline = time.time() + 85.0
    MAX_SUM = 4096
    
    # Precompute conflict masks as bitmasks
    # conflict_mask[v] = bitmask where bit u is set if u conflicts with v
    conflict_mask = [0] * (MAX_SUM + 1)
    for a in range(1, MAX_SUM + 1):
        for b in range(2 * a, MAX_SUM + 1, a):
            conflict_mask[a] |= (1 << b)
            conflict_mask[b] |= (1 << a)
    
    best_elements = []
    best_size = 0
    
    def try_build_fast(order, start_elements=None):
        nonlocal deadline
        
        if start_elements is not None:
            current_elements = list(start_elements)
            current_sums = {0}
            for x in current_elements:
                new_sums = set(s + x for s in current_sums)
                if current_sums & new_sums:
                    return current_elements
                current_sums |= new_sums
        else:
            current_elements = []
            current_sums = {0}
        
        sums_bitset = 1  # bit 0 set
        
        for candidate in order:
            if time.time() > deadline:
                break
            
            # Collision check
            shifted = sums_bitset << candidate
            if shifted & sums_bitset:
                continue
            
            # Check max sum
            if (sums_bitset.bit_length() - 1) + candidate > MAX_SUM:
                continue
            
            # Check divisibility: for each new sum, check conflicts with old sums
            # New sums bitmask = shifted
            # For each new sum ns, check if sums_bitset & conflict_mask[ns] != 0
            valid = True
            # Iterate through set bits in shifted
            temp = shifted
            while temp:
                # Get lowest set bit
                lb = temp & (-temp)
                ns = lb.bit_length() - 1
                temp ^= lb
                
                # Check conflict with existing sums
                if sums_bitset & conflict_mask[ns]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check new sums against each other
            # For each new sum ns, check if (shifted & conflict_mask[ns]) has any bit set other than ns itself
            valid = True
            temp = shifted
            while temp:
                lb = temp & (-temp)
                ns = lb.bit_length() - 1
                temp ^= lb
                # Check if any other new sum conflicts with ns
                other_conflicts = shifted & conflict_mask[ns]
                # Remove ns itself from consideration
                other_conflicts &= ~(1 << ns)
                if other_conflicts:
                    valid = False
                    break
            
            if valid:
                current_elements.append(candidate)
                current_sums |= set(s + candidate for s in current_sums)
                sums_bitset |= shifted
        
        return current_elements
    
    # Strategy 1: Descending
    order_desc = list(range(256, 0, -1))
    elems = try_build_fast(order_desc)
    if len(elems) > best_size:
        best_size = len(elems)
        best_elements = list(elems)
    
    # Strategy 2: Randomized restarts
    random.seed(42)
    for restart in range(10000):
        if time.time() > deadline - 5.0:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        large = [x for x in range(128, 257)]
        random.shuffle(large)
        num_large = random.randint(2, 6)
        front = large[:num_large]
        rest = [x for x in order if x not in front]
        order = front + rest
        elems = try_build_fast(order)
        if len(elems) > best_size:
            best_size = len(elems)
            best_elements = list(elems)
    
    # Strategy 3: Local search
    if best_elements and time.time() < deadline - 15.0:
        used = set(best_elements)
        for i in range(len(best_elements)):
            if time.time() > deadline - 5.0:
                break
            for candidate in range(1, 257):
                if candidate in used:
                    continue
                new_elements = best_elements[:i] + [candidate] + best_elements[i+1:]
                if _validate_full(new_elements):
                    new_used = set(new_elements)
                    remaining = [x for x in range(1, 257) if x not in new_used]
                    random.shuffle(remaining)
                    extended = try_build_fast(remaining, start_elements=new_elements)
                    if len(extended) > best_size:
                        best_size = len(extended)
                        best_elements = list(extended)
                        break
    
    if best_elements:
        if not _validate_full(best_elements):
            best_elements = [20, 21, 23]
    return list(best_elements)

def try_build_from(start_elements, remaining_candidates):
    if not start_elements:
        return []
    current_elements = list(start_elements)
    current_sums = {0}
    for x in current_elements:
        new_sums = set(s + x for s in current_sums)
        if current_sums & new_sums:
            return current_elements
        current_sums |= new_sums
    for candidate in remaining_candidates:
        new_sums = set(s + candidate for s in current_sums)
        if current_sums & new_sums:
            continue
        valid = True
        for ns in new_sums:
            if ns == 0:
                continue
            for es in current_sums:
                if es > 0 and es != ns:
                    if es > ns and es % ns == 0:
                        valid = False
                        break
                    if ns > es and ns % es == 0:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            current_elements.append(candidate)
            current_sums |= new_sums
    return current_elements

if __name__ == "__main__":
    print(construct())
