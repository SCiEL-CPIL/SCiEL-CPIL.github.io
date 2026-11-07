import time
import sys
import random

def _validate_full(elements):
    """
    Full strict validation.
    1. All elements distinct, in 1..256, type int.
    2. All non-empty subset sums are unique.
    3. No two distinct non-empty subset sums divide one another.
    """
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
        
    non_empty_sums = [s for s in sums if s > 0]
    non_empty_sums.sort()
    n = len(non_empty_sums)
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
    deadline = time.time() + 80.0
    
    # Precompute divisibility forbidden pairs
    # For each sum value s, precompute a bitset of all t != s where s|t or t|s
    MAX_SUM = 7680  # 30 * 256
    
    # forbidden_mask[s] = bitset where bit t is set if s and t are incompatible (one divides the other, s != t)
    forbidden_mask = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        # s divides t: t = 2s, 3s, ...
        for t in range(2 * s, MAX_SUM + 1, s):
            mask |= (1 << t)
        # t divides s: t = s/2, s/3, ... (integer division)
        for t in range(1, s):
            if s % t == 0:
                mask |= (1 << t)
        forbidden_mask[s] = mask
    
    best_elements = []
    best_size = 0
    
    def backtrack(current_elements, sums_bitset, current_sums_set, candidates, depth):
        """
        Backtracking search.
        sums_bitset: Python int, bit i set iff sum i is achievable
        current_sums_set: set of achievable sums (for iteration)
        candidates: list of candidate values to try (in order)
        """
        nonlocal best_elements, best_size
        
        if time.time() > deadline:
            return
        
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best_elements = list(current_elements)
        
        # Try to add more elements
        for idx, cand in enumerate(candidates):
            if time.time() > deadline:
                return
            
            # Check collision: shifted = sums_bitset << cand
            shifted = sums_bitset << cand
            if shifted & sums_bitset:
                continue  # Collision
            
            # Check divisibility: for each new sum ns = s + cand, check if ns is forbidden with any existing sum
            # New sums are in 'shifted' bitset
            # For each existing sum es, we need: no new sum ns such that forbidden_mask[es] has bit ns set
            # Equivalently: for each existing es, (forbidden_mask[es] & shifted) must be 0
            # But also: new sums among themselves must not be forbidden pairs
            # (forbidden_mask[ns] & shifted) should not have any other new sum... but shifted already excludes 0
            # Actually we need: for each new sum ns, (forbidden_mask[ns] & (sums_bitset | shifted)) & ~(1 << ns) == 0
            # Simplified: check existing sums against new sums, and new sums among themselves
            
            valid = True
            
            # Check: any existing sum es has a forbidden relationship with any new sum?
            # For each es in current_sums_set (es > 0), check (forbidden_mask[es] & shifted) != 0
            for es in current_sums_set:
                if es > 0 and es <= MAX_SUM:
                    if forbidden_mask[es] & shifted:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check: new sums among themselves
            # For each new sum ns (bits in shifted), check if forbidden_mask[ns] has any other bit in shifted set
            # This is trickier. Let's iterate new sums.
            # New sums are the bits in 'shifted'
            # Extract new sums:
            new_sums_list = []
            temp = shifted
            while temp:
                # Get lowest set bit
                lb = temp & (-temp)
                ns = lb.bit_length() - 1
                new_sums_list.append(ns)
                temp ^= lb
            
            # Check pairs among new sums
            for i in range(len(new_sums_list)):
                ns1 = new_sums_list[i]
                if ns1 > MAX_SUM:
                    continue
                # Check against all other new sums
                for j in range(i + 1, len(new_sums_list)):
                    ns2 = new_sums_list[j]
                    if ns2 > MAX_SUM:
                        continue
                    if ns1 < ns2 and ns2 % ns1 == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Also check: new sums vs existing sums in the other direction
            # Already covered by the forbidden_mask[es] & shifted check above (which is symmetric in the sense that
            # forbidden_mask[es] includes all t where es|t or t|es)
            
            # Valid! Add candidate
            current_elements.append(cand)
            new_sums_set = set(new_sums_list)
            current_sums_set_new = current_sums_set | new_sums_set
            sums_bitset_new = sums_bitset | shifted
            
            backtrack(current_elements, sums_bitset_new, current_sums_set_new, candidates[idx+1:], depth + 1)
            
            current_elements.pop()
    
    # Strategy: Use backtracking with candidate ordering
    # Start with large numbers first (they create larger sums, fewer divisibility issues)
    
    # Multiple randomized restarts with different candidate orderings
    random.seed(12345)
    
    # Initial greedy to get a starting point
    def greedy_build(candidate_order):
        current_elements = []
        sums_bitset = 1  # bit 0
        current_sums_set = {0}
        
        for cand in candidate_order:
            if time.time() > deadline:
                break
            shifted = sums_bitset << cand
            if shifted & sums_bitset:
                continue
            
            valid = True
            for es in current_sums_set:
                if es > 0 and es <= MAX_SUM:
                    if forbidden_mask[es] & shifted:
                        valid = False
                        break
            if not valid:
                continue
            
            # Check new sums among themselves
            new_sums_list = []
            temp = shifted
            while temp:
                lb = temp & (-temp)
                ns = lb.bit_length() - 1
                new_sums_list.append(ns)
                temp ^= lb
            
            for i in range(len(new_sums_list)):
                ns1 = new_sums_list[i]
                if ns1 > MAX_SUM:
                    continue
                for j in range(i + 1, len(new_sums_list)):
                    ns2 = new_sums_list[j]
                    if ns2 > MAX_SUM:
                        continue
                    if ns1 < ns2 and ns2 % ns1 == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_elements.append(cand)
                current_sums_set |= set(new_sums_list)
                sums_bitset |= shifted
        
        return current_elements
    
    # Try several candidate orderings
    orderings = []
    
    # Descending
    orderings.append(list(range(256, 0, -1)))
    
    # Randomized with large bias
    for _ in range(20):
        order = list(range(1, 257))
        random.shuffle(order)
        # Bias towards larger numbers
        large = [x for x in order if x > 128]
        small = [x for x in order if x <= 128]
        random.shuffle(large)
        random.shuffle(small)
        orderings.append(large + small)
    
    # Try each ordering with greedy first, then backtracking from the greedy result
    for order in orderings:
        if time.time() > deadline - 10:
            break
        
        # Greedy pass
        greedy_result = greedy_build(order)
        if len(greedy_result) > best_size:
            best_size = len(greedy_result)
            best_elements = list(greedy_result)
        
        # Backtracking: try to extend beyond greedy
        # Rebuild state from greedy result
        if greedy_result:
            sums_bitset = 1
            current_sums_set = {0}
            for x in greedy_result:
                sums_bitset |= (sums_bitset << x)
                current_sums_set |= set(s + x for s in list(current_sums_set))
            
            # Remaining candidates
            used = set(greedy_result)
            remaining = [x for x in order if x not in used]
            
            # Backtrack from here
            backtrack(list(greedy_result), sums_bitset, current_sums_set, remaining, len(greedy_result))
    
    # Final validation
    if best_elements:
        if not _validate_full(best_elements):
            best_elements = [20, 21, 23]
    
    return list(best_elements)

if __name__ == "__main__":
    print(construct())
