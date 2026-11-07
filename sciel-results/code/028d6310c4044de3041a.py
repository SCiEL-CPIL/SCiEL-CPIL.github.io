import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    MAX_SUM = 7680  # 30 * 256
    
    # Precompute divisibility conflict masks
    # forbidden[s] is a bitset where bit t is set if s and t are divisibility-related
    # (s divides t or t divides s, s != t)
    forbidden = [0] * (MAX_SUM + 1)
    for a in range(1, MAX_SUM + 1):
        for b in range(2 * a, MAX_SUM + 1, a):
            forbidden[a] |= (1 << b)
            forbidden[b] |= (1 << a)
    
    def subset_sums_bitset(elements):
        """Return bitset of all nonempty subset sums."""
        n = len(elements)
        sums = 0
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums |= (1 << s)
        return sums
    
    def is_valid_bitset(sums_bitset):
        """Check no two distinct sums divide one another."""
        temp = sums_bitset
        while temp:
            lowbit = temp & (-temp)
            s = lowbit.bit_length() - 1
            if forbidden[s] & sums_bitset:
                return False
            temp ^= lowbit
        return True
    
    def conflict_count(sums_bitset, candidate):
        """Count how many existing sums conflict with new sums if we add candidate."""
        shifted = sums_bitset << candidate
        new_sums = sums_bitset | shifted | (1 << candidate)
        # Count conflicts: for each bit in new_sums, check if forbidden[bit] intersects new_sums
        # But we need to count pairs, so let's count how many new bits have conflicts
        temp = new_sums
        count = 0
        while temp:
            lowbit = temp & (-temp)
            s = lowbit.bit_length() - 1
            if forbidden[s] & new_sums:
                count += 1
            temp ^= lowbit
        return count
    
    def try_add(elements, sums_bitset, x):
        """Try to add x. Returns (new_sums, True) if valid, else (None, False)."""
        if x in elements:
            return None, False
        shifted = sums_bitset << x
        new_sums = sums_bitset | shifted | (1 << x)
        if not is_valid_bitset(new_sums):
            return None, False
        return new_sums, True
    
    def greedy_grow(elements, candidates, sums_bitset, deadline, max_iter=200):
        """Greedily add candidates, preferring those with minimal conflict."""
        elements = list(elements)
        sums = sums_bitset
        iters = 0
        while iters < max_iter and time.time() < deadline:
            iters += 1
            best_x = None
            best_conflict = float('inf')
            best_sums = None
            
            for x in candidates:
                if x in elements:
                    continue
                ns, ok = try_add(elements, sums, x)
                if ok:
                    # All valid additions have 0 conflicts by definition
                    # But we want to prefer "safer" additions
                    # Use a heuristic: prefer larger numbers (fewer future conflicts)
                    if best_x is None or x > best_x:
                        best_x = x
                        best_sums = ns
                        break  # Take first valid in order, or we could score them
            
            if best_x is not None:
                elements.append(best_x)
                sums = best_sums
            else:
                break
        
        return elements, sums
    
    def local_search(current, sums_bitset, deadline, max_iter=500):
        """Local search with add and swap moves."""
        current = list(current)
        sums = sums_bitset
        all_nums = list(range(1, 257))
        best = list(current)
        best_size = len(current)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            used = set(current)
            candidates = [x for x in all_nums if x not in used]
            random.shuffle(candidates)
            
            # Try to add
            for c in candidates:
                ns, ok = try_add(current, sums, c)
                if ok:
                    current.append(c)
                    sums = ns
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = list(current)
                    break
            
            if not improved:
                # Try swap: remove one, add another
                if len(current) >= 2:
                    remove_idx = random.randint(0, len(current) - 1)
                    removed = current.pop(remove_idx)
                    new_sums = subset_sums_bitset(current)
                    
                    used = set(current)
                    candidates = [x for x in all_nums if x not in used]
                    random.shuffle(candidates)
                    
                    added_any = False
                    for c in candidates:
                        ns, ok = try_add(current, new_sums, c)
                        if ok:
                            current.append(c)
                            new_sums = ns
                            added_any = True
                            if len(current) > best_size:
                                best_size = len(current)
                                best = list(current)
                            break
                    
                    if not added_any:
                        current.append(removed)
                        new_sums = subset_sums_bitset(current)
                    else:
                        sums = new_sums
        
        return best, best_size
    
    # Start with a known valid set
    best_result = [31, 32, 34]
    best_size = 3
    
    all_nums = list(range(1, 257))
    
    # Strategy 1: Greedy from various starting points
    for start_val in range(200, 257, 2):
        if time.time() > deadline - 20:
            break
        start = [start_val]
        sums = subset_sums_bitset(start)
        nearby = sorted(all_nums, key=lambda x: abs(x - start_val))
        nearby = [x for x in nearby if x != start_val]
        result, _ = greedy_grow(start, nearby, sums, deadline - 10)
        if len(result) > best_size:
            best_size = len(result)
            best_result = list(result)
    
    # Strategy 2: Randomized restarts
    for trial in range(300):
        if time.time() > deadline - 15:
            break
        
        k = random.randint(1, 3)
        start = random.sample(all_nums, k)
        sums = subset_sums_bitset(start)
        
        remaining = [x for x in all_nums if x not in start]
        random.shuffle(remaining)
        
        result, _ = greedy_grow(start, remaining, sums, deadline - 10)
        if len(result) > best_size:
            best_size = len(result)
            best_result = list(result)
    
    # Strategy 3: Local search on best found
    if len(best_result) >= 2 and time.time() < deadline - 5:
        current_sums = subset_sums_bitset(best_result)
        improved, improved_size = local_search(best_result, current_sums, deadline)
        if improved_size > best_size:
            best_size = improved_size
            best_result = improved
    
    # Final validation
    n = len(best_result)
    if n == 0:
        return [31, 32, 34]
    
    # Check distinctness
    if len(best_result) != len(set(best_result)):
        return [31, 32, 34]
    
    for x in best_result:
        if not isinstance(x, int) or isinstance(x, bool):
            return [31, 32, 34]
        if not (1 <= x <= 256):
            return [31, 32, 34]
    
    # Check subset sums
    sums_set = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += best_result[i]
        if s in sums_set:
            return [31, 32, 34]
        sums_set.add(s)
    
    # Check divisibility
    sums_list = sorted(sums_set)
    for i in range(len(sums_list)):
        a = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            b = sums_list[j]
            if b % a == 0:
                return [31, 32, 34]
    
    return list(best_result)
