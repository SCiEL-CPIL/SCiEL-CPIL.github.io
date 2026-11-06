"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    MASK = (1 << 201) - 1
    TARGET_BIT = 1 << 200
    
    def greedy(order):
        """Greedy: include numbers in given order if it doesn't create sum 200.
        Uses bitset for speed."""
        reachable = 1  # bit 0 set
        selected = []
        for x in order:
            new_bits = (reachable << x) & MASK
            if new_bits & TARGET_BIT:
                continue
            reachable |= new_bits
            selected.append(x)
        return selected
    
    def compute_reachable(lst):
        """Compute reachable bitset for a list."""
        r = 1
        for x in lst:
            r |= (r << x) & MASK
        return r
    
    def local_improve(selected, excluded):
        """Try to add excluded elements or perform swaps to increase cardinality."""
        best = list(selected)
        excluded_list = list(excluded)
        excluded_set = set(excluded_list)
        best_set = set(best)
        reachable = compute_reachable(best)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try adding each excluded element directly
            for i, x in enumerate(excluded_list):
                new_bits = (reachable << x) & MASK
                if not (new_bits & TARGET_BIT):
                    best.append(x)
                    best_set.add(x)
                    excluded_list.pop(i)
                    excluded_set.discard(x)
                    reachable |= new_bits
                    improved = True
                    break
            if improved:
                continue
            
            # Phase 2: Try swap remove-1-add-1 (rearrange for future additions)
            for i, y in enumerate(best):
                # Compute reachable without y using prefix/suffix trick is complex; 
                # for small sets, just recompute
                best_wo_y = [z for z in best if z != y]
                reachable_no_y = compute_reachable(best_wo_y)
                for j, x in enumerate(excluded_list):
                    if x == y:
                        continue
                    new_bits = (reachable_no_y << x) & MASK
                    if not (new_bits & TARGET_BIT):
                        # Check if this swap actually helps: try to add more after
                        # Just do the swap and let next iteration try to add
                        best[i] = x
                        best_set.discard(y)
                        best_set.add(x)
                        excluded_list[j] = y
                        excluded_set.discard(x)
                        excluded_set.add(y)
                        reachable = compute_reachable(best)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # Phase 3: Try swap remove-1-add-2
            for i, y in enumerate(best):
                best_wo_y = [z for z in best if z != y]
                reachable_no_y = compute_reachable(best_wo_y)
                found = False
                for a_idx in range(len(excluded_list)):
                    x1 = excluded_list[a_idx]
                    b1 = (reachable_no_y << x1) & MASK
                    if b1 & TARGET_BIT:
                        continue
                    for b_idx in range(len(excluded_list)):
                        if b_idx == a_idx:
                            continue
                        x2 = excluded_list[b_idx]
                        b2 = (b1 << x2) & MASK
                        if b2 & TARGET_BIT:
                            continue
                        # Perform swap: remove y, add x1, x2
                        best[i] = x1
                        best.append(x2)
                        best_set.discard(y)
                        best_set.update([x1, x2])
                        # Remove x1, x2 from excluded, add y
                        new_excluded = []
                        for k, ex in enumerate(excluded_list):
                            if k == a_idx or k == b_idx:
                                continue
                            new_excluded.append(ex)
                        new_excluded.append(y)
                        excluded_list = new_excluded
                        excluded_set = set(excluded_list)
                        reachable = compute_reachable(best)
                        improved = True
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if improved:
                continue
            
            # Phase 4: Try swap remove-2-add-3
            n_best = len(best)
            for i in range(n_best):
                for j in range(i+1, n_best):
                    y1, y2 = best[i], best[j]
                    best_wo = [z for k, z in enumerate(best) if k != i and k != j]
                    reachable_no = compute_reachable(best_wo)
                    found = False
                    for a in range(len(excluded_list)):
                        x1 = excluded_list[a]
                        b1 = (reachable_no << x1) & MASK
                        if b1 & TARGET_BIT:
                            continue
                        for b in range(len(excluded_list)):
                            if b == a:
                                continue
                            x2 = excluded_list[b]
                            b2 = (b1 << x2) & MASK
                            if b2 & TARGET_BIT:
                                continue
                            for c in range(len(excluded_list)):
                                if c == a or c == b:
                                    continue
                                x3 = excluded_list[c]
                                b3 = (b2 << x3) & MASK
                                if b3 & TARGET_BIT:
                                    continue
                                # Perform swap: remove y1,y2 add x1,x2,x3
                                new_best = best_wo + [x1, x2, x3]
                                best_set.discard(y1)
                                best_set.discard(y2)
                                best_set.update([x1, x2, x3])
                                new_excluded = []
                                for k, ex in enumerate(excluded_list):
                                    if k == a or k == b or k == c:
                                        continue
                                    new_excluded.append(ex)
                                new_excluded.extend([y1, y2])
                                excluded_list = new_excluded
                                excluded_set = set(excluded_list)
                                best = new_best
                                reachable = compute_reachable(best)
                                improved = True
                                found = True
                                break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
        
        return best
    
    # Try multiple orderings
    all_numbers = list(range(1, 101))
    best_result = []
    
    # Descending order
    r = greedy(all_numbers[::-1])
    if len(r) > len(best_result):
        best_result = r
    
    # Ascending order
    r = greedy(all_numbers)
    if len(r) > len(best_result):
        best_result = r
    
    # Modular orderings: group by remainder mod small numbers
    for mod in [2, 3, 4, 5, 7]:
        groups = [[] for _ in range(mod)]
        for x in all_numbers:
            groups[x % mod].append(x)
        # Interleave groups
        order = []
        max_len = max(len(g) for g in groups)
        for i in range(max_len):
            for g in groups:
                if i < len(g):
                    order.append(g[i])
        r = greedy(order)
        if len(r) > len(best_result):
            best_result = r
    
    # Weighted: prefer numbers that are "harder" to use in sum 200
    # Numbers > 100/2 = 50 can't pair with another to make 200 unless... 
    # Actually any x <= 100 can be part of sum 200.
    # Try ordering by x/200 (smaller first) or by complement (100-x) first
    r = greedy(sorted(all_numbers, key=lambda x: abs(x - 50)))
    if len(r) > len(best_result):
        best_result = r
    
    # Random shuffles
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 15:
        order = all_numbers[:]
        rng.shuffle(order)
        r = greedy(order)
        if len(r) > len(best_result):
            best_result = r
            attempts = 0
        else:
            attempts += 1
        if attempts > 100:
            break
    
    # Local improvement
    selected_set = set(best_result)
    excluded = [x for x in all_numbers if x not in selected_set]
    best_result = local_improve(best_result, excluded)
    
    # Try local improvement from other greedy results too
    for order in [all_numbers[::-1], all_numbers]:
        r = greedy(order)
        if len(r) >= len(best_result) - 2:  # close enough to be worth improving
            sel_set = set(r)
            exc = [x for x in all_numbers if x not in sel_set]
            r_improved = local_improve(r, exc)
            if len(r_improved) > len(best_result):
                best_result = r_improved
    
    if not best_result:
        return [100]
    
    return best_result
# EVOLVE-BLOCK-END
