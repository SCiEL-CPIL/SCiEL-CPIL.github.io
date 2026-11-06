import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute LCM table for O(1) lookup
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            lcm_table[a][b] = _lcm(a, b)
            lcm_table[b][a] = lcm_table[a][b]
    
    # Find all bad triples: (a, b, c) with a < b < c and lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    bad_triples.append((a, b, c))
    
    # Build conflict structures using bitmasks
    # For each element x, conflict_pairs[x] = set of (u, v) pairs such that (x, u, v) is a bad triple
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
        
    # For each x, build conflict_dict[x][u] = bitmask of v such that (u,v) in conflict_pairs[x]
    conflict_dict = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v in conflict_pairs[x]:
            if u not in conflict_dict[x]:
                conflict_dict[x][u] = 0
            conflict_dict[x][u] |= (1 << (v - 1))
            if v not in conflict_dict[x]:
                conflict_dict[x][v] = 0
            conflict_dict[x][v] |= (1 << (u - 1))
    
    # Fast can_add using conflict_dict
    conflict_list = [list() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v_mask in conflict_dict[x].items():
            conflict_list[x].append((u, v_mask))

    def can_add(x, mask):
        for u, v_mask in conflict_list[x]:
            if (mask & (1 << (u - 1))) and (mask & v_mask):
                return False
        return True

    def popcount(x):
        return bin(x).count('1')

    def greedy(order, mask=0):
        selected = mask
        for x in order:
            if not (selected & (1 << (x - 1))):
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def try_add_two(mask_no_x, exclude_set):
        new_mask = mask_no_x
        for y in range(1, n + 1):
            if y in exclude_set or (new_mask & (1 << (y - 1))):
                continue
            if not can_add(y, new_mask):
                continue
            new_mask_y = new_mask | (1 << (y - 1))
            for z in range(y + 1, n + 1):
                if z in exclude_set or (new_mask_y & (1 << (z - 1))):
                    continue
                if can_add(z, new_mask_y):
                    return (new_mask_y | (1 << (z - 1)), 2)
        for y in range(1, n + 1):
            if y in exclude_set or (new_mask & (1 << (y - 1))):
                continue
            if can_add(y, new_mask):
                return (new_mask | (1 << (y - 1)), 1)
        return (new_mask, 0)

    def local_search(mask, max_iters=50):
        improved = True
        iters = 0
        
        while improved and time.time() < deadline - 2 and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add any missing element
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
            if improved:
                continue

            # 2. Try 1-for-2 swap: remove x, add y, z
            best_new_mask = mask
            best_size = popcount(mask)
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]

            for x in s_list:
                if time.time() > deadline - 2:
                    break
                mask_no_x = mask & ~(1 << (x - 1))
                new_mask, added_count = try_add_two(mask_no_x, {x})
                if added_count >= 2:
                    sz = popcount(new_mask)
                    if sz > best_size:
                        best_size = sz
                        best_new_mask = new_mask

            if best_size > popcount(mask):
                mask = best_new_mask
                improved = True
                continue

            # 3. Try 2-for-3 swap
            if time.time() < deadline - 3:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list)
                best_new_mask = mask
                best_size = popcount(mask)
                for i in range(min(len(s_list), 20)):
                    if time.time() > deadline - 3:
                        break
                    x = s_list[i]
                    for j in range(i + 1, min(len(s_list), i + 5)):
                        y = s_list[j]
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        new_mask = mask_no_xy
                        added_count = 0
                        for z in range(1, n + 1):
                            if z == x or z == y:
                                continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            sz = popcount(new_mask)
                            if sz > best_size:
                                best_size = sz
                                best_new_mask = new_mask
                if best_size > popcount(mask):
                    mask = best_new_mask
                    improved = True
                    continue

            # 4. Random perturbation if stuck
            if not improved and iters > 5:
                if len(s_list) > 4:
                    remove_idx = random.sample(range(len(s_list)), 2)
                    perturb_mask = mask
                    for idx in remove_idx:
                        perturb_mask &= ~(1 << (s_list[idx] - 1))
                    for x in range(1, n + 1):
                        if not (perturb_mask & (1 << (x - 1))):
                            if can_add(x, perturb_mask):
                                perturb_mask |= (1 << (x - 1))
                    if popcount(perturb_mask) > popcount(mask):
                        mask = perturb_mask
                        improved = True
                        continue

            if not improved:
                break
                
        return mask

    # Precompute degrees for sorting
    degrees = [len(conflict_pairs[k]) for k in range(n + 1)]

    # Branch and bound for maximum independent set in the 3-uniform hypergraph
    # We use a recursive approach with strong pruning
    def branch_and_bound(candidates, current_mask, current_size, best_size_ref):
        """
        candidates: bitmask of elements that could potentially be added
        current_mask: bitmask of elements already selected
        current_size: number of elements in current_mask
        best_size_ref: list containing [best_size] for mutable reference
        """
        if not candidates:
            return current_mask, current_size
        
        # Upper bound: current_size + popcount(candidates)
        if current_size + popcount(candidates) <= best_size_ref[0]:
            return current_mask, current_size
        
        # Try to add all possible elements greedily to get a lower bound
        temp_mask = current_mask
        temp_size = current_size
        temp_candidates = candidates
        while temp_candidates:
            # Pick the element with lowest index for determinism in this sub-problem
            x = (temp_candidates & -temp_candidates).bit_length()
            if can_add(x, temp_mask):
                temp_mask |= (1 << (x - 1))
                temp_size += 1
            temp_candidates &= ~(1 << (x - 1))
        
        if temp_size > best_size_ref[0]:
            best_size_ref[0] = temp_size
            # Update global best
            if temp_size > best_mask_ref[0]:
                best_mask_ref[0] = temp_mask
        
        # Now do branch and bound
        # Pick an element to branch on
        # Heuristic: pick element with fewest conflicts (easiest to include)
        best_x = -1
        best_conflict_count = float('inf')
        temp_c = candidates
        while temp_c:
            x = (temp_c & -temp_c).bit_length()
            # Count how many other candidates conflict with x
            conflict_count = 0
            temp_c2 = candidates
            while temp_c2:
                y = (temp_c2 & -temp_c2).bit_length()
                if y != x and not can_add(y, current_mask | (1 << (x - 1))):
                    conflict_count += 1
                temp_c2 &= ~(1 << (y - 1))
            if conflict_count < best_conflict_count:
                best_conflict_count = conflict_count
                best_x = x
            temp_c &= ~(1 << (x - 1))
        
        if best_x == -1:
            return current_mask, current_size
        
        x = best_x
        bit_x = 1 << (x - 1)
        
        # Branch 1: include x
        if can_add(x, current_mask):
            new_mask = current_mask | bit_x
            new_size = current_size + 1
            # Remove x and any elements that conflict with x from candidates
            new_candidates = candidates & ~bit_x
            temp_c = new_candidates
            while temp_c:
                y = (temp_c & -temp_c).bit_length()
                if not can_add(y, new_mask):
                    new_candidates &= ~(1 << (y - 1))
                temp_c &= ~(1 << (y - 1))
            
            result_mask, result_size = branch_and_bound(new_candidates, new_mask, new_size, best_size_ref)
            if result_size > best_size_ref[0]:
                best_size_ref[0] = result_size
                if result_size > best_mask_ref[0]:
                    best_mask_ref[0] = result_mask
        
        # Branch 2: exclude x
        new_candidates = candidates & ~bit_x
        result_mask, result_size = branch_and_bound(new_candidates, current_mask, current_size, best_size_ref)
        if result_size > best_size_ref[0]:
            best_size_ref[0] = result_size
            if result_size > best_mask_ref[0]:
                best_mask_ref[0] = result_mask
        
        return best_mask_ref[0], best_size_ref[0]

    best_mask = 0
    best_size = 0

    # Run multiple randomized greedy passes with diverse orderings
    num_restarts = 800

    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break

        strategy = i % 8
        if strategy == 0:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random() * 0.5)
        elif strategy == 1:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random() * 0.5)
        elif strategy == 2:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 3:
            weights = [1.0 / (1 + degrees[x]) for x in range(1, n + 1)]
            order = list(range(1, n + 1))
            random.shuffle(order)
            order.sort(key=lambda x: random.random() / weights[x - 1])
        elif strategy == 4:
            partial = random.sample(range(1, n + 1), 30)
            rest = [x for x in range(1, n + 1) if x not in partial]
            random.shuffle(rest)
            order = partial + rest
        elif strategy == 5:
            def is_prime(m):
                if m < 2: return False
                for p in range(2, int(m**0.5) + 1):
                    if m % p == 0: return False
                return True
            order = sorted(range(1, n + 1), key=lambda x: (not is_prime(x), degrees[x]))
        elif strategy == 6:
            def is_pow2(m):
                return (m & (m - 1)) == 0 and m > 0
            order = sorted(range(1, n + 1), key=lambda x: (not is_pow2(x), degrees[x]))
        else:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] * random.random())

        mask = greedy(order)
        mask = local_search(mask, max_iters=30)

        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Try branch and bound on the best solution found
    if time.time() < deadline - 10:
        # Use the best mask as a starting point
        best_mask_ref = [best_mask]
        best_size_ref = [best_size]
        
        # Candidates are all elements not in best_mask
        candidates = 0
        for x in range(1, n + 1):
            if not (best_mask & (1 << (x - 1))):
                candidates |= (1 << (x - 1))
        
        # Only attempt B&B if the candidate set is small enough
        if popcount(candidates) <= 60:
            try:
                result_mask, result_size = branch_and_bound(candidates, best_mask, best_size, best_size_ref)
                if result_size > best_size:
                    best_size = result_size
                    best_mask = result_mask
            except Exception:
                pass

    # Final polish with more iterations
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask, max_iters=100)
    
    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
    
    if not result:
        result = [1]
        
    return result
