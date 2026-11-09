"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    max_n = 128
    nums = list(range(1, max_n + 1))
    
    # Precompute lcm table
    lcm_table = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    for i in range(1, max_n + 1):
        for j in range(i, max_n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflict pairs
    # conflict_pairs[x] = list of (a, b) such that {a, b, x} is a bad triple
    conflict_pairs = [[] for _ in range(max_n + 1)]
    
    for a in range(1, max_n + 1):
        for b in range(a + 1, max_n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, max_n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Precompute bitmasks for conflict pairs for faster checking
    conflict_pairs_bits = [[] for _ in range(max_n + 1)]
    for x in range(1, max_n + 1):
        for a, b in conflict_pairs[x]:
            conflict_pairs_bits[x].append((1 << (a-1), 1 << (b-1)))
    
    def greedy(order, mask=0):
        for x in order:
            ok = True
            for bit_a, bit_b in conflict_pairs_bits[x]:
                if (mask & bit_a) and (mask & bit_b):
                    ok = False
                    break
            if ok:
                mask |= (1 << (x-1))
        return mask
    
    def mask_to_set(mask):
        return [i+1 for i in range(max_n) if (mask >> i) & 1]
    
    def conflict_count(x):
        return len(conflict_pairs[x])
    
    # Precompute conflict-sorted order (fewest conflicts first)
    conflict_sorted = sorted(range(1, max_n + 1), key=conflict_count)
    
    best_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # 1. Random Restarts with Greedy
    orders = [
        list(range(1, max_n + 1)),
        list(range(max_n, 0, -1)),
        list(conflict_sorted),
        list(reversed(conflict_sorted)),
    ]
    
    for order in orders:
        m = greedy(order)
        size = bin(m).count("1")
        if size > best_size:
            best_size = size
            best_mask = m
    
    # Random shuffles
    for trial in range(20000):
        if time.time() > deadline - 5.0:
            break
        order = list(range(1, max_n + 1))
        rng.shuffle(order)
        m = greedy(order)
        size = bin(m).count("1")
        if size > best_size:
            best_size = size
            best_mask = m
    
    # 2. Local Search (Iterated Local Search) with 1-swap and 2-for-3 swaps
    improved = True
    current_mask = best_mask
    current_size = best_size
    
    def try_add_to_mask(mask, x):
        """Check if x can be added to mask, return new mask or None."""
        for bit_a, bit_b in conflict_pairs_bits[x]:
            if (mask & bit_a) and (mask & bit_b):
                return None
        return mask | (1 << (x-1))
    
    def local_search_1swap(mask, size):
        """Try 1-swap: remove one, add one. Return (new_mask, new_size) or (mask, size)."""
        set_bits = [i+1 for i in range(max_n) if (mask >> i) & 1]
        not_in_set = [x for x in range(1, max_n + 1) if not (mask & (1 << (x-1)))]
        
        for x in set_bits:
            mask_without_x = mask & ~(1 << (x-1))
            for y in not_in_set:
                new_mask = try_add_to_mask(mask_without_x, y)
                if new_mask is not None:
                    # Same size, but different composition
                    # Check if this allows more additions
                    # For now, accept if it's "better" in terms of future potential
                    # We'll just return it as a candidate for further improvement
                    return new_mask, size
        return mask, size
    
    def local_search_2for3(mask, size):
        """Try 2-for-3 swap: remove two, add three. Return (new_mask, new_size) or (mask, size)."""
        set_bits = [i+1 for i in range(max_n) if (mask >> i) & 1]
        not_in_set = [x for x in range(1, max_n + 1) if not (mask & (1 << (x-1)))]
        
        if len(set_bits) < 2 or len(not_in_set) < 3:
            return mask, size
        
        # Try all pairs to remove
        for i in range(len(set_bits)):
            for j in range(i+1, len(set_bits)):
                x, y = set_bits[i], set_bits[j]
                mask_without_xy = mask & ~(1 << (x-1)) & ~(1 << (y-1))
                
                # Try to add 3 elements
                added = []
                for z in not_in_set:
                    if len(added) >= 3:
                        break
                    new_mask = try_add_to_mask(mask_without_xy, z)
                    if new_mask is not None:
                        mask_without_xy = new_mask
                        added.append(z)
                
                if len(added) >= 3:
                    new_size = size + len(added) - 2
                    if new_size > size:
                        return mask_without_xy, new_size
        return mask, size
    
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Phase A: Perturb + greedy repair
        for k in [1, 2, 3, 5, 8]:
            if time.time() > deadline - 2.0:
                break
            
            # Pick k random elements to remove
            set_bits = [i+1 for i in range(max_n) if (current_mask >> i) & 1]
            if len(set_bits) < k:
                continue
                
            removed = rng.sample(set_bits, k)
            new_mask = current_mask
            for x in removed:
                new_mask &= ~(1 << (x-1))
            
            # Greedily add back, prioritizing vertices with fewer conflicts
            candidates = [x for x in range(1, max_n + 1) if not (new_mask & (1 << (x-1)))]
            # Sort by conflict count, then randomize ties
            rng.shuffle(candidates)
            candidates.sort(key=conflict_count)
            
            m = greedy(candidates, new_mask)
            size = bin(m).count("1")
            
            if size > current_size:
                current_size = size
                current_mask = m
                if size > best_size:
                    best_size = size
                    best_mask = m
                improved = True
                break # Break k loop, restart while loop with new current
        
        # Phase B: 2-for-3 swap local search
        new_mask, new_size = local_search_2for3(current_mask, current_size)
        if new_size > current_size:
            current_size = new_size
            current_mask = new_mask
            if size > best_size:
                best_size = size
                best_mask = m
            improved = True
    
    # 3. Branch and Bound with color-based upper bound
    # Build conflict graph adjacency as bitmasks
    # adj[x] = bitmask of vertices that conflict with x (in the sense that they can't both be in the set)
    # Actually, the conflict is: if x is in the set, then for each (a,b) in conflict_pairs[x],
    # we can't have both a and b in the set. This is not a standard graph conflict.
    # Let's reframe: the problem is to find a maximum subset S such that for all x in S,
    # there is no pair (a,b) in conflict_pairs[x] with both a,b in S.
    # This is equivalent to: for each x, the set S \ {x} must not contain any conflict pair of x.
    # This is a 3-uniform hypergraph independence problem, not a standard graph MIS.
    # So color-based bound doesn't directly apply. We'll stick with the safe-count bound but improve it.
    
    def count_safe_vertices(remaining_mask, current_mask):
        count = 0
        for x in range(1, max_n + 1):
            if not (remaining_mask & (1 << (x-1))):
                continue
            safe = True
            for bit_a, bit_b in conflict_pairs_bits[x]:
                if (current_mask & bit_a) and (current_mask & bit_b):
                    safe = False
                    break
            if safe:
                count += 1
        return count
    
    def branch_and_bound(remaining_mask, current_mask, current_size):
        nonlocal best_mask, best_size
        
        if time.time() > deadline - 1.0:
            return
        
        # Upper bound
        safe_count = count_safe_vertices(remaining_mask, current_mask)
        if current_size + safe_count <= best_size:
            return
        
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Find vertices that can be included
        includable = []
        for x in range(1, max_n + 1):
            if not (remaining_mask & (1 << (x-1))):
                continue
            can_include = True
            for bit_a, bit_b in conflict_pairs_bits[x]:
                if (current_mask & bit_a) and (current_mask & bit_b):
                    can_include = False
                    break
            if can_include:
                includable.append(x)
        
        if not includable:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Choose vertex with most conflicts (to prune more)
        x = max(includable, key=conflict_count)
        bit_x = 1 << (x-1)
        
        # Option 1: Include x
        new_current = current_mask | bit_x
        new_remaining = remaining_mask & ~bit_x
        branch_and_bound(new_remaining, new_current, current_size + 1)
        
        # Option 2: Exclude x
        new_remaining = remaining_mask & ~bit_x
        branch_and_bound(new_remaining, current_mask, current_size)
    
    # Run B&B from the current best solution's complement
    remaining_mask = 0
    for x in range(1, max_n + 1):
        if not (best_mask & (1 << (x-1))):
            remaining_mask |= (1 << (x-1))
    
    branch_and_bound(remaining_mask, best_mask, best_size)
    
    # 4. Final polish: try to add any remaining vertices that fit
    for x in range(1, max_n + 1):
        if time.time() > deadline - 0.5:
            break
        if not (best_mask & (1 << (x-1))):
            new_mask = try_add_to_mask(best_mask, x)
            if new_mask is not None:
                best_mask = new_mask
                best_size += 1
    
    return mask_to_set(best_mask)
# EVOLVE-BLOCK-END
