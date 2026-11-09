"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def construct():
    N = 128
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Precompute bad_pairs[x]: list of (a, b) such that {a, b, x} is a bad triple.
    # a < b, a != x, b != x
    bad_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x: continue
            for b in range(a + 1, N + 1):
                if b == x: continue
                if gcd_table[a][b] == gcd_table[a][x] == gcd_table[b][x]:
                    bad_pairs[x].append((a, b))

    def can_add(mask, x):
        """Check if x can be added to the set represented by mask."""
        for a, b in bad_pairs[x]:
            if ((mask >> a) & 1) and ((mask >> b) & 1):
                return False
        return True

    def greedy_build(order):
        """Greedy build following given order."""
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    def mask_to_set(mask):
        return [i for i in range(1, N + 1) if (mask >> i) & 1]

    def local_search(mask, time_limit=5.0):
        """Local search: try to add elements, swap elements."""
        start = time.time()
        improved = True
        current_mask = mask
        
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding any element
            for x in range(1, N + 1):
                if not ((current_mask >> x) & 1):
                    if can_add(current_mask, x):
                        current_mask |= (1 << x)
                        improved = True
                        break
            
            # If we can't add, try 1-swap
            if not improved:
                # Iterate over elements in current set
                for u in range(1, N + 1):
                    if not ((current_mask >> u) & 1):
                        continue
                    mask_without_u = current_mask & ~(1 << u)
                    for v in range(1, N + 1):
                        if not ((current_mask >> v) & 1):
                            if can_add(mask_without_u, v):
                                current_mask = mask_without_u | (1 << v)
                                improved = True
                                break
                    if improved:
                        break
            
            # If still no improvement, try 2-swap (only if time allows)
            if not improved and time.time() - start < time_limit - 0.5:
                # Get list of elements in set
                sel_list = [i for i in range(1, N + 1) if (current_mask >> i) & 1]
                for i in range(len(sel_list)):
                    if time.time() - start >= time_limit - 0.5:
                        break
                    u1 = sel_list[i]
                    for j in range(i + 1, len(sel_list)):
                        if time.time() - start >= time_limit - 0.5:
                            break
                        u2 = sel_list[j]
                        mask_without_u1u2 = current_mask & ~(1 << u1) & ~(1 << u2)
                        found = False
                        for v1 in range(1, N + 1):
                            if found or ((current_mask >> v1) & 1):
                                continue
                            if not can_add(mask_without_u1u2, v1):
                                continue
                            mask_with_v1 = mask_without_u1u2 | (1 << v1)
                            for v2 in range(v1 + 1, N + 1):
                                if found or ((current_mask >> v2) & 1):
                                    continue
                                if can_add(mask_with_v1, v2):
                                    current_mask = mask_with_v1 | (1 << v2)
                                    improved = True
                                    found = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        
        return current_mask

    best_mask = 0
    best_size = 0
    
    base_order = list(range(1, N + 1))
    
    # 1. Natural order
    m = greedy_build(base_order)
    m = local_search(m, time_limit=2.0)
    if bin(m).count('1') > best_size:
        best_size = bin(m).count('1')
        best_mask = m

    # 2. Reverse order
    m = greedy_build(base_order[::-1])
    m = local_search(m, time_limit=2.0)
    if bin(m).count('1') > best_size:
        best_size = bin(m).count('1')
        best_mask = m

    # 3. Sorted by number of bad pairs (fewest first)
    sorted_order = sorted(range(1, N + 1), key=lambda x: len(bad_pairs[x]))
    m = greedy_build(sorted_order)
    m = local_search(m, time_limit=2.0)
    if bin(m).count('1') > best_size:
        best_size = bin(m).count('1')
        best_mask = m

    # 4. Random restarts
    no_improve_count = 0
    while time.time() < deadline:
        random.shuffle(base_order)
        m = greedy_build(base_order)
        m = local_search(m, time_limit=1.0)
        size = bin(m).count('1')
        if size > best_size:
            best_size = size
            best_mask = m
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100 and best_size >= 80:
                break

    # Final polish
    for x in range(1, N + 1):
        if not ((best_mask >> x) & 1):
            if can_add(best_mask, x):
                best_mask |= (1 << x)
                best_size += 1

    return mask_to_set(best_mask) if best_size > 0 else [1]

# EVOLVE-BLOCK-END
