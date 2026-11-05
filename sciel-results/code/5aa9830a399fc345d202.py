"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    target = 200
    mask = (1 << (target + 1)) - 1  # bits 0..200
    target_bit = 1 << target
    
    def try_order(order, initial_set=None, initial_reach=None):
        """Greedy: add elements in given order if it doesn't create sum=target."""
        if initial_reach is None:
            reachable = 1  # bit 0 set
        else:
            reachable = initial_reach
        if initial_set is None:
            selected = []
        else:
            selected = list(initial_set)
        
        selected_set = set(selected)
        for x in order:
            if x in selected_set:
                continue
            new_sums = (reachable << x) & mask
            if not (new_sums & target_bit):
                selected.append(x)
                selected_set.add(x)
                reachable |= new_sums
        return selected, reachable
    
    def recompute_reach(S):
        """Compute reachable sums for set S."""
        reachable = 1
        for x in sorted(S):
            reachable |= (reachable << x) & mask
        return reachable
    
    def is_valid(S):
        """Check if set S has no subset summing to target."""
        return not (recompute_reach(S) & target_bit)
    
    def greedy_add(S, reach, candidates):
        """Try to add elements from candidates in order, return new S and reach."""
        S_set = set(S)
        for x in candidates:
            if x in S_set:
                continue
            new_sums = (reach << x) & mask
            if not (new_sums & target_bit):
                S.append(x)
                S_set.add(x)
                reach |= new_sums
        return S, reach
    
    best = []
    best_size = 0
    
    # Generate many diverse orderings
    all_numbers = list(range(1, 101))
    
    # Strategy 1: Start with guaranteed-valid set {67,...,100}
    S0 = list(range(67, 101))
    reach0 = recompute_reach(S0)
    small = list(range(1, 67))
    
    orders1 = [
        small,
        list(reversed(small)),
        small[::2] + small[1::2],
        small[1::2] + small[::2],
    ]
    for _ in range(4):
        tmp = small[:]
        random.shuffle(tmp)
        orders1.append(tmp)
    
    for order in orders1:
        S, reach = try_order(order, S0, reach0)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Strategy 2: Pure greedy from scratch with various orders
    all_orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(100, 50, -1)) + list(range(50, 0, -1)),
        list(range(1, 51)) + list(range(100, 50, -1)),
    ]
    # Alternating from ends
    order5 = []
    lo, hi = 1, 100
    while lo <= hi:
        order5.append(hi)
        if lo != hi:
            order5.append(lo)
        lo += 1
        hi -= 1
    all_orders.append(order5)
    # Middle out
    all_orders.append(list(range(50, 101)) + list(range(49, 0, -1)))
    # More random orders
    for _ in range(6):
        tmp = all_numbers[:]
        random.shuffle(tmp)
        all_orders.append(tmp)
    # Weighted: prefer large numbers first
    for _ in range(4):
        tmp = all_numbers[:]
        random.shuffle(tmp)
        tmp.sort(key=lambda x: -x + random.uniform(-20, 20))
        all_orders.append(tmp)
    
    for order in all_orders:
        S, reach = try_order(order)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Strategy 3: Multi-start local search with multiple restarts
    S = list(best)
    S_set = set(S)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        for y in S_list:
            if time.time() > deadline - 5:
                break
            # Remove y
            S_temp = [x for x in S if x != y]
            S_temp_set = set(S_temp)
            reach_temp = recompute_reach(S_temp)
            # Try to add any valid element
            candidates = [x for x in range(100, 0, -1) if x not in S_temp_set]
            random.shuffle(candidates)
            for z in candidates:
                new_reach = (reach_temp | (reach_temp << z)) & mask
                if not (new_reach & target_bit):
                    S_new = S_temp + [z]
                    S_new_set = S_temp_set | {z}
                    if len(S_new) > len(S):
                        S = S_new
                        S_set = S_new_set
                        if len(S) > best_size:
                            best = list(S)
                            best_size = len(S)
                        improved = True
                        break
            if improved:
                break
    
    # Strategy 4: Two-swap local search (remove 2, add 2)
    S = list(best)
    S_set = set(S)
    reach = recompute_reach(S)
    
    two_swap_improved = True
    while two_swap_improved and time.time() < deadline - 3:
        two_swap_improved = False
        S_list = list(S)
        random.shuffle(S_list)
        for i, y1 in enumerate(S_list):
            if time.time() > deadline - 3:
                break
            for j, y2 in enumerate(S_list):
                if i == j:
                    continue
                if time.time() > deadline - 3:
                    break
                # Remove y1 and y2
                S_temp = [x for x in S if x != y1 and x != y2]
                S_temp_set = set(S_temp)
                reach_temp = recompute_reach(S_temp)
                # Try to add two valid elements
                candidates = [x for x in range(100, 0, -1) if x not in S_temp_set]
                random.shuffle(candidates)
                added = 0
                for z in candidates:
                    if added >= 2:
                        break
                    new_reach = (reach_temp | (reach_temp << z)) & mask
                    if not (new_reach & target_bit):
                        S_temp.append(z)
                        S_temp_set.add(z)
                        reach_temp = new_reach
                        added += 1
                if len(S_temp) > len(S):
                    S = S_temp
                    S_set = S_temp_set
                    reach = reach_temp
                    if len(S) > best_size:
                        best = list(S)
                        best_size = len(S)
                    two_swap_improved = True
                    break
            if two_swap_improved:
                break
    
    # Final pass: try to add any remaining valid elements
    S = list(best)
    reach = recompute_reach(S)
    S_final, reach_final = try_order(list(range(1, 101)), S, reach)
    if len(S_final) > best_size:
        best = S_final
        best_size = len(S_final)
    
    # Verify best is valid
    if not is_valid(best):
        # Fallback to known valid set
        best = list(range(67, 101))
    
    return best
# EVOLVE-BLOCK-END
