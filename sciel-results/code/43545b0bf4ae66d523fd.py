"""Direct construction exploiting divisibility structure of the LCM constraint."""
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)

    # Precompute conflict masks
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    weights = [1.0 / numbers[i] for i in range(n)]

    def reciprocal_sum_from_mask(mask):
        total = 0.0
        for i in range(n):
            if mask & (1 << i):
                total += weights[i]
        return total

    def can_add(i, selected_mask):
        return (conflict_mask[i] & selected_mask) == 0

    # === DIRECT CONSTRUCTION ===
    # Key structural facts:
    # 1. If 2 in S, no other even can be (lcm(2,x)=x<=256 for even x)
    # 2. Odds > 128 are mutually compatible (lcm > 256 always)
    # 3. Evens don't conflict with odds > 128 (lcm(e,o) > 256 for even e>=2, odd o>128)
    # 4. So: include 2 + all odds>128, OR exclude 2 + select compatible evens + all odds>128

    # Build the direct solution: {2} union {odd numbers in (128, 256]}
    selected_mask = 0
    for i in range(n):
        val = numbers[i]
        if val == 2:
            selected_mask |= (1 << i)
        elif val % 2 == 1 and val > 128:
            selected_mask |= (1 << i)
    
    best_mask = selected_mask
    best_sum = reciprocal_sum_from_mask(selected_mask)
    
    # Verify validity
    def is_valid(mask):
        for i in range(n):
            if mask & (1 << i):
                for j in range(i+1, n):
                    if mask & (1 << j):
                        if conflict_mask[i] & (1 << j):
                            return False
        return True
    
    assert is_valid(best_mask), "Direct construction is invalid!"
    
    # === GREEDY FROM MULTIPLE STARTS ===
    def greedy(order):
        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
        return mask
    
    # Ordering 1: weight descending
    order_w = list(range(n))
    order_w.sort(key=lambda i: -weights[i])
    mask_w = greedy(order_w)
    sum_w = reciprocal_sum_from_mask(mask_w)
    if sum_w > best_sum:
        best_sum = sum_w
        best_mask = mask_w
    
    # Ordering 2: "no 2" strategy - exclude 2, prioritize evens by weight
    order_no2 = [i for i in range(n) if numbers[i] != 2]
    order_no2.sort(key=lambda i: -weights[i])
    mask_no2 = greedy(order_no2)
    sum_no2 = reciprocal_sum_from_mask(mask_no2)
    if sum_no2 > best_sum:
        best_sum = sum_no2
        best_mask = mask_no2
    
    # Ordering 3: "no 2" with evens first (to maximize even coverage)
    order_ev_first = [i for i in range(n) if numbers[i] != 2 and numbers[i] % 2 == 0]
    order_odd_first = [i for i in range(n) if numbers[i] != 2 and numbers[i] % 2 == 1]
    order_ev_first.sort(key=lambda i: -weights[i])
    order_odd_first.sort(key=lambda i: -weights[i])
    mask_mixed = greedy(order_ev_first + order_odd_first)
    sum_mixed = reciprocal_sum_from_mask(mask_mixed)
    if sum_mixed > best_sum:
        best_sum = sum_mixed
        best_mask = mask_mixed

    # Ordering 4: "no 2" - prioritize small evens that are powers of 2 or have few divisors
    # Actually, let's try: exclude 2, but include all odds > 128, then greedily add evens
    mask_no2_odds = 0
    for i in range(n):
        val = numbers[i]
        if val % 2 == 1 and val > 128:
            mask_no2_odds |= (1 << i)
    
    # Now greedily add evens (excluding 2) that don't conflict with mask_no2_odds or each other
    evens_no2 = [i for i in range(n) if numbers[i] != 2 and numbers[i] % 2 == 0]
    evens_no2.sort(key=lambda i: -weights[i])
    mask_no2_odds_ev = mask_no2_odds
    for i in evens_no2:
        if can_add(i, mask_no2_odds_ev):
            mask_no2_odds_ev |= (1 << i)
    sum_no2_odds_ev = reciprocal_sum_from_mask(mask_no2_odds_ev)
    if sum_no2_odds_ev > best_sum:
        best_sum = sum_no2_odds_ev
        best_mask = mask_no2_odds_ev

    # === IMPROVED LOCAL SEARCH ===
    def try_additions(mask):
        new_mask = mask
        for i in range(n):
            if not (new_mask & (1 << i)):
                if can_add(i, new_mask):
                    new_mask |= (1 << i)
        return new_mask
    
    def local_search_with_swaps(mask, max_iters=300):
        """Iterative local search with removal+fill and swap moves."""
        current_mask = mask
        current_sum = reciprocal_sum_from_mask(current_mask)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE - 2:
            improved = False
            iters += 1
            
            # Phase 1: Add all possible
            new_mask = try_additions(current_mask)
            if new_mask != current_mask:
                current_mask = new_mask
                current_sum = reciprocal_sum_from_mask(current_mask)
                improved = True
                continue
            
            # Phase 2: Try 1-removal + greedy fill
            found_improvement = False
            for i in range(n):
                if elapsed() > DEADLINE - 2:
                    break
                if not (current_mask & (1 << i)):
                    continue
                mask_no_i = current_mask & ~(1 << i)
                new_mask = try_additions(mask_no_i)
                new_s = reciprocal_sum_from_mask(new_mask)
                if new_s > current_sum + 1e-12:
                    current_mask = new_mask
                    current_sum = new_s
                    improved = True
                    found_improvement = True
                    break
            
            if found_improvement or elapsed() > DEADLINE - 2:
                continue
            
            # Phase 3: Try swap moves (remove one, add one)
            found_improvement = False
            removed = []
            for i in range(n):
                if elapsed() > DEADLINE - 2:
                    break
                if not (current_mask & (1 << i)):
                    continue
                removed.append(i)
            
            # Try swapping each removed element with each non-removed element
            non_removed = [i for i in range(n) if not (current_mask & (1 << i))]
            for i in removed:
                if elapsed() > DEADLINE - 2:
                    break
                mask_no_i = current_mask & ~(1 << i)
                for j in non_removed:
                    if elapsed() > DEADLINE - 2:
                        break
                    if can_add(j, mask_no_i):
                        new_mask = mask_no_i | (1 << j)
                        new_s = current_sum - weights[i] + weights[j]
                        if new_s > current_sum + 1e-12:
                            current_mask = new_mask
                            current_sum = new_s
                            improved = True
                            found_improvement = True
                            # Update non_removed list
                            non_removed = [k for k in range(n) if not (current_mask & (1 << k))]
                            break
                if found_improvement:
                    break
            
            if found_improvement or elapsed() > DEADLINE - 2:
                continue
            
            # Phase 4: Try 2-removal + greedy fill
            found_improvement = False
            for i in range(n):
                if elapsed() > DEADLINE - 2:
                    break
                if not (current_mask & (1 << i)):
                    continue
                for j in range(i+1, n):
                    if elapsed() > DEADLINE - 2:
                        break
                    if not (current_mask & (1 << j)):
                        continue
                    mask_no_ij = current_mask & ~(1 << i) & ~(1 << j)
                    new_mask = try_additions(mask_no_ij)
                    new_s = reciprocal_sum_from_mask(new_mask)
                    if new_s > current_sum + 1e-12:
                        current_mask = new_mask
                        current_sum = new_s
                        improved = True
                        found_improvement = True
                        break
                if found_improvement:
                    break
        
        return current_mask, current_sum
    
    # Run local search from best candidate
    selected = best_mask
    selected, sel_sum = local_search_with_swaps(selected, max_iters=150)
    if sel_sum > best_sum:
        best_sum = sel_sum
        best_mask = selected
    
    # Also run local search from the "no 2" greedy result
    if mask_no2 != best_mask:
        sel2, sum2 = local_search_with_swaps(mask_no2, max_iters=80)
        if sum2 > best_sum:
            best_sum = sum2
            best_mask = sel2
    
    # Also run local search from the "no 2 odds+evens" result
    if mask_no2_odds_ev != best_mask and elapsed() < DEADLINE - 10:
        sel3, sum3 = local_search_with_swaps(mask_no2_odds_ev, max_iters=80)
        if sum3 > best_sum:
            best_sum = sum3
            best_mask = sel3

    # === SIMULATED ANNEALING on even selection ===
    # The key decision is which evens to include (excluding 2 if we don't include 2)
    # Odds > 128 are always included; odds <= 128 are conditionally included
    # Let's do SA on the full set with swap moves
    
    if elapsed() < DEADLINE - 15:
        # SA parameters
        sa_temp = 0.05
        sa_cool = 0.995
        sa_iters = 5000
        sa_mask = best_mask
        sa_sum = best_sum
        best_sa_mask = sa_mask
        best_sa_sum = sa_sum
        
        for it in range(sa_iters):
            if elapsed() > DEADLINE - 5:
                break
            if it % 100 == 0:
                sa_temp *= sa_cool
            
            # Random move: swap a selected with a non-selected
            sel_list = [i for i in range(n) if sa_mask & (1 << i)]
            non_sel_list = [i for i in range(n) if not (sa_mask & (1 << i))]
            
            if not sel_list or not non_sel_list:
                break
            
            i = random.choice(sel_list)
            j = random.choice(non_sel_list)
            
            new_mask = sa_mask & ~(1 << i) | (1 << j)
            
            # Check validity: j must not conflict with any other selected element
            valid = True
            for k in range(n):
                if k != j and new_mask & (1 << k):
                    if conflict_mask[j] & (1 << k):
                        valid = False
                        break
            
            if not valid:
                continue
            
            new_sum = sa_sum - weights[i] + weights[j]
            delta = new_sum - sa_sum
            
            if delta > 0 or random.random() < pow(2.718281828, delta / sa_temp):
                sa_mask = new_mask
                sa_sum = new_sum
                if sa_sum > best_sa_sum:
                    best_sa_sum = sa_sum
                    best_sa_mask = sa_mask
        
        if best_sa_sum > best_sum:
            best_sum = best_sa_sum
            best_mask = best_sa_mask
            # Polish the SA result
            if elapsed() < DEADLINE - 3:
                polished, pol_sum = local_search_with_swaps(best_mask, max_iters=50)
                if pol_sum > best_sum:
                    best_sum = pol_sum
                    best_mask = polished

    # Final polish: single-pass addition
    if elapsed() < DEADLINE - 2:
        final_mask = try_additions(best_mask)
        final_sum = reciprocal_sum_from_mask(final_mask)
        if final_sum > best_sum:
            best_sum = final_sum
            best_mask = final_mask

    # Convert to result
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(numbers[i])
    result.sort()
    
    if not result:
        result = [2]
    
    return result
