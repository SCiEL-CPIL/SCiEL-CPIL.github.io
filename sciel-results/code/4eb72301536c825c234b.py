import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    N = 256
    
    # Find all primes up to N
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, N + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, N + 1) if sieve[i]]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # For each number 2..N, compute its prime factor set
    def prime_factor_set(n):
        s = set()
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                s.add(p)
                while n % p == 0:
                    n //= p
        if n > 1:
            s.add(n)
        return s
    
    # Enumerate all valid prime subsets (product <= N)
    # A subset is valid if product of its primes <= N
    valid_subsets = []  # list of (frozenset_of_prime_indices, product_value)
    
    # Use recursion to enumerate subsets with product <= N
    def enumerate_subsets(idx, current_set, current_product):
        if time.time() > deadline - 5:
            return
        if current_set:
            valid_subsets.append((frozenset(current_set), current_product))
        for i in range(idx, num_primes):
            p = primes[i]
            new_product = current_product * p
            if new_product > N:
                break
            current_set.append(i)
            enumerate_subsets(i + 1, current_set, new_product)
            current_set.pop()
    
    enumerate_subsets(0, [], 1)
    
    num_subsets = len(valid_subsets)
    subset_primes = [s[0] for s in valid_subsets]
    subset_values = [s[1] for s in valid_subsets]
    
    # Precompute conflict matrix: conflict[i][j] = True if subsets i and j share a prime
    # Use a bitset approach for efficiency
    # First, map each prime to a bit position
    prime_bit = {p: (1 << i) for i, p in enumerate(primes)}
    
    # Convert each subset's prime set to a bitmask
    subset_masks = []
    for ps in subset_primes:
        mask = 0
        for p in ps:
            mask |= prime_bit[p]
        subset_masks.append(mask)
    
    # Two subsets conflict if their masks have common bits
    def masks_conflict(m1, m2):
        return (m1 & m2) != 0
    
    # Greedy initialization: sort by value descending, pick greedily
    def greedy_init():
        selected = []
        used_mask = 0
        # Sort by value descending, then by size ascending
        order = sorted(range(num_subsets), key=lambda i: (-subset_values[i], len(subset_primes[i])))
        for idx in order:
            if time.time() > deadline - 5:
                break
            if not masks_conflict(subset_masks[idx], used_mask):
                selected.append(idx)
                used_mask |= subset_masks[idx]
        return selected
    
    # Improved local search with efficient conflict checking
    def local_search(selected, max_iters=10000):
        selected = list(selected)
        selected_set = set(selected)
        current_sum = sum(subset_values[i] for i in selected)
        best_selected = list(selected)
        best_sum = current_sum
        best_selected_set = set(best_selected)
        
        # Precompute for each subset, which other subsets conflict with it
        # For efficiency, we'll check on the fly using bitmasks
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try adding a subset - find the best one to add
            used_mask = 0
            for idx in selected:
                used_mask |= subset_masks[idx]
            
            best_add_idx = -1
            best_add_gain = 0
            for idx in range(num_subsets):
                if idx in selected_set:
                    continue
                if not masks_conflict(subset_masks[idx], used_mask):
                    gain = subset_values[idx]
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_idx = idx
            
            if best_add_idx >= 0 and current_sum + best_add_gain > best_sum:
                best_sum = current_sum + best_add_gain
                best_selected = selected + [best_add_idx]
                best_selected_set = selected_set | {best_add_idx}
                selected = best_selected
                selected_set = best_selected_set
                current_sum = best_sum
                improved = True
                continue
            
            # Try swap: remove one, add one
            if not selected:
                break
            
            best_swap = None  # (remove_idx, add_idx, new_sum)
            for remove_idx in selected:
                new_selected_without = [i for i in selected if i != remove_idx]
                new_used_mask = 0
                for idx in new_selected_without:
                    new_used_mask |= subset_masks[idx]
                new_sum = current_sum - subset_values[remove_idx]
                
                # Find best add
                for add_idx in range(num_subsets):
                    if add_idx in selected_set and add_idx != remove_idx:
                        continue
                    if not masks_conflict(subset_masks[add_idx], new_used_mask):
                        swap_sum = new_sum + subset_values[add_idx]
                        if swap_sum > best_sum:
                            if best_swap is None or swap_sum > best_swap[2]:
                                best_swap = (remove_idx, add_idx, swap_sum)
            
            if best_swap:
                remove_idx, add_idx, new_sum = best_swap
                best_sum = new_sum
                best_selected = [i for i in selected if i != remove_idx] + [add_idx]
                best_selected_set = (selected_set - {remove_idx}) | {add_idx}
                selected = best_selected
                selected_set = best_selected_set
                current_sum = new_sum
                improved = True
                continue
            
            # Try 2-swap: remove two, add two
            if len(selected) < 2:
                if random.random() < 0.1 and selected:
                    # Perturb
                    remove_idx = random.choice(selected)
                    selected = [i for i in selected if i != remove_idx]
                    selected_set = set(selected)
                    current_sum = sum(subset_values[i] for i in selected)
                else:
                    break
            
            if random.random() < 0.3:
                # Try 2-swap
                best_2swap = None  # (rem1, rem2, add1, add2, new_sum)
                sel_list = list(selected)
                for i1 in range(len(sel_list)):
                    for i2 in range(i1 + 1, len(sel_list)):
                        rem1, rem2 = sel_list[i1], sel_list[i2]
                        new_selected_without = [i for i in selected if i != rem1 and i != rem2]
                        new_used_mask = 0
                        for idx in new_selected_without:
                            new_used_mask |= subset_masks[idx]
                        new_sum = current_sum - subset_values[rem1] - subset_values[rem2]
                        
                        # Find best pair to add
                        candidates = []
                        for add_idx in range(num_subsets):
                            if add_idx in selected_set:
                                continue
                            if not masks_conflict(subset_masks[add_idx], new_used_mask):
                                candidates.append(add_idx)
                        
                        # Try pairs from candidates
                        for j1 in range(len(candidates)):
                            for j2 in range(j1 + 1, len(candidates)):
                                a1, a2 = candidates[j1], candidates[j2]
                                if not masks_conflict(subset_masks[a1], subset_masks[a2]):
                                    two_sum = new_sum + subset_values[a1] + subset_values[a2]
                                    if two_sum > best_sum:
                                        if best_2swap is None or two_sum > best_2swap[4]:
                                            best_2swap = (rem1, rem2, a1, a2, two_sum)
            
            if best_2swap:
                rem1, rem2, a1, a2, new_sum = best_2swap
                best_sum = new_sum
                best_selected = [i for i in selected if i != rem1 and i != rem2] + [a1, a2]
                best_selected_set = (selected_set - {rem1, rem2}) | {a1, a2}
                selected = best_selected
                selected_set = best_selected_set
                current_sum = new_sum
                improved = True
                continue
            
            # No improvement, perturb or stop
            if random.random() < 0.05 and selected:
                remove_idx = random.choice(selected)
                selected = [i for i in selected if i != remove_idx]
                selected_set = set(selected)
                current_sum = sum(subset_values[i] for i in selected)
            else:
                break
        
        return best_selected, best_sum
    
    # Run multiple restarts with diverse strategies
    best_overall = []
    best_overall_sum = 0
    
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if r == 0:
            selected = greedy_init()
        elif r == 1:
            # Start with largest primes only
            selected = []
            used_mask = 0
            for p in sorted(primes, reverse=True):
                if time.time() > deadline - 5:
                    break
                # Find subset for this prime (just the prime itself)
                for idx in range(num_subsets):
                    if len(subset_primes[idx]) == 1 and list(subset_primes[idx])[0] == prime_to_idx[p]:
                        if not masks_conflict(subset_masks[idx], used_mask):
                            selected.append(idx)
                            used_mask |= subset_masks[idx]
                        break
        elif r == 2:
            # Random valid selection
            selected = []
            used_mask = 0
            order = list(range(num_subsets))
            random.shuffle(order)
            for idx in order:
                if time.time() > deadline - 5:
                    break
                if not masks_conflict(subset_masks[idx], used_mask):
                    selected.append(idx)
                    used_mask |= subset_masks[idx]
                if len(selected) >= 15:
                    break
        else:
            # Perturbed greedy
            selected = greedy_init()
            if selected:
                # Remove 1-2 random elements
                num_remove = random.randint(1, min(2, len(selected)))
                for _ in range(num_remove):
                    if selected:
                        remove_idx = random.choice(selected)
                        selected = [i for i in selected if i != remove_idx]
        
        result, result_sum = local_search(selected)
        if result_sum > best_overall_sum:
            best_overall_sum = result_sum
            best_overall = result
    
    # Convert selected subsets back to numbers
    result_numbers = [1]  # Always include 1
    for idx in best_overall:
        prod = subset_values[idx]
        result_numbers.append(prod)
    
    # Verify: all distinct, all in 1..256, pairwise coprime
    assert len(result_numbers) == len(set(result_numbers))
    for x in result_numbers:
        assert 1 <= x <= 256
    for i in range(len(result_numbers)):
        for j in range(i + 1, len(result_numbers)):
            assert math.gcd(result_numbers[i], result_numbers[j]) == 1
    
    return result_numbers
