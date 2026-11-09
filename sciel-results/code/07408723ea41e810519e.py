import time
import math
from collections import defaultdict

def construct():
    """
    Returns a list of distinct ints from 1..256, pairwise coprime, maximizing sum.
    
    Strategy:
    1. Start with the base solution: {1} U {L_p for all primes p <= 256}, where L_p is the largest power of p <= 256.
    2. Identify "swaps": composite numbers c where c > sum(L_p for p|c).
    3. Solve the weighted set packing problem: select a subset of swaps such that no prime is used in more than one swap, maximizing total gain.
    4. Use branch-and-bound with tight pruning to find the optimal set of swaps.
    """
    start_time = time.time()
    deadline = start_time + 88  # Safety margin
    
    N = 256
    
    # 1. Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Define base solution
    prime_power_map = {}
    base_sum = 1
    selected_base = {1: True}
    
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_power_map[p] = power
        base_sum += power
        selected_base[power] = True
        
    best_sum = base_sum
    best_set = list(selected_base.keys())
    
    # 3. Identify potential "swaps" (composites c where c > sum of L_p for p|c)
    candidates = []
    for c in range(4, N + 1):
        if is_prime[c]:
            continue
        # Find prime factors of c
        div_primes = []
        temp = c
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                div_primes.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            div_primes.append(temp)
            
        # Calculate gain
        base_val = sum(prime_power_map[p] for p in div_primes)
        gain = c - base_val
        if gain > 0:
            candidates.append((c, div_primes, gain))
            
    # Dominance pruning: if two candidates have the same prime set, keep only the one with max gain
    prime_set_map = defaultdict(list)
    for c, div_primes, gain in candidates:
        ps = frozenset(div_primes)
        prime_set_map[ps].append((c, div_primes, gain))
    
    candidates = []
    for ps, items in prime_set_map.items():
        best_item = max(items, key=lambda x: x[2])
        candidates.append(best_item)
    
    # Sort candidates by gain descending to help pruning
    candidates.sort(key=lambda x: -x[2])
    
    num_candidates = len(candidates)
    
    # Map prime to bitmask for fast conflict checking
    prime_to_bit = {p: (1 << i) for i, p in enumerate(primes)}
    
    # Convert candidate div_primes to bitmask
    candidate_masks = []
    for c, div_primes, gain in candidates:
        mask = 0
        for p in div_primes:
            mask |= prime_to_bit[p]
        candidate_masks.append(mask)
    
    # Precompute suffix sums of gains for O(1) upper bound
    suffix_sum = [0] * (num_candidates + 1)
    for i in range(num_candidates - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + candidates[i][2]
    
    # Greedy initialization to get a good initial solution
    greedy_gain = 0
    greedy_used_mask = 0
    greedy_selections = []
    for i in range(num_candidates):
        c, div_primes, gain = candidates[i]
        mask = candidate_masks[i]
        if (greedy_used_mask & mask) == 0:
            greedy_gain += gain
            greedy_used_mask |= mask
            greedy_selections.append((c, div_primes, gain))
    
    if greedy_gain > 0:
        new_set = set(selected_base.keys())
        for c, div_primes, g in greedy_selections:
            for p in div_primes:
                lp = prime_power_map[p]
                if lp in new_set:
                    new_set.remove(lp)
            new_set.add(c)
        best_sum = base_sum + greedy_gain
        best_set = list(new_set)
    
    # Track best selections for reconstruction
    best_selections = list(greedy_selections) if greedy_gain > 0 else []
    
    # Use iterative deepening with tighter bounds
    # Key insight: for a given idx and used_mask, the maximum additional gain we can get
    # is bounded by the sum of gains of all candidates from idx onward that don't conflict
    # with used_mask. We precompute this more tightly.
    
    def get_tight_bound(idx, used_mask):
        """Compute tighter upper bound on remaining gain."""
        total = 0
        for i in range(idx, num_candidates):
            if (candidate_masks[i] & used_mask) == 0:
                total += candidates[i][2]
        return total
    
    # Branch and bound without memo (memo was too aggressive and could miss solutions)
    nodes_explored = [0]
    
    def backtrack(idx, current_gain, used_mask, current_selections):
        nonlocal best_sum, best_set, best_selections
        nonlocal deadline, start_time
        
        nodes_explored[0] += 1
        
        # Time check every 1000 nodes
        if nodes_explored[0] % 1000 == 0 and time.time() > deadline:
            return
            
        # Pruning: use tight bound
        remaining_bound = get_tight_bound(idx, used_mask)
        if current_gain + remaining_bound <= best_sum - base_sum:
            return
            
        if idx == num_candidates:
            if current_gain > best_sum - base_sum:
                best_sum = base_sum + current_gain
                best_selections = list(current_selections)
                # Construct set
                new_set = set(selected_base.keys())
                for c, div_primes, g in current_selections:
                    for p in div_primes:
                        lp = prime_power_map[p]
                        if lp in new_set:
                            new_set.remove(lp)
                    new_set.add(c)
                best_set = list(new_set)
            return
            
        # Try taking the candidate first (higher gain candidates are already sorted)
        c, div_primes, gain = candidates[idx]
        mask = candidate_masks[idx]
        
        if (used_mask & mask) == 0:
            current_selections.append((c, div_primes, gain))
            backtrack(idx + 1, current_gain + gain, used_mask | mask, current_selections)
            current_selections.pop()
            
        # Option: Skip candidate at idx
        backtrack(idx + 1, current_gain, used_mask, current_selections)
    
    # Run backtracking
    backtrack(0, 0, 0, [])
    
    # Local improvement pass: try to add more composites or swap
    if time.time() < deadline - 5:
        current_set = set(best_set)
        used_primes = set()
        for x in current_set:
            temp = x
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    used_primes.add(p)
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                used_primes.add(temp)
        
        # Try adding each unused composite that doesn't conflict
        for c in range(4, N + 1):
            if c in current_set or is_prime[c]:
                continue
            div_primes = []
            temp = c
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    div_primes.append(p)
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                div_primes.append(temp)
            
            # Check if any prime in div_primes is already used
            if any(p in used_primes for p in div_primes):
                continue
            
            # Check if c conflicts with any element in current_set
            conflict = False
            for x in current_set:
                if math.gcd(c, x) != 1:
                    conflict = True
                    break
            if not conflict:
                current_set.add(c)
                used_primes.update(div_primes)
                best_sum = sum(current_set)
                best_set = list(current_set)
        
        # Try swapping: for each composite in the set, see if we can replace it with a better one
        for c in list(current_set):
            if c < 4 or is_prime[c]:
                continue
            # Find prime factors of c
            div_primes_c = []
            temp = c
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    div_primes_c.append(p)
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                div_primes_c.append(temp)
            
            # Try replacing c with a larger composite using same or subset of primes
            for c2 in range(c + 1, N + 1):
                if c2 in current_set or is_prime[c2]:
                    continue
                div_primes_c2 = []
                temp2 = c2
                for p in primes:
                    if p * p > temp2:
                        break
                    if temp2 % p == 0:
                        div_primes_c2.append(p)
                        while temp2 % p == 0:
                            temp2 //= p
                if temp2 > 1:
                    div_primes_c2.append(temp2)
                
                # c2 must use only primes that c uses (or a subset)
                if not all(p in div_primes_c for p in div_primes_c2):
                    continue
                
                # Check if c2 conflicts with other elements
                conflict = False
                for x in current_set:
                    if x != c and math.gcd(c2, x) != 1:
                        conflict = True
                        break
                if not conflict:
                    current_set.remove(c)
                    current_set.add(c2)
                    best_sum = sum(current_set)
                    best_set = list(current_set)
                    break
    
    # Final validation
    s = set(best_set)
    if len(s) != len(best_set):
        return [1] + [prime_power_map[p] for p in primes]
        
    for i in range(len(best_set)):
        for j in range(i + 1, len(best_set)):
            if math.gcd(best_set[i], best_set[j]) != 1:
                return [1] + [prime_power_map[p] for p in primes]
                
    if not all(1 <= x <= 256 for x in best_set):
        return [1] + [prime_power_map[p] for p in primes]
        
    # Ensure return type is list of ints
    return [int(x) for x in best_set]
