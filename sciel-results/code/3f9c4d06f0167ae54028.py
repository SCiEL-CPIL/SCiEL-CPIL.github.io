"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 82s safety margin for 90s limit
    
    N = 128
    
    # Precompute GCD table
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]
    
    all_numbers = list(range(1, N + 1))
    
    # Precompute for each pair (a,b), the set of x such that gcd(a,x)=gcd(b,x)=gcd(a,b)
    # This allows O(1) conflict checking via bitsets
    # conflict_mask[a][b] = bitmask of x where gcd(a,x)=gcd(b,x)=gcd(a,b)
    conflict_mask = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = g[a][b]
            mask = 0
            for x in range(1, N + 1):
                if x != a and x != b:
                    if g[a][x] == gab and g[b][x] == gab:
                        mask |= (1 << x)
            conflict_mask[a][b] = mask
            conflict_mask[b][a] = mask
    
    def can_add(x, S_list):
        """Check if x can be added to S_list without creating a violating triple."""
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            gax = g[a][x]
            for j in range(i + 1, n):
                b = S_list[j]
                if g[a][b] == gax and g[b][x] == gax:
                    return False
        return True
    
    def can_add_fast(x, S_list):
        """Faster check using precomputed conflict masks."""
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            # Check if x is in conflict_mask[a][b] for any b in S_list (b != a)
            for j in range(n):
                if j == i:
                    continue
                b = S_list[j]
                if (conflict_mask[a][b] >> x) & 1:
                    return False
        return True
    
    def greedy_add_order(order, S_init=None):
        """Greedy: add elements in given order if they don't create violations."""
        S = set(S_init) if S_init else set()
        S_list = list(S)
        for x in order:
            if x in S:
                continue
            if can_add_fast(x, S_list):
                S.add(x)
                S_list.append(x)
        return S
    
    def count_conflicts_with_set(x, S_list):
        """Count how many pairs in S_list would conflict with x."""
        count = 0
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (conflict_mask[a][b] >> x) & 1:
                    count += 1
        return count
    
    def swap_improve(S):
        """Try to improve S by removing one element and adding others."""
        S = set(S)
        S_list = list(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each element and greedily adding others
            for rem in S_list:
                S_trial = S - {rem}
                S_trial_list = [x for x in S_list if x != rem]
                # Try to add elements not in S, sorted by conflict count
                candidates = [x for x in all_numbers if x not in S_trial]
                
                # Sort by conflict count (fewer conflicts first)
                if len(candidates) <= 80:
                    candidates.sort(key=lambda x: count_conflicts_with_set(x, S_trial_list))
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if can_add_fast(x, S_trial_list):
                        S_trial.add(x)
                        S_trial_list.append(x)
                
                if len(S_trial) > len(S):
                    S = S_trial
                    S_list = list(S)
                    improved = True
                    break
        return S
    
    def find_violations_fast(S_list):
        """Find all violating triples using precomputed masks."""
        violations = []
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                # Check all c > j
                for k in range(j + 1, n):
                    c = S_list[k]
                    if (conflict_mask[a][b] >> c) & 1:
                        violations.append((a, b, c))
        return violations
    
    def local_search_remove(S):
        """Remove elements to eliminate all violations."""
        S = set(S)
        S_list = list(S)
        
        while True:
            if time.time() > deadline:
                return S
            violations = find_violations_fast(S_list)
            if not violations:
                return S
            
            # Count conflicts per element
            conflict_count = {}
            for a, b, c in violations:
                conflict_count[a] = conflict_count.get(a, 0) + 1
                conflict_count[b] = conflict_count.get(b, 0) + 1
                conflict_count[c] = conflict_count.get(c, 0) + 1
            
            # Remove element with max conflicts
            max_conf = -1
            to_remove = None
            for elem in S:
                cc = conflict_count.get(elem, 0)
                if cc > max_conf:
                    max_conf = cc
                    to_remove = elem
                elif cc == max_conf and random.random() < 0.5:
                    to_remove = elem
            
            S.remove(to_remove)
            S_list = [x for x in S_list if x != to_remove]
    
    # Strategy 1: Multiple random greedy passes
    best_set = set()
    
    for trial in range(400):
        if time.time() > deadline - 10:
            break
        
        if trial % 50 == 0:
            random.seed(trial * 7919 + 13)
        
        # Different ordering strategies
        if trial % 4 == 0:
            order = list(all_numbers)
            random.shuffle(order)
        elif trial % 4 == 1:
            # Sort by number of divisors (fewer divisors = less likely to conflict)
            def num_divisors(n):
                count = 0
                for d in range(1, int(n**0.5) + 1):
                    if n % d == 0:
                        count += 2
                        if d * d == n:
                            count -= 1
                return count
            order = sorted(all_numbers, key=lambda x: (num_divisors(x), x))
        elif trial % 4 == 2:
            # Sort by value (smaller first)
            order = list(range(1, N + 1))
            random.shuffle(order)
        else:
            # Sort by number of distinct prime factors (fewer = better)
            def num_distinct_primes(n):
                count = 0
                temp = n
                for p in range(2, int(n**0.5) + 1):
                    if temp % p == 0:
                        count += 1
                        while temp % p == 0:
                            temp //= p
                if temp > 1:
                    count += 1
                return count
            order = sorted(all_numbers, key=lambda x: (num_distinct_primes(x), x))
        
        S = greedy_add_order(order)
        
        # Try swap improvement
        if time.time() < deadline - 5:
            S = swap_improve(S)
        
        if len(S) > len(best_set):
            best_set = S
    
    # Strategy 2: Start from dense subsets and prune
    dense_starts = [
        list(range(1, N + 1)),
        [x for x in range(1, N + 1) if x % 2 == 1],  # odds
        [x for x in range(1, N + 1) if x % 2 == 0],  # evens
        [x for x in range(1, N + 1) if x % 2 == 0 or x % 3 == 0],
        [x for x in range(1, N + 1) if x % 3 != 0],
        [x for x in range(1, N + 1) if x % 5 != 0],
        [x for x in range(1, N + 1) if x % 2 != 0 or x > 64],  # odds + large evens
        [x for x in range(1, N + 1) if x % 4 != 0],  # not divisible by 4
    ]
    
    for ds in dense_starts:
        if time.time() > deadline - 15:
            break
        S = local_search_remove(ds)
        if time.time() < deadline - 5:
            S = swap_improve(S)
        if len(S) > len(best_set):
            best_set = S
    
    # Strategy 3: Iterated local search from best_set
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        # Perturb: remove 2-4 random elements, then re-add greedily
        S_perturbed = set(best_set)
        S_list = list(S_perturbed)
        k = min(4, len(S_list))
        to_remove = random.sample(S_list, k)
        for x in to_remove:
            S_perturbed.discard(x)
        
        # Re-add removed elements and try others
        S_final = swap_improve(S_perturbed)
        if len(S_final) > len(best_set):
            best_set = S_final
    
    # Strategy 4: Final polish - try to add any remaining elements
    if time.time() < deadline - 2:
        S_list = list(best_set)
        candidates = [x for x in all_numbers if x not in best_set]
        candidates.sort(key=lambda x: count_conflicts_with_set(x, S_list))
        for x in candidates:
            if can_add_fast(x, S_list):
                best_set.add(x)
                S_list.append(x)
    
    # Final: ensure no violations
    final_S = local_search_remove(best_set)
    
    return list(final_S)
# EVOLVE-BLOCK-END
