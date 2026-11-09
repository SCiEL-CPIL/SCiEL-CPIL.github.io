import math
import time
import heapq
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210

    # Compute parity sets using sieve
    def compute_parity_sets(limit):
        spf = list(range(limit + 1))
        for i in range(2, int(limit ** 0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, limit + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        parity_cache = [frozenset() for _ in range(limit + 1)]
        for i in range(2, limit + 1):
            s = set()
            temp = i
            while temp > 1:
                p = spf[temp]
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    s.add(p)
            parity_cache[i] = frozenset(s)
        return parity_cache

    parity_cache = compute_parity_sets(max_val)
    target_parity = frozenset({2, 3, 5, 7})

    best_answer = None
    best_raw = float('inf')

    def update_best(ans, raw):
        nonlocal best_answer, best_raw
        if raw < best_raw:
            prod = 210
            for x in ans:
                prod *= x
            r = math.isqrt(prod)
            if r * r == prod:
                if all(isinstance(x, int) and 210 < x <= max_val for x in ans):
                    if len(set(ans)) == len(ans):
                        best_answer = [int(x) for x in ans]
                        best_raw = raw

    # 1. Single element
    for k in range(2, 100):
        x = 210 * k * k
        if x > max_val:
            break
        update_best([x], x - 210)

    # 2. Two elements
    for k in range(1, 200):
        if time.time() > deadline - 40:
            break
        val = 210 * k * k
        if val > max_val * max_val:
            break
        sqrt_val = math.isqrt(val)
        lo = max(211, math.ceil(val / max_val))
        hi = min(sqrt_val, max_val)
        for x in range(lo, hi + 1):
            if val % x == 0:
                y = val // x
                if y > 210 and y <= max_val and x != y:
                    m = max(x, y)
                    if m - 210 < best_raw:
                        update_best([x, y], m - 210)

    # 3. Bottleneck Dijkstra on parity states
    # State: (current_parity, max_value_so_far, used_numbers_set)
    # We want to reach target_parity with minimal max_value
    # But tracking used numbers is expensive. Instead, we track the set of used numbers implicitly.
    # 
    # Alternative: for each candidate maximum M, check if there exists a set of distinct
    # numbers in [211, M] whose product has parity = target.
    # Use BFS/DFS on parity states with the constraint that all numbers <= M.
    
    # For efficiency, for each M, build a set of available parity transitions
    # and do BFS from empty set to target_parity
    
    if time.time() < deadline - 20:
        # Try increasing M
        for M in range(250, min(700, max_val + 1)):
            if time.time() > deadline - 10:
                break
            if M - 210 >= best_raw:
                break  # No point trying larger M if we already have better
            
            # Build list of (number, parity) for numbers in [211, M]
            # Group by parity
            parity_to_nums = defaultdict(list)
            for i in range(211, M + 1):
                parity_to_nums[parity_cache[i]].append(i)
            
            # BFS on parity states
            # State: (parity, frozenset_of_used_numbers)
            # Too expensive. Instead, use the fact that we just need ANY set.
            # 
            # Key insight: if we have multiple numbers with the same parity, we can use any of them.
            # So we just need to know which parity sets are available.
            # 
            # BFS where state is just the current parity, and we track which numbers we've used.
            # But we need distinct numbers, so we need to track usage.
            # 
            # Alternative: for each parity P, keep a list of numbers with that parity.
            # When transitioning, pick a number we haven't used yet.
            # 
            # This is still complex. Let's use a different approach:
            # For 3 elements, we already have a method. For 4 elements, use pair grouping.
            
            # Let's do 3 elements with the improved method
            found = False
            for z in range(211, M + 1):
                if time.time() > deadline - 8:
                    break
                pz = parity_cache[z]
                required = target_parity.symmetric_difference(pz)
                
                # Find x, y with parity(x) XOR parity(y) = required, x,y != z
                for P1, nums1 in parity_to_nums.items():
                    P2 = P1.symmetric_difference(required)
                    if P2 not in parity_to_nums:
                        continue
                    nums2 = parity_to_nums[P2]
                    
                    # Find x in nums1, y in nums2, all distinct
                    x_list = [x for x in nums1 if x != z]
                    y_list = [y for y in nums2 if y != z]
                    
                    if not x_list or not y_list:
                        continue
                    
                    x = min(x_list)
                    y_candidates = [y for y in y_list if y != x]
                    if not y_candidates:
                        continue
                    y = min(y_candidates)
                    
                    m = max(x, y, z)
                    if m <= M and m - 210 < best_raw:
                        update_best([x, y, z], m - 210)
                        found = True
                        break
                if found:
                    break
            
            if best_raw <= M - 210:
                break

    # 4. Four elements
    if time.time() < deadline - 15 and best_raw > 350:
        M4 = 450
        parity_to_nums_4 = defaultdict(list)
        for i in range(211, M4 + 1):
            parity_to_nums_4[parity_cache[i]].append(i)
        
        pair_parity_map = defaultdict(list)
        for i in range(211, M4 + 1):
            pi = parity_cache[i]
            for j in range(i + 1, M4 + 1):
                pj = parity_cache[j]
                pxy = pi.symmetric_difference(pj)
                pair_parity_map[pxy].append((max(i, j), i, j))
        
        found4 = False
        for P1, pairs1 in pair_parity_map.items():
            if time.time() > deadline - 10:
                break
            P2 = target_parity.symmetric_difference(P1)
            if P2 not in pair_parity_map:
                continue
            pairs2 = pair_parity_map[P2]
            
            pairs1_sorted = sorted(pairs1, key=lambda t: t[0])
            pairs2_sorted = sorted(pairs2, key=lambda t: t[0])
            
            for m1, a, b in pairs1_sorted[:100]:
                if time.time() > deadline - 10:
                    break
                for m2, c, d in pairs2_sorted[:100]:
                    if a != c and a != d and b != c and b != d:
                        m = max(m1, m2)
                        if m - 210 < best_raw:
                            update_best([a, b, c, d], m - 210)
                            found4 = True
                            break
                if found4:
                    break
            if found4:
                break

    if best_answer:
        return best_answer
    return [840]
