import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 90s kill (using ~58s of 90s budget)
    
    n = 192
    max_size = 48
    
    # Precompute divisors for all possible sums (max sum = 3*192 = 576)
    max_sum = 3 * n
    divisors_of = [[] for _ in range(max_sum + 1)]
    for s in range(1, max_sum + 1):
        for d in range(1, min(s, n) + 1):
            if s % d == 0:
                divisors_of[s].append(d)
    
    # Fast validity check: can we add element 'a' to set S?
    # S is a set of integers
    def can_add(a, S):
        S_list = list(S)
        K = len(S_list)
        S_set = S  # Already a set
        
        # Check 1: a must not divide sum of any 3 distinct elements in S
        # For each multiple M of a (up to 3*n), check if 3 distinct elements in S sum to M
        for M in range(a, max_sum + 1, a):
            if M < 3:
                continue
            # Look for x, y, z in S, distinct, with x+y+z = M
            # Iterate pairs (x, y) with x < y, then z = M - x - y
            # Need z in S, z != x, z != y
            for i in range(K):
                x = S_list[i]
                if x >= M:
                    break
                for j in range(i + 1, K):
                    y = S_list[j]
                    if x + y >= M:
                        break
                    z = M - x - y
                    if z <= y:  # z must be > y for distinctness (since x < y < z)
                        continue
                    if z in S_set:
                        return False
        
        # Check 2: No element b in S should divide (a + u + v) for distinct u, v in S
        # where b != a, b != u, b != v
        # For each pair (u, v) in S, compute T = a + u + v
        # Check if any divisor of T is in S and is not a, u, or v
        for i in range(K):
            u = S_list[i]
            for j in range(i + 1, K):
                v = S_list[j]
                T = a + u + v
                if T > max_sum:
                    continue
                # Check all divisors of T
                for b in divisors_of[T]:
                    if b == a or b == u or b == v:
                        continue
                    if b in S_set:
                        return False
        
        return True
    
    # Full validity check for a set S
    def is_valid_set(S):
        S_list = sorted(S)
        S_set = S
        K = len(S_list)
        if K < 4:
            return True
        
        for idx, a in enumerate(S_list):
            # Check if a divides sum of any 3 distinct other elements
            others = [x for i, x in enumerate(S_list) if i != idx]
            for M in range(a, max_sum + 1, a):
                if M < 3:
                    continue
                for i in range(len(others)):
                    x = others[i]
                    if x >= M:
                        break
                    for j in range(i + 1, len(others)):
                        y = others[j]
                        if x + y >= M:
                            break
                        z = M - x - y
                        if z <= y:
                            continue
                        if z in S_set:
                            return False
        
        # Check the other direction: for each b in S, no 3 distinct others sum to multiple of b
        # This is symmetric to the above, already covered
        return True
    
    # Greedy construction with a given order
    def greedy_build(order):
        S = set()
        for x in order:
            if len(S) >= max_size:
                break
            if can_add(x, S):
                S.add(x)
        return S
    
    # Generate various ordering strategies
    def get_orders():
        orders = []
        # Descending (largest first) - tends to work well
        orders.append(list(range(n, 0, -1)))
        # Ascending
        orders.append(list(range(1, n + 1)))
        # Random orders with fixed seed
        rng = random.Random(12345)
        for _ in range(20):
            o = list(range(1, n + 1))
            rng.shuffle(o)
            orders.append(o)
        # Strategic orders: prefer numbers with fewer divisors (primes, large numbers)
        # Numbers that are hard to divide sums: large numbers, primes
        primes = [x for x in range(2, n + 1) if all(x % p != 0 for p in range(2, int(x**0.5) + 1))]
        non_primes = [x for x in range(1, n + 1) if x not in primes]
        orders.append(primes + non_primes)
        orders.append(sorted(range(1, n + 1), key=lambda x: -len(divisors_of[x * 3])))
        return orders
    
    # Local search: try to improve a set by adding elements or swapping
    def local_search(S, time_limit):
        current_set = set(S)
        best_set = set(S)
        best_size = len(S)
        all_numbers = set(range(1, n + 1))
        
        while time.time() < deadline - 2:
            if len(current_set) >= max_size:
                break
            
            improved = False
            candidates = list(all_numbers - current_set)
            if not candidates:
                break
            
            # Try adding a random candidate
            random.shuffle(candidates)
            for x in candidates[:30]:  # Try up to 30 random additions
                if can_add(x, current_set):
                    current_set.add(x)
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    break
            
            if improved:
                continue
            
            # Try 1-for-1 swaps: remove one, add one
            S_list = list(current_set)
            if not S_list:
                break
            random.shuffle(S_list)
            
            swap_found = False
            for y in S_list[:20]:  # Try removing up to 20 random elements
                S_without_y = current_set - {y}
                for x in candidates[:30]:
                    if x in S_without_y:
                        continue
                    if can_add(x, S_without_y):
                        # Valid swap
                        current_set = S_without_y
                        current_set.add(x)
                        swap_found = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break
                if swap_found:
                    break
            
            if not swap_found:
                # Try 2-for-1: remove two, add one (net -1, but might enable future adds)
                # Or 1-for-2: remove one, add two (net +1)
                # Let's try 1-for-2
                for y in S_list[:10]:
                    S_without_y = current_set - {y}
                    # Try to find two elements x1, x2 that can both be added
                    for x1 in candidates[:20]:
                        if x1 in S_without_y:
                            continue
                        if can_add(x1, S_without_y):
                            S_temp = S_without_y | {x1}
                            for x2 in candidates[:20]:
                                if x2 == x1 or x2 in S_temp:
                                    continue
                                if can_add(x2, S_temp):
                                    current_set = S_temp | {x2}
                                    swap_found = True
                                    if len(current_set) > best_size:
                                        best_size = len(current_set)
                                        best_set = set(current_set)
                                    break
                        if swap_found:
                            break
                    if swap_found:
                        break
                
                if not swap_found:
                    # Stuck, break out
                    break
        
        return best_set
    
    # Main search
    best_overall = set()
    best_size_overall = 0
    
    # Try multiple greedy initializations
    orders = get_orders()
    for o in orders:
        if time.time() > deadline - 15:
            break
        S = greedy_build(o)
        if len(S) > best_size_overall:
            best_size_overall = len(S)
            best_overall = set(S)
    
    # Local search from the best greedy solution
    if time.time() < deadline - 5:
        improved = local_search(best_overall, deadline)
        if len(improved) > best_size_overall:
            best_size_overall = len(improved)
            best_overall = improved
    
    # Try a few more local searches from different starting points
    for o in orders[:5]:
        if time.time() > deadline - 3:
            break
        S = greedy_build(o)
        if len(S) >= best_size_overall - 2:  # Only if close to best
            improved = local_search(S, deadline)
            if len(improved) > best_size_overall:
                best_size_overall = len(improved)
                best_overall = improved
    
    return list(best_overall)
