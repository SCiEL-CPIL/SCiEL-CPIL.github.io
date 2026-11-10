import time
import math
from functools import reduce

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, keep 5s buffer
    
    N = 22
    universe = list(range(1, N + 1))
    
    # Choose a large prime P such that P > MaxPossibleNumerator
    # Max numerator for diff of two subset sums of [1..22] reciprocals
    # D = lcm(1..22)
    D = 1
    for i in range(1, N + 1):
        D = math.lcm(D, i)
        
    # Max sum of reciprocals is H_22 ~ 3.7
    # Max numerator N_max ~ 2 * D * H_22
    N_max = 2 * D * 4  # Safe bound
    
    # Find a prime P > N_max
    # 10^9 + 7 is prime and > 1.4e8 * 8 ~ 1.1e9? 
    # D = 2^4 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19
    # D = 16 * 9 * 5 * 7 * 11 * 13 * 17 * 19
    # 16*9=144, 144*5=720, 720*7=5040, 5040*11=55440, 55440*13=720720, 720720*17=12252240, 12252240*19=232792560
    # D ~ 2.3 * 10^8
    # N_max ~ 2 * 2.3e8 * 4 ~ 1.8 * 10^9
    # So P = 10^9 + 7 is NOT large enough! 1.8e9 > 1e9.
    # We need a larger prime.
    # Let's use P = 10^12 + 39? Or just a larger known prime.
    # 2^61 - 1 is prime? Mersenne prime. 2^61 - 1 ~ 2.3 * 10^18.
    # Python handles big ints natively, but modular arithmetic with big ints is slower than 64-bit.
    # However, 10^12 fits in 64-bit.
    # Let's find a prime around 10^12.
    # 1000000000039 is prime?
    # Or simply use Python's arbitrary precision and pick P = 10**18 + 9?
    # Actually, we can just use a prime like 99999999999973?
    # Let's use P = 10**15 + 37 (need to verify primality or just use a known one).
    # Known prime: 1000000007 is too small.
    # Let's use P = 2305843009213693951 (2^61 - 1).
    # Or P = 1000000000000000003? 
    # Let's just use a large prime that fits in 64 bits for speed, or accept big int overhead.
    # The overhead of big ints in Python is significant.
    # Let's try to find a prime P in [2*N_max, 10*N_max].
    # N_max ~ 2 * 2.3e8 * 4 = 1.84e9.
    # So we need P > 1.84e9.
    # 2^31 - 1 = 2147483647 (Prime).
    # 2^31 - 1 > 1.84e9.
    # So P = 2147483647 works and fits in 32-bit/64-bit efficiently.
    
    P = 2147483647  # 2^31 - 1, a Mersenne prime
    
    # Verify P > N_max
    # D = 232792560
    # N_max = 2 * D * 4 = 1862340480
    # P = 2147483647
    # P > N_max holds.
    
    # Precompute modular inverses
    weights = [0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = pow(a, P - 2, P)
        
    # We want to find the largest K such that there exists a subset of size K
    # with all subset sums distinct.
    
    best_set = []
    
    # Helper to check if a set of elements is dissociated
    def is_dissociated_mod(elems):
        if not elems:
            return True
        sums = {0}
        for e in elems:
            w = weights[e]
            new_sums = set()
            for s in sums:
                ns = (s + w) % P
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums = sums | new_sums
        return True

    # Backtracking search for a set of size K
    # We try to build a set of size K from the universe.
    # To maximize speed, we can order elements.
    # Elements with smaller values (larger reciprocals) might be more "distinctive"?
    # Or larger values (smaller reciprocals) are closer to 0, might cause collisions?
    # Let's try sorting in descending order of value (ascending reciprocal) or vice versa.
    # Actually, let's just iterate K from 22 down to 1.
    
    # Optimization: If we find a set of size K, we return it.
    # We use a recursive search.
    
    # To avoid deep recursion limits or slow Python recursion, we can use a stack or just standard recursion.
    # Depth is max 22, so recursion is fine.
    
    # Pruning:
    # 1. If current size + remaining available < K, prune.
    # 2. If current set is not dissociated, prune (handled by only adding if valid).
    
    # We will try to find a set of size K.
    # We can use a heuristic: try to add elements that are "easy" to fit?
    # Or just standard order.
    
    # Let's try to find the max K.
    
    # To speed up, we can use a greedy approach to find a good lower bound,
    # then backtracking to see if we can go higher.
    
    # Greedy:
    greedy_set = []
    remaining = list(range(1, N + 1))
    # Try adding in order of decreasing magnitude of reciprocal (increasing a? no, decreasing a)
    # 1/1 is largest.
    for a in sorted(universe, reverse=True): # Try larger a first? 
        # Actually, let's try both orders in a quick check?
        pass

    # Let's just do backtracking for K from 22 down to 1.
    # The first valid K found is the maximum.
    
    # To make backtracking faster, we can pre-compute some things?
    # No, the collision check is dynamic.
    
    # We'll use a global variable for the found set to avoid passing around.
    found_set = []
    found = False
    
    def backtrack(start_idx, current_set, current_sums, K):
        nonlocal found_set, found
        if found:
            return
        
        if len(current_set) == K:
            found_set = current_set.copy()
            found = True
            return
        
        # Pruning: not enough elements left
        if len(current_set) + (N - start_idx) < K:
            return
            
        # Time check
        if time.time() > deadline - 1:
            return
            
        for i in range(start_idx, N + 1):
            e = i
            # Check if we can add e
            w = weights[e]
            # Check collisions
            # New sums = { (s + w) % P for s in current_sums }
            # If any new sum is in current_sums or in new_sums, collision.
            new_sums_set = set()
            valid = True
            for s in current_sums:
                ns = (s + w) % P
                if ns in current_sums or ns in new_sums_set:
                    valid = False
                    break
                new_sums_set.add(ns)
                
            if valid:
                next_sums = current_sums | new_sums_set
                current_set.append(e)
                backtrack(i + 1, current_set, next_sums, K)
                current_set.pop()
                if found:
                    return

    # We try K from 22 down to 1
    # However, backtracking for K=22 might be very slow if no solution exists.
    # We can add a timeout per K?
    # Or just let it run until deadline.
    
    # To be safe, we can start with a known good size (e.g., 10) and go up?
    # No, we want the MAX. So we must check 22, 21, ...
    # If 22 is impossible, the search space is huge.
    # Pruning is key.
    
    # Alternative: Use the fact that if a set of size K is dissociated,
    # any subset is dissociated.
    # We can search for the max size directly.
    
    # Let's try K from 22 down to 1.
    # If we find one, we return it.
    
    # To speed up the search for large K, we can use a heuristic ordering of elements.
    # Elements that are "hard" to fit should be tried later? Or earlier?
    # If we try to fit small reciprocals (large a) first, the sums stay small?
    # Mod P, "small" doesn't matter, only collisions.
    
    # Let's just run the backtracking.
    
    # One optimization: Sort the universe?
    # The indices in `universe` are 1..22.
    # We iterate `i` from `start_idx` to `N`.
    # This fixes the order of addition.
    # We can try different permutations? No, too expensive.
    # But we can try a few heuristic orderings for the *initial* greedy pass to get a good lower bound,
    # then use that to prune?
    # Actually, if we are searching for K, we don't need a lower bound to prune,
    # except to know if K is possible.
    
    # Let's just run the loop.
    
    # To avoid getting stuck on K=22 for 50 seconds, we can limit the time per K?
    # If K=22 takes too long, we might miss K=21.
    # But if K=22 is impossible, we *must* prove it or timeout.
    # Given the constraints, it's likely that K=22, 21, 20 are impossible.
    # The max is likely around 10-15.
    
    # We can use a "fail fast" heuristic?
    # No, exact check is required.
    
    # Let's run it.
    
    # We'll try K from 22 down to 1.
    # If we don't find any, we return the best found in a previous greedy phase?
    # Let's add a quick greedy phase to ensure we have *some* valid set.
    
    # Greedy Phase
    best_greedy = []
    # Try a few orderings
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        list(universe)
    ]
    for order in orders:
        cur = []
        cur_sums = {0}
        for e in order:
            w = weights[e]
            new_sums_set = set()
            valid = True
            for s in cur_sums:
                ns = (s + w) % P
                if ns in cur_sums or ns in new_sums_set:
                    valid = False
                    break
                new_sums_set.add(ns)
            if valid:
                cur.append(e)
                cur_sums = cur_sums | new_sums_set
        if len(cur) > len(best_greedy):
            best_greedy = cur
            
    # Now, search for K > len(best_greedy)
    # We can start K from min(22, len(best_greedy) + 5) down to len(best_greedy) + 1?
    # No, we want the MAX. So we should check 22, 21, ...
    # But if we know 22 is impossible, we waste time.
    # Let's just check 22 down to 1.
    
    # To prevent timeout, we'll check time inside backtrack.
    
    for K in range(N, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # If K <= len(best_greedy), we already have a valid set of this size.
        # We can just return best_greedy if we can't find a larger one?
        # But we want to find if a *larger* one exists.
        # So we only need to search K > len(best_greedy).
        # If we find one, we return it.
        # If we don't find any K > len(best_greedy), we return best_greedy.
        
        if K <= len(best_greedy):
            break
            
        found = False
        found_set = []
        current_set = []
        current_sums = {0}
        
        # We can add a simple timeout for each K to avoid getting stuck?
        # If K=22 takes 10s, we might not have time for K=21.
        # Let's allocate time proportionally?
        # Or just let the global deadline handle it.
        
        backtrack(1, current_set, current_sums, K)
        
        if found:
            return sorted(found_set)
            
    # If no larger set found, return the best greedy set
    return sorted(best_greedy)

# EVOLVE-BLOCK-END
