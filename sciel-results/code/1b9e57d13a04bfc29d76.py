import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # 1. Generate primes up to 256
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(256)
    small_primes = [p for p in primes if p <= 13]  # 2, 3, 5, 7, 11, 13
    large_primes = [p for p in primes if p > 13]

    # 2. Generate candidate pool U: numbers in [2,256] with all prime factors in small_primes
    U = []
    for x in range(2, 257):
        y = x
        valid = True
        for p in small_primes:
            while y % p == 0:
                y //= p
        if y == 1:
            U.append(x)
        else:
            valid = False # y > 1 means there's a prime factor > 13

    nU = len(U)
    
    # Sort U to help greedy: 
    # Numbers with fewer prime factors are "safer" (less likely to divide products of others? 
    # Actually, small numbers divide more things. Large numbers are divided by fewer things.
    # In an antichain, we often pick larger numbers.
    # Let's try sorting by value descending first.
    
    # Precompute antichain conflicts
    # antichain_conflict[i] = set of j where U[i] | U[j] or U[j] | U[i]
    antichain_conflict = [set() for _ in range(nU)]
    for i in range(nU):
        for j in range(i + 1, nU):
            if U[i] % U[j] == 0 or U[j] % U[i] == 0:
                antichain_conflict[i].add(j)
                antichain_conflict[j].add(i)

    # Precompute product divisibility: 
    # prod_conflict[i] = set of pairs (j, k) such that U[i] | U[j] * U[k]
    # To save memory and time, we can store for each i, a list of sets, or just a 3D check.
    # Given nU ~ 100, nU^3 = 1,000,000. 
    # Let's precompute a 3D boolean table or a list of sets of tuples.
    # Actually, we just need to check validity of a set C.
    # Let's precompute for each i, a bitmask of j such that U[i] | U[j] * U[k] is possible?
    # No, it depends on k.
    
    # Let's use the triple_conflict approach from the current program but optimized.
    # triple_conflict[a][b] = set of c where U[c] | U[a] * U[b]
    # This allows O(1) check for "is c in triple_conflict[a][b]?"
    
    triple_conflict = [[set() for _ in range(nU)] for _ in range(nU)]
    for a in range(nU):
        for b in range(nU):
            if a == b:
                continue
            ua = U[a]
            ub = U[b]
            prod = ua * ub
            for c in range(nU):
                if c == a or c == b:
                    continue
                if prod % U[c] == 0:
                    triple_conflict[a][b].add(c)

    def can_add_idx(x_idx, C_set):
        """Check if U[x_idx] can be added to set C (given as set of indices)."""
        # Antichain check
        for y_idx in C_set:
            if y_idx in antichain_conflict[x_idx]:
                return False
        # Product checks
        C_list = list(C_set)
        nC = len(C_list)
        for i in range(nC):
            y_idx = C_list[i]
            for j in range(i + 1, nC):
                z_idx = C_list[j]
                # x !| y*z
                if x_idx in triple_conflict[y_idx][z_idx]:
                    return False
                # y !| x*z
                if y_idx in triple_conflict[x_idx][z_idx]:
                    return False
                # z !| x*y
                if z_idx in triple_conflict[x_idx][y_idx]:
                    return False
        return True

    def greedy_search(order):
        """Greedy: add elements in given order if they fit."""
        C_set = set()
        C_list = []
        for x_idx in order:
            if can_add_idx(x_idx, C_set):
                C_set.add(x_idx)
                C_list.append(x_idx)
        return C_list

    # Initial solution: all large primes
    best_solution = large_primes[:]
    best_C = []
    best_C_size = 0

    # Greedy with different orderings
    orders = [
        list(range(nU)),                      # ascending
        list(range(nU - 1, -1, -1)),          # descending
        sorted(range(nU), key=lambda i: -len([1 for p in small_primes if U[i] % p == 0])), # fewer factors first?
    ]

    # Add random orderings
    rng = random.Random(42)
    for _ in range(100):
        order = list(range(nU))
        rng.shuffle(order)
        orders.append(order)

    for order in orders:
        if time.time() > deadline:
            break
        C_list = greedy_search(order)
        if len(C_list) > best_C_size:
            best_C_size = len(C_list)
            best_C = C_list[:]
            candidate = large_primes + [U[i] for i in best_C]
            best_solution = candidate

    # Local search: try to improve by removing one and adding another
    def local_search(C_list, max_iter=500):
        """Improve C by swap operations."""
        C_set = set(C_list)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            C_list_copy = C_list[:]
            found_swap = False
            for i, rem_idx in enumerate(C_list_copy):
                if found_swap:
                    break
                C_set.remove(rem_idx)
                # Try adding each element not in C
                for add_idx in range(nU):
                    if add_idx in C_set:
                        continue
                    if can_add_idx(add_idx, C_set):
                        # Check if this swap actually increases size? 
                        # No, swap keeps size same. 
                        # We want to INCREASE size. 
                        # So we should try to ADD without removing first.
                        # But if we are at a local maximum, we might need to swap to enable a future add.
                        # However, the objective is cardinality. 
                        # A simple local search for MAX cardinality usually involves:
                        # 1. Try to add any element. If success, done.
                        # 2. If no element can be added, try a swap (remove 1, add 1) to change the configuration, 
                        #    hoping that a different element can be added next.
                        
                        # Let's just do: if we can add ANY element, we do it (size +1).
                        # The loop above is for swaps. 
                        # Let's restructure local_search to first try additions.
                        
                        pass 
                C_set.add(rem_idx) # restore if no swap taken
                
            # Attempt 1: Can we add any new element?
            for add_idx in range(nU):
                if add_idx in C_set:
                    continue
                if can_add_idx(add_idx, C_set):
                    C_set.add(add_idx)
                    C_list.append(add_idx)
                    improved = True
                    break
            
            # Attempt 2: If no addition possible, try swaps to escape local optimum
            if not improved:
                for i, rem_idx in enumerate(C_list):
                    C_set.remove(rem_idx)
                    for add_idx in range(nU):
                        if add_idx in C_set or add_idx == rem_idx:
                            continue
                        if can_add_idx(add_idx, C_set):
                            # Perform swap
                            C_set.add(add_idx)
                            C_list.remove(rem_idx)
                            C_list.append(add_idx)
                            improved = True
                            break
                    C_set.add(rem_idx) # Restore if no swap made
                    if improved:
                        break
                        
        return C_list

    if time.time() < deadline:
        improved_C = local_search(best_C, max_iter=200)
        if len(improved_C) > best_C_size:
            best_C_size = len(improved_C)
            best_C = improved_C[:]
            candidate = large_primes + [U[i] for i in best_C]
            best_solution = candidate

    # More randomized restarts with time remaining
    if time.time() < deadline:
        for _ in range(500):
            if time.time() > deadline:
                break
            order = list(range(nU))
            rng.shuffle(order)
            C_list = greedy_search(order)
            if len(C_list) > best_C_size:
                best_C_size = len(C_list)
                best_C = C_list[:]
                candidate = large_primes + [U[i] for i in best_C]
                best_solution = candidate
            # Quick local search on this result
            if len(C_list) >= best_C_size - 1 and time.time() < deadline - 5:
                improved_C = local_search(C_list, max_iter=50)
                if len(improved_C) > best_C_size:
                    best_C_size = len(improved_C)
                    best_C = improved_C[:]
                    candidate = large_primes + [U[i] for i in best_C]
                    best_solution = candidate

    # Final verification
    result = list(best_solution)
    result = [int(x) for x in result]

    return result
