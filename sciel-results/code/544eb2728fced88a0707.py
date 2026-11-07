"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 256
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, N + 1) if is_prime(p)]
    best_set = set(primes)
    
    # Efficient validity check using precomputation
    # For a set S, check if for all distinct a,b,c in S: a ∤ b*c
    
    def is_valid_set(s):
        """Check validity of set s. O(n^3) but with early exit."""
        lst = list(s)
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(n):
                if j == i:
                    continue
                b = lst[j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = lst[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    # Greedy: try to build set by adding elements in various orders
    def try_greedy(order):
        s = set()
        for a in order:
            if time.time() - start_time > deadline:
                break
            # Check if adding a keeps the set valid
            lst = list(s)
            ok = True
            # a as divisor: a ∤ b*c for all distinct b,c in s
            for i in range(len(lst)):
                b = lst[i]
                for j in range(len(lst)):
                    if j == i:
                        continue
                    c = lst[j]
                    if (b * c) % a == 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                # b as divisor: for b in s, b ∤ a*c for all c in s, c≠b
                for b in lst:
                    for c in lst:
                        if c == b:
                            continue
                        if (a * c) % b == 0:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                s.add(a)
        return s
    
    # Try multiple orderings
    orderings = [
        range(2, N + 1),           # ascending
        range(N, 1, -1),           # descending
        primes + [x for x in range(2, N + 1) if not is_prime(x)],  # primes first
        [x for x in range(2, N + 1) if not is_prime(x)] + primes,  # composites first
    ]
    
    for order in orderings:
        if time.time() - start_time > deadline:
            break
        s = try_greedy(list(order))
        if is_valid_set(s) and len(s) > len(best_set):
            best_set = s
    
    # Local search: try single swaps
    if time.time() - start_time < deadline:
        current = set(best_set)
        for _ in range(50):
            if time.time() - start_time > deadline:
                break
            improved = False
            for remove in list(current):
                if time.time() - start_time > deadline:
                    break
                for add in range(2, N + 1):
                    if add in current:
                        continue
                    new_set = (current - {remove}) | {add}
                    if is_valid_set(new_set):
                        current = new_set
                        if len(current) > len(best_set):
                            best_set = current
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Verify final result
    if not is_valid_set(best_set):
        best_set = set(primes)
    
    return [int(x) for x in sorted(best_set)]

# EVOLVE-BLOCK-END
