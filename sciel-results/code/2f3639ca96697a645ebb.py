import time
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    
    def recip_sum(prime_list):
        s = Fraction(0, 1)
        for p in prime_list:
            s += Fraction(1, p)
        return s

    candidate_sets = [
        [2, 5, 7, 11, 17],
        [3, 5, 7, 11, 13, 17, 19, 23],
        [2, 5, 7, 13, 17],
        [2, 3, 13, 17],
        [2, 5, 11, 13, 17], # 0.5+0.2+0.09+0.076+0.058 = 0.924
        [3, 5, 7, 11, 13, 17, 19], # 0.956
    ]
    
    valid_sets = []
    seen = set()
    for ps in candidate_sets:
        key = tuple(sorted(ps))
        if key in seen: continue
        seen.add(key)
        if recip_sum(ps) <= 1:
            valid_sets.append(ps)

    best_L = 0
    best_assign = None
    best_set = None

    def solve_csp(L, prime_list):
        """
        Returns assignment dict {p: r} if 1..L can be covered, else None.
        """
        n = L
        covered = bytearray(n + 1)
        # uncovered_count[p_idx][r]
        p_idx_map = {p: i for i, p in enumerate(prime_list)}
        num_primes = len(prime_list)
        
        # Initialize uncovered counts
        # For each prime p, count uncovered y in 1..n for each residue r
        unc_counts = []
        for p in prime_list:
            counts = [0] * p
            for y in range(1, n + 1):
                counts[y % p] += 1
            unc_counts.append(counts)
        
        used = [False] * num_primes
        assignment = [None] * num_primes
        
        # Precompute indices for fast update
        # For prime p_idx, residue r, the positions are r, r+p, ...
        # We need to update covered and unc_counts for other primes?
        # No, unc_counts is for the *current* state of uncovered numbers.
        # When we cover a number y, we must decrement unc_counts for ALL primes at residue y % p.
        
        def find_next_uncovered(covered, start):
            for i in range(start, n + 1):
                if not covered[i]:
                    return i
            return n + 1

        def dfs(next_unc, remaining_recip_sum):
            if next_unc > n:
                return True
            
            x = next_unc
            # Heuristic: try primes that cover most uncovered numbers
            # Gain for prime p is unc_counts[p_idx][x % p]
            
            candidates = []
            for i, p in enumerate(prime_list):
                if used[i]:
                    continue
                r = x % p
                gain = unc_counts[i][r]
                # If gain is 0, it means x is already covered? 
                # But x is next_uncovered, so it must be uncovered.
                # So gain >= 1.
                candidates.append((gain, i, r))
            
            # Sort by gain descending
            candidates.sort(reverse=True)
            
            # Pruning: Check if remaining reciprocal sum is enough?
            # This is a weak check.
            
            for gain, i, r in candidates:
                p = prime_list[i]
                
                # Check if this assignment is valid (it always is, as p is unused)
                used[i] = True
                assignment[i] = r
                
                # Update covered and unc_counts
                # Mark all y ≡ r (mod p) as covered
                # For each such y, decrement unc_counts[j][y % prime_list[j]] for all j
                new_next_unc = next_unc
                start_y = r if r > 0 else p
                for y in range(start_y, n + 1, p):
                    if not covered[y]:
                        covered[y] = 1
                        # Update unc_counts for all primes
                        for j in range(num_primes):
                            pj = prime_list[j]
                            unc_counts[j][y % pj] -= 1
                        # Update next_unc if y was the blocker
                        if y == next_unc:
                            # We need to find the new next_unc
                            pass # We'll compute it after the loop or inside?
                            # Better to compute after all updates for this prime?
                            # No, next_unc is the *smallest* uncovered.
                            # If we cover next_unc, the new next_unc is > next_unc.
                
                # Find new next_unc
                # Since we only cover numbers >= x, and x was the smallest uncovered,
                # the new next_unc is the smallest y > x that is still uncovered.
                # We can just scan from x+1.
                new_next_unc = find_next_uncovered(covered, x + 1)
                
                # Recurse
                if dfs(new_next_unc, remaining_recip_sum - 1.0/p):
                    return True
                
                # Undo
                used[i] = False
                assignment[i] = None
                start_y = r if r > 0 else p
                for y in range(start_y, n + 1, p):
                    if covered[y]:
                        covered[y] = 0
                        for j in range(num_primes):
                            pj = prime_list[j]
                            unc_counts[j][y % pj] += 1

            return False

        # Initial call
        # remaining_recip_sum is not strictly needed for pruning in this simple version, 
        # but we can pass it.
        return dfs(1, 1.0) if dfs(1, 1.0) else None

    # We need to return the actual assignment
    # The dfs above returns True/False but doesn't return the assignment directly
    # We can modify dfs to return the assignment or use a global variable.
    
    # Let's refactor to return the assignment
    
    def solve_csp_ret(L, prime_list):
        n = L
        covered = bytearray(n + 1)
        num_primes = len(prime_list)
        
        unc_counts = []
        for p in prime_list:
            counts = [0] * p
            for y in range(1, n + 1):
                counts[y % p] += 1
            unc_counts.append(counts)
        
        used = [False] * num_primes
        assignment = [None] * num_primes
        
        def find_next_uncovered(start):
            for i in range(start, n + 1):
                if not covered[i]:
                    return i
            return n + 1

        def dfs(next_unc):
            if next_unc > n:
                return True
            
            x = next_unc
            candidates = []
            for i, p in enumerate(prime_list):
                if used[i]:
                    continue
                r = x % p
                gain = unc_counts[i][r]
                candidates.append((gain, i, r))
            
            candidates.sort(reverse=True)
            
            for gain, i, r in candidates:
                p = prime_list[i]
                used[i] = True
                assignment[i] = r
                
                start_y = r if r > 0 else p
                for y in range(start_y, n + 1, p):
                    if not covered[y]:
                        covered[y] = 1
                        for j in range(num_primes):
                            pj = prime_list[j]
                            unc_counts[j][y % pj] -= 1
                
                new_next_unc = find_next_uncovered(x + 1)
                
                if dfs(new_next_unc):
                    return True
                
                used[i] = False
                assignment[i] = None
                for y in range(start_y, n + 1, p):
                    if covered[y]:
                        covered[y] = 0
                        for j in range(num_primes):
                            pj = prime_list[j]
                            unc_counts[j][y % pj] += 1

            return False

        if dfs(1):
            res = {}
            for i, p in enumerate(prime_list):
                if used[i]:
                    res[p] = assignment[i]
            return res
        return None

    # Try to find max L for each set
    for ps in valid_sets:
        if time.time() > deadline - 10:
            break
            
        # Binary search or linear scan?
        # Let's try a few L values.
        # If L=1000 works, great.
        # If not, try lower.
        # We can binary search L between 1 and 1000.
        
        lo, hi = 1, 1000
        best_l_for_set = 0
        best_assign_for_set = None
        
        while lo <= hi and time.time() < deadline - 5:
            mid = (lo + hi) // 2
            # Try to cover 1..mid
            assign = solve_csp_ret(mid, ps)
            if assign is not None:
                best_l_for_set = mid
                best_assign_for_set = assign
                lo = mid + 1
            else:
                hi = mid - 1
        
        if best_l_for_set > best_L:
            best_L = best_l_for_set
            best_assign = best_assign_for_set
            best_set = ps

    if best_assign is not None:
        result = []
        for p in best_set:
            if p in best_assign:
                result.append([p, best_assign[p]])
        return result
    
    # Fallback
    return [[2, 1], [3, 2]] # Dummy
