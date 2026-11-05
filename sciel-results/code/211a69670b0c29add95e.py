import time
import math
from itertools import combinations
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill

    N = 12
    MAXV = 10000
    MAXSUM = 2 * MAXV  # 20000

    # Precompute prime factors for all numbers up to 20000 using sieve
    def build_pf(limit):
        pf = [set() for _ in range(limit + 1)]
        for p in range(2, limit + 1):
            if not pf[p]:  # p is prime if no factor found yet
                for m in range(p, limit + 1, p):
                    pf[m].add(p)
        return pf

    pf = build_pf(MAXSUM)

    def get_primes_for_sums(nums):
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(pf[nums[i] + nums[j]])
        return all_primes

    # Start with a reasonable baseline
    baseline = list(range(1, 13))
    best_set = list(baseline)
    best_count = len(get_primes_for_sums(baseline))

    def smooth_numbers(primes, max_val):
        """Generate all numbers with only prime factors from 'primes' up to max_val."""
        # Use a boolean array for faster membership testing
        smooth = bytearray(max_val + 1)
        smooth[1] = 1
        for p in primes:
            for x in range(1, max_val + 1):
                if smooth[x] and x * p <= max_val:
                    smooth[x * p] = 1
                elif not smooth[x] and x % p == 0:
                    # x has a factor not in primes, skip
                    pass
        # More efficient: iterative multiplication
        smooth = bytearray(max_val + 1)
        smooth[1] = 1
        result = [1]
        for p in primes:
            new_entries = []
            for x in result:
                v = x
                while v * p <= max_val:
                    v *= p
                    if not smooth[v]:
                        smooth[v] = 1
                        new_entries.append(v)
            result.extend(new_entries)
        return smooth

    def find_smooth_solution(prime_set, deadline):
        """Find 12 distinct numbers <= MAXV whose pairwise sums are all S-smooth."""
        ps = list(prime_set)
        smooth_sums = smooth_numbers(ps, MAXSUM)
        
        # Candidate pool: numbers 1..800 (expanded for better coverage)
        cand_range = list(range(1, 801))
        
        # Precompute compatibility using bitmask for speed
        # compat[x] will be a set of indices of compatible y's
        compat = [set() for _ in range(len(cand_range))]
        for i in range(len(cand_range)):
            x = cand_range[i]
            for j in range(len(cand_range)):
                if i != j:
                    y = cand_range[j]
                    if x + y <= MAXSUM and smooth_sums[x + y]:
                        compat[i].add(j)
        
        chosen = []
        used = [False] * len(cand_range)
        
        def backtrack(start_idx):
            if time.time() > deadline:
                return False
            if len(chosen) == N:
                return True
            for i in range(start_idx, len(cand_range)):
                if used[i]:
                    continue
                x = cand_range[i]
                # Check x compatible with all chosen
                ok = True
                for c_idx in chosen:
                    if i not in compat[c_idx]:
                        ok = False
                        break
                if not ok:
                    continue
                chosen.append(i)
                used[i] = True
                if backtrack(i + 1):
                    return True
                chosen.pop()
                used[i] = False
            return False
        
        if backtrack(0):
            return [cand_range[i] for i in chosen]
        return None

    # Try 2-prime sets to find a 2-prime solution (optimal)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    two_prime_sets = list(combinations(small_primes, 2))
    
    # Prioritize pairs with smaller primes (more likely to yield smooth sums)
    two_prime_sets.sort(key=lambda ps: ps[0] + ps[1])
    
    for ps in two_prime_sets:
        if time.time() > deadline:
            break
        sol = find_smooth_solution(ps, deadline)
        if sol is not None:
            c = len(get_primes_for_sums(sol))
            if c < best_count:
                best_count = c
                best_set = sol
            if best_count <= 2:
                break

    # If no 2-prime solution found, try 3-prime sets
    if best_count > 2 and time.time() < deadline:
        three_prime_sets = list(combinations(small_primes[:10], 3))
        three_prime_sets.sort(key=lambda ps: ps[0] + ps[1] + ps[2])
        for ps in three_prime_sets:
            if time.time() > deadline:
                break
            sol = find_smooth_solution(ps, deadline)
            if sol is not None:
                c = len(get_primes_for_sums(sol))
                if c < best_count:
                    best_count = c
                    best_set = sol
                if best_count <= 3:
                    break

    # Local search to improve if not already optimal
    if best_count > 2 and time.time() < deadline - 5:
        def local_search(start, max_time):
            current = list(start)
            current_primes = get_primes_for_sums(current)
            current_count = len(current_primes)
            
            # Precompute pair prime contributions for faster updates
            def compute_pair_primes(nums, idx):
                """Get primes from pairs involving index idx."""
                result = set()
                for j in range(len(nums)):
                    if j != idx:
                        result.update(pf[nums[idx] + nums[j]])
                return result
            
            improved = True
            while improved and time.time() < max_time:
                improved = False
                for idx in range(12):
                    if time.time() > max_time:
                        break
                    old_val = current[idx]
                    old_pair_primes = compute_pair_primes(current, idx)
                    
                    # Try replacing with values that have fewer prime factors in sums
                    # First try values close to old_val
                    tried = set()
                    for offset in range(1, 300):
                        for new_val in [old_val + offset, old_val - offset]:
                            if new_val < 1 or new_val > 10000:
                                continue
                            if new_val == old_val or new_val in tried:
                                continue
                            tried.add(new_val)
                            if any(new_val == current[j] for j in range(12) if j != idx):
                                continue
                            
                            new_pair_primes = compute_pair_primes(current, idx) if False else set()
                            for j in range(12):
                                if j != idx:
                                    new_pair_primes.update(pf[new_val + current[j]])
                            
                            remaining = current_primes - old_pair_primes
                            test_primes = remaining | new_pair_primes
                            
                            if len(test_primes) < current_count:
                                current[idx] = new_val
                                current_primes = test_primes
                                current_count = len(test_primes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            return current, current_count
        
        best_set, best_count = local_search(best_set, deadline - 5)

    # Final validation
    assert len(set(best_set)) == 12, f"Not 12 distinct: {best_set}"
    assert all(1 <= x <= 10000 for x in best_set), f"Out of range: {best_set}"

    return best_set
