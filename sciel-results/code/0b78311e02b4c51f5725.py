import time
import random
import sys
import bisect

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 82  # Keep safety margin for evaluator overhead
    
    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            sums = sums | {s + x for s in sums}
        return sums
    
    def check_divisibility_fast(sums_set):
        """
        Fast check: no two distinct sums in sums_set divide each other.
        Uses sorted list and for each element checks multiples via membership.
        """
        if len(sums_set) <= 1:
            return True
        sorted_sums = sorted(sums_set)
        max_val = sorted_sums[-1]
        for u in sorted_sums:
            if u == 0:
                continue
            for mult in range(2, max_val // u + 1):
                if u * mult in sums_set:
                    return False
        return True

    def can_add_fast(x, current_sums):
        """
        Fast check: can we add x? Returns (bool, new_sums) or (False, None).
        """
        new_sums = {x + s for s in current_sums}
        
        # Check distinctness
        if new_sums & current_sums:
            return False, None
        
        # Combined set
        combined = current_sums | new_sums
        non_empty = combined - {0}
        
        # Check divisibility in combined set
        if not check_divisibility_fast(non_empty):
            return False, None
        
        return True, combined
    
    def greedy_build(candidate_order, start_set=None, start_sums=None):
        """Greedy build using a given order of candidates."""
        if start_set is not None:
            S = list(start_set)
            current_sums = set(start_sums)
        else:
            S = []
            current_sums = {0}
        
        existing = set(S)
        for x in candidate_order:
            if time.time() - start_time > time_limit:
                break
            if x in existing:
                continue
            valid, new_sums = can_add_fast(x, current_sums)
            if valid:
                S.append(x)
                existing.add(x)
                current_sums = new_sums
        return S, current_sums
    
    def local_search(S, sums_set, max_iter=300):
        """Try to improve the set by removing one element and adding a different one."""
        S_copy = list(S)
        sums_copy = set(sums_set)
        improved = True
        iter_count = 0
        
        while improved and time.time() - start_time < time_limit and iter_count < max_iter:
            improved = False
            iter_count += 1
            S_work = list(S_copy)
            sums_work = set(sums_copy)
            
            # Shuffle removal order for diversity
            remove_order = list(range(len(S_work)))
            random.shuffle(remove_order)
            
            for remove_idx in remove_order:
                if time.time() - start_time > time_limit:
                    return S_copy, sums_copy
                
                x_removed = S_work[remove_idx]
                S_no = S_work[:remove_idx] + S_work[remove_idx+1:]
                sums_no = compute_sums(S_no)
                in_set = set(S_no)
                
                # Try adding each candidate not in S_no
                # Try larger numbers first (they tend to have fewer divisibility conflicts)
                for x in range(256, 0, -1):
                    if x in in_set:
                        continue
                    valid, new_sums = can_add_fast(x, sums_no)
                    if valid:
                        S_new = S_no + [x]
                        S_copy = S_new
                        sums_copy = new_sums
                        improved = True
                        break
                if improved:
                    break
        
        return S_copy, sums_copy
    
    def count_divisors(n):
        """Count number of divisors of n."""
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    # Precompute divisor counts for ordering
    div_counts = {n: count_divisors(n) for n in range(1, 257)}
    
    # Strategy: Multiple random restarts with different heuristics
    best_set = []
    best_size = 0
    best_sums = {0}
    
    random.seed(12345)
    
    # Precompute primes for candidate ordering
    primes = [p for p in range(129, 257) if all(p % i != 0 for i in range(2, int(p**0.5)+1))]
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart < 20:
            # Start with numbers having few divisors (primes, prime powers)
            candidates = list(range(1, 257))
            candidates.sort(key=lambda x: (div_counts[x], -x))
            random.shuffle(candidates[:30])
        elif restart < 40:
            # Start with larger numbers first, with some randomization
            candidates = list(range(256, 0, -1))
            random.shuffle(candidates[:50])
        elif restart < 60:
            # Random order
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif restart < 80:
            # Weighted toward larger numbers with noise
            candidates = list(range(1, 257))
            candidates.sort(key=lambda x: -x + random.gauss(0, 40))
        else:
            # Mixed: large primes first, then sorted by divisor count
            remaining = [x for x in range(1, 257) if x not in primes]
            remaining.sort(key=lambda x: (div_counts[x], -x))
            candidates = primes + remaining
            random.shuffle(candidates[10:])
        
        S, sums = greedy_build(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            best_sums = sums
    
    # Local search: try to improve best_set
    if best_size >= 2:
        S_ls, sums_ls = local_search(best_set, best_sums)
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_set = S_ls[:]
            best_sums = sums_ls
    
    # Additional local search rounds with perturbation
    for round_num in range(25):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb: remove 1-3 random elements and try to refill
        if len(best_set) >= 3:
            S_pert = list(best_set)
            random.shuffle(S_pert)
            num_remove = min(random.randint(1, 3), len(S_pert) - 1)
            S_pert = S_pert[num_remove:]
            sums_pert = compute_sums(S_pert)
            
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            S_new, sums_new = greedy_build(candidates, S_pert, sums_pert)
            
            if len(S_new) > best_size:
                best_size = len(S_new)
                best_set = S_new[:]
                best_sums = sums_new
    
    # Final local search
    if best_size >= 2:
        S_ls2, sums_ls2 = local_search(best_set, best_sums)
        if len(S_ls2) > best_size:
            best_size = len(S_ls2)
            best_set = S_ls2[:]
            best_sums = sums_ls2
    
    # One more round of greedy extension from best
    if time.time() - start_time < time_limit - 5:
        candidates = list(range(256, 0, -1))
        random.shuffle(candidates)
        S_ext, sums_ext = greedy_build(candidates, best_set, best_sums)
        if len(S_ext) > best_size:
            best_size = len(S_ext)
            best_set = S_ext[:]
            best_sums = sums_ext
        S_ls3, sums_ls3 = local_search(best_set, best_sums)
        if len(S_ls3) > best_size:
            best_size = len(S_ls3)
            best_set = S_ls3[:]
            best_sums = sums_ls3
    
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty = final_sums - {0}
        if len(non_empty) != (1 << len(best_set)) - 1:
            best_set = []
        elif not check_divisibility_fast(non_empty):
            best_set = []
    
    # Ensure we return a valid set
    if not best_set or len(best_set) < 1:
        return [256]
    
    # Convert to pure Python ints
    return [int(x) for x in best_set]

if __name__ == "__main__":
    res = construct()
    print(res)
