"""Construct large sets with divisibility-incomparable subset sums."""
import time
import random
import math
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    # Helper: get all non-empty subset sums, return None if duplicates exist
    def get_subset_sums(elements):
        sums = set()
        for x in elements:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums or ns in new_sums:
                    return None  # duplicate sum detected
                new_sums.add(ns)
            if x in sums or x in new_sums:
                return None
            new_sums.add(x)
            sums = sums | new_sums
        return sums

    # Helper: check if all subset sums are divisibility-incomparable
    def check_divisibility(sums):
        if sums is None:
            return False
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                if b % a == 0:
                    return False
        return True

    # Helper: full validity check
    def is_valid(elements):
        sums = get_subset_sums(elements)
        if sums is None:
            return False
        return check_divisibility(sums)

    # Helper: check if adding cand to current elements maintains validity
    # Uses incremental approach: track existing sums, check new sums against old and each other
    def check_addition(elements, cand):
        # Get current sums
        sums = get_subset_sums(elements)
        if sums is None:
            return False, None
        # New sums = old sums + cand
        new_sums = set()
        for s in sums:
            ns = s + cand
            if ns in sums or ns in new_sums:
                return False, None
            new_sums.add(ns)
        if cand in sums or cand in new_sums:
            return False, None
        new_sums.add(cand)
        all_sums = sums | new_sums
        if not check_divisibility(all_sums):
            return False, None
        return True, all_sums

    # Phase 1: Structured construction using prime powers
    # Use primes: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97
    # Powers: p^1, p^2, etc. where p^k <= 256
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # For each prime, generate powers up to 256
    prime_powers = {}
    for p in primes:
        powers = []
        pk = p
        while pk <= 256:
            powers.append(pk)
            pk *= p
        prime_powers[p] = powers
    
    # Start with base primes (one from each prime)
    base_elements = [p for p in primes]
    
    # Try to build a valid set starting from base
    current = []
    current_sums = None
    
    # Sort candidates: try primes first (smaller, more flexible)
    all_candidates = []
    for p in primes:
        for pk in prime_powers[p]:
            all_candidates.append(pk)
    # Deduplicate (2^k might equal other values, but prime powers of distinct primes are unique)
    all_candidates = list(set(all_candidates))
    all_candidates.sort()
    
    # Greedy: add candidates if valid
    for cand in all_candidates:
        if time.time() > deadline - 20.0:
            break
        if cand in current:
            continue
        valid, new_sums = check_addition(current, cand)
        if valid:
            current.append(cand)
            current_sums = new_sums
    
    best_set = list(current)
    best_size = len(current)
    
    # Phase 2: Random restarts with greedy construction
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 25.0:
        restarts += 1
        # Random subset of candidates, shuffled
        shuffled = list(all_candidates)
        random.shuffle(shuffled)
        
        cur = []
        cur_sums = None
        for cand in shuffled:
            if cand in cur:
                continue
            if cur_sums is None:
                cur_sums = get_subset_sums(cur)
                if cur_sums is None:
                    break
            valid, new_sums = check_addition(cur, cand)
            if valid:
                cur.append(cand)
                cur_sums = new_sums
        
        if len(cur) > best_size:
            best_size = len(cur)
            best_set = list(cur)
        
        if best_size >= 12:  # good enough, move to local search
            break
    
    # Phase 3: Local search - try to add more elements to best_set
    def local_search(elements, time_limit):
        elements = list(elements)
        sums = get_subset_sums(elements)
        if sums is None:
            return elements
        
        # Get candidates not in elements
        in_set = set(elements)
        candidates = [c for c in all_candidates if c not in in_set]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for cand in candidates:
                if cand in set(elements):
                    continue
                valid, new_sums = check_addition(elements, cand)
                if valid:
                    elements.append(cand)
                    sums = new_sums
                    improved = True
                    # Remove from candidates list
                    if cand in candidates:
                        candidates.remove(cand)
                    break
        return elements
    
    if best_set and time.time() < deadline - 10.0:
        improved = local_search(best_set, deadline - 5.0)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = list(improved)
    
    # Phase 4: Try swap-based local search
    if best_set and time.time() < deadline - 10.0:
        # Try replacing one element with another to enable adding more
        in_set = set(best_set)
        candidates = [c for c in all_candidates if c not in in_set]
        
        for i in range(len(best_set)):
            if time.time() > deadline - 10.0:
                break
            for cand in candidates:
                if cand in in_set:
                    continue
                # Try swapping best_set[i] with cand
                test = list(best_set)
                test[i] = cand
                test_sums = get_subset_sums(test)
                if test_sums is None:
                    continue
                if check_divisibility(test_sums):
                    # Now try to add more
                    test_elements = list(test)
                    test_sums_cur = test_sums
                    for extra in candidates:
                        if extra in set(test_elements):
                            continue
                        valid, new_sums = check_addition(test_elements, extra)
                        if valid:
                            test_elements.append(extra)
                            test_sums_cur = new_sums
                    if len(test_elements) > best_size:
                        best_size = len(test_elements)
                        best_set = list(test_elements)
                        break
    
    # Final verification
    if best_set:
        final_sums = get_subset_sums(best_set)
        if final_sums is not None and check_divisibility(final_sums):
            # Ensure all are Python int, in range 1..256, distinct
            result = [int(x) for x in best_set]
            if all(1 <= x <= 256 for x in result) and len(set(result)) == len(result):
                return result
    
    # Fallback
    return [31, 32, 34]
