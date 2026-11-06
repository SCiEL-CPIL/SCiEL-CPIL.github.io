"""Combined program: systematic enumeration + local search for pairwise-coprime sumsets."""
import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Precompute small primes up to 256 (max sum = 128+128)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
                    113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
                    181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
                    251]
    
    # Precompute prime mask for each number 1..256
    # mask[n] = bitmask where bit i is set if small_primes[i] divides n
    prime_mask = [0] * 257
    for n in range(2, 257):
        m = 0
        for i, p in enumerate(small_primes):
            if p > n:
                break
            if n % p == 0:
                m |= (1 << i)
        prime_mask[n] = m
    
    def pairwise_coprime(sums_list):
        """Check if all values in sums_list are pairwise coprime using bitmask AND."""
        n = len(sums_list)
        for i in range(n):
            mi = prime_mask[sums_list[i]]
            for j in range(i + 1, n):
                if mi & prime_mask[sums_list[j]]:
                    return False
        return True
    
    def evaluate(A, B):
        """Return (cardinality, is_valid) for a pair of 3-element lists."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        card = len(distinct)
        if card < 5:
            return card, False
        return card, pairwise_coprime(distinct)
    
    odds = [i for i in range(1, 129) if i % 2 != 0]
    evens = [i for i in range(1, 129) if i % 2 == 0]
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_card = 5
    
    # --- Phase 1: Systematic enumeration of all (odd_triplet, even_triplet) ---
    # 64 odds -> C(64,3) = 41664 trios; 64 evens -> C(64,3) = 41664 trios
    # Total pairs: ~1.7 billion -- too many for full enumeration.
    # Instead: enumerate odd trios, and for each, sample/check a subset of even trios.
    # Better: enumerate all odd trios, and for each, do a quick scan of promising even trios.
    
    # Actually, let's be smarter: enumerate all odd trios (41664), and for each,
    # check all even trios (41664). That's 41664*41664 ≈ 1.74 billion -- still too many.
    
    # Alternative: enumerate all 9-sum configurations where sums are forced to be coprime.
    # Let's do: for each odd trio, find even trios that give 9 distinct coprime sums.
    # We can precompute: for a given odd trio O = {o1,o2,o3}, the 9 sums with even trio E
    # are {o_i + e_j}. We need all 9 distinct and pairwise coprime.
    
    # Let's do a two-phase approach:
    # Phase 1a: For each odd trio, check a random subset of even trios (fast).
    # Phase 1b: Systematic scan with early termination.
    
    all_odd_trios = list(combinations(odds, 3))
    all_even_trios = list(combinations(evens, 3))
    
    # Phase 1a: Random sampling of even trios for each odd trio
    # Limit: check at most ~200 even trios per odd trio in random phase
    max_random_per_trio = 200
    checked_count = 0
    for o_trio in all_odd_trios:
        if time.time() > deadline - 20:
            break
        o1, o2, o3 = o_trio
        # Quick pre-check: the 3 base sums with any even must be odd
        # Sample even trios
        for _ in range(max_random_per_trio):
            e_trio = random.sample(evens, 3)
            e1, e2, e3 = e_trio
            sums = [o1+e1, o1+e2, o1+e3,
                    o2+e1, o2+e2, o2+e3,
                    o3+e1, o3+e2, o3+e3]
            if len(set(sums)) < 9:
                continue
            if pairwise_coprime(sums):
                return [list(o_trio), list(e_trio)]
            checked_count += 1
    
    # Phase 1b: Systematic scan -- for each odd trio, scan all even trios
    # But 41664 * 41664 is too much. Let's limit: check all even trios for the first
    # N odd trios, then randomize.
    
    # Actually, let's do a smarter systematic approach:
    # For each odd trio, iterate through all even trios but with a time check every 1000.
    # 41664 odd trios * 41664 even trios = 1.74B. Even at 10M/s that's 174s. Too slow.
    
    # Let's cap: check all even trios for first 500 odd trios systematically,
    # then random for the rest.
    
    sys_odd_limit = 500
    for idx, o_trio in enumerate(all_odd_trios):
        if time.time() > deadline - 15:
            break
        o1, o2, o3 = o_trio
        if idx < sys_odd_limit:
            # Systematic scan of all even trios
            for e_trio in all_even_trios:
                e1, e2, e3 = e_trio
                sums = [o1+e1, o1+e2, o1+e3,
                        o2+e1, o2+e2, o2+e3,
                        o3+e1, o3+e2, o3+e3]
                if len(set(sums)) < 9:
                    continue
                if pairwise_coprime(sums):
                    return [list(o_trio), list(e_trio)]
        else:
            # Random sampling
            for _ in range(100):
                e_trio = random.sample(evens, 3)
                e1, e2, e3 = e_trio
                sums = [o1+e1, o1+e2, o1+e3,
                        o2+e1, o2+e2, o2+e3,
                        o3+e1, o3+e2, o3+e3]
                if len(set(sums)) < 9:
                    continue
                if pairwise_coprime(sums):
                    return [list(o_trio), list(e_trio)]
    
    # --- Phase 2: Local search to find high-cardinality (6-8) coprime sumsets ---
    # If we couldn't find 9, try to maximize cardinality with local search.
    
    def local_search(A, B, remaining_time):
        """Hill-climb to maximize cardinality of pairwise-coprime distinct sums."""
        current_A = list(A)
        current_B = list(B)
        curr_card, curr_valid = evaluate(current_A, current_B)
        if not curr_valid:
            # Try to fix: find a valid starting point
            return list(A), list(B), curr_card
        
        all_nums = list(range(1, 129))
        no_improve = 0
        max_no_improve = 15
        
        while time.time() < start_time + remaining_time:
            if curr_card == 9:
                return list(current_A), list(current_B), curr_card
            
            improved = False
            
            # Try improving A
            for _ in range(80):
                idx = random.randint(0, 2)
                old_val = current_A[idx]
                new_val = random.choice(all_nums)
                if new_val == old_val:
                    continue
                test_A = list(current_A)
                test_A[idx] = new_val
                if len(set(test_A)) != 3:
                    continue
                card, valid = evaluate(test_A, current_B)
                if valid and card > curr_card:
                    current_A = test_A
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    break
                elif valid and card == curr_card and random.random() < 0.05:
                    current_A = test_A
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    break
            
            # Try improving B
            for _ in range(80):
                idx = random.randint(0, 2)
                old_val = current_B[idx]
                new_val = random.choice(all_nums)
                if new_val == old_val:
                    continue
                test_B = list(current_B)
                test_B[idx] = new_val
                if len(set(test_B)) != 3:
                    continue
                card, valid = evaluate(current_A, test_B)
                if valid and card > curr_card:
                    current_B = test_B
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    break
                elif valid and card == curr_card and random.random() < 0.05:
                    current_B = test_B
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    break
            
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
        
        return list(current_A), list(current_B), curr_card
    
    # Try local search from seed and from several structured starts
    structured_starts = [
        ([1, 3, 5], [2, 4, 6]),
        ([1, 5, 7], [2, 4, 8]),
        ([1, 7, 11], [2, 6, 10]),
        ([3, 5, 9], [2, 4, 8]),
        ([1, 3, 7], [4, 8, 12]),
        ([1, 9, 15], [2, 6, 14]),
        ([3, 7, 11], [4, 10, 16]),
        ([1, 5, 13], [2, 8, 14]),
    ]
    
    for sA, sB in structured_starts:
        if time.time() > deadline - 5:
            break
        card, valid = evaluate(sA, sB)
        if valid and card > best_card:
            best_A, best_B = list(sA), list(sB)
            best_card = card
    
    # Run local search from best and from a few random valid starts
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        la, lb, card = local_search(best_A, best_B, deadline - start_time)
        if card > best_card:
            best_A, best_B = la, lb
            best_card = card
    
    # Also try from random valid configurations
    for _ in range(3):
        if time.time() > deadline - 3:
            break
        if random.random() < 0.5:
            A = random.sample(odds, 3)
            B = random.sample(evens, 3)
        else:
            A = random.sample(evens, 3)
            B = random.sample(odds, 3)
        card, valid = evaluate(A, B)
        if valid and card > best_card:
            best_A, best_B = A, B
            best_card = card
            la, lb, card2 = local_search(best_A, best_B, deadline - start_time)
            if card2 > best_card:
                best_A, best_B = la, lb
                best_card = card2
    
    return [best_A, best_B]
