"""Program for the suite task; uses Sidon-based construction with greedy extension."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    n = 512
    max_card = 48
    
    # Step 1: Generate a dense Sidon (B2) sequence in [1, 512].
    # A Sidon sequence has distinct pairwise sums (i<j). This guarantees no 3-term AP (k=2 mean-free).
    # We use a greedy approach to pick the smallest possible numbers that maintain the Sidon property.
    
    S = []
    used_sums = set()
    
    for x in range(1, n + 1):
        valid_sidon = True
        for i in range(len(S)):
            s_val = S[i]
            # Check if x + s_val is already in used_sums
            # Also need to check if x is a "difference" that creates a collision?
            # Sidon condition: all sums S[i] + S[j] for i < j are distinct.
            # When adding x, new sums are x + S[i] for all i.
            # These must be distinct from each other (they are, since S[i] are distinct)
            # and distinct from all existing sums.
            if (x + s_val) in used_sums:
                valid_sidon = False
                break
        if valid_sidon:
            # Add new sums to used_sums
            for i in range(len(S)):
                used_sums.add(x + S[i])
            S.append(x)
            if len(S) >= max_card:
                break
    
    # If we didn't get enough, we might need to continue, but Sidon sequences are sparse.
    # Max Sidon size in [1, N] is approx sqrt(N). sqrt(512) ~ 22.
    # So we will likely get around 22-25 elements. We need 48.
    # The Sidon property is a strong start, but we need to extend beyond Sidon.
    # The task allows any set where no element is the mean of a subset of OTHERS.
    # Sidon prevents k=2 violations. It does not prevent k>=3.
    # So we can greedily add more numbers, checking for mean-free property for all k.
    
    # Let's refine the greedy extension.
    # Current S is a Sidon set.
    # We want to add more elements to reach 48.
    # We will iterate x from 1 to 512 again (or from where we left off? No, restart to fill gaps).
    # Actually, the Sidon greedy above picked the *smallest* possible numbers.
    # This leaves many gaps. We can try to fill those gaps with numbers that don't violate the mean-free condition.
    
    # Let's restart the construction with a combined check:
    # 1. Maintain Sidon property? No, we don't need to maintain Sidon property for the final set,
    #    only the mean-free property. The Sidon property was just a heuristic to avoid k=2 violations.
    #    However, if we drop the Sidon constraint during extension, we might introduce k=2 violations.
    #    So, we should check for k=2 violations explicitly, or maintain a structure that helps.
    
    # Better approach:
    # Greedily add numbers 1..512.
    # For each candidate x, check if adding it to current S creates any mean-free violation.
    # Violation types:
    # A. x is the mean of a subset of S (size >= 2).
    # B. Some a in S is the mean of a subset of S U {x} excluding a (size >= 2).
    
    # We will implement a fast check using bitsets for subset sums of differences.
    # To speed up, we can precompute subset sum bitsets for S?
    # But S changes.
    
    # Let's use the incremental check derived earlier:
    # 1. Check if 0 is a subset sum of {s - x for s in S} (non-empty).
    # 2. For each a in S, check if (a - x) is a subset sum of {s - a for s in S, s != a} (non-empty if target=0, else any).
    
    # To make this fast, we can use Python integers as bitsets for the subset sums.
    # The range of sums is bounded.
    # Max diff is 511. Max size 48. Range ~ [-25000, 25000].
    # Offset by 25000. Bitset size ~ 50000 bits = 6250 bytes.
    # This is very fast.
    
    S = []
    
    # Helper to compute subset sum bitset for a list of integers.
    # Returns a set of possible sums? Or a bitset?
    # Since we need to check specific targets, and the number of elements is small (<50),
    # a set of sums might be sufficient and simpler to implement correctly with negative numbers.
    # Let's estimate the number of sums.
    # If we have 40 numbers, 2^40 is too big for a set if they are all distinct.
    # But the sums are bounded.
    # Actually, the number of distinct subset sums can be large.
    # However, we are checking for *specific* targets.
    # Maybe we can use a DP that only tracks reachable sums?
    
    # Let's try a different optimization:
    # Precompute for the current S, for each a in S, the set of subset sums of {s - a for s in S, s != a}.
    # This is expensive to update.
    
    # Alternative:
    # Since n=512 is small, we can just try all x from 1 to 512.
    # For each x, we do the check.
    # The check involves iterating over a in S (up to 48) and computing subset sums of S\{a} (up to 47 elements).
    # Computing subset sums of 47 elements by brute force is 2^47, which is too slow.
    # We MUST use a bitset or a bounded set of sums.
    
    # Let's use a bitset represented by a Python integer.
    # We shift the sums by OFFSET to handle negatives.
    # OFFSET = 25000.
    # Bit i represents if sum = i - OFFSET is possible.
    
    OFFSET = 25000
    MASK = (1 << (2 * OFFSET + 1)) - 1  # This is huge, 1 << 50001. Python handles big ints, but it's slow.
    # Actually, we don't need the full mask if we just check bits.
    
    def subset_sums_bitset(nums):
        """Returns a set of possible sums? No, let's return a boolean array or a set of ints."""
        # Given the constraints, a set of ints might be okay if the number of distinct sums is small.
        # For random-ish numbers, the sums spread out.
        # But with 47 numbers, the number of distinct sums can be up to 2^47? No, bounded by range.
        # Range is ~50000. So at most 50000 distinct sums.
        # A set of 50000 ints is fine.
        
        sums = {0}
        for num in nums:
            new_sums = set()
            for s in sums:
                new_sums.add(s + num)
            sums |= new_sums
        return sums
    
    S = []
    
    for x in range(1, n + 1):
        valid = True
        
        # Check 1: Is x the mean of any subset of S (size >= 2)?
        # Equivalent to: 0 is a subset sum of {s - x for s in S} with size >= 1 (since s!=x).
        if S:
            diffs = [s - x for s in S]
            sums_set = subset_sums_bitset(diffs)
            if 0 in sums_set:
                # We must ensure the subset is non-empty.
                # If 0 is in sums_set, it could be from empty set.
                # Since no d is 0, any non-empty subset summing to 0 has size >= 2.
                # So we just need to check if 0 is in sums_set AND not only from empty set.
                # How to check?
                # If we remove 0 from the initial set, does 0 still appear?
                # Or simpler: check if there is a non-empty subset summing to 0.
                # We can modify subset_sums_bitset to return non-empty sums?
                # Or just check: if any d == 0, invalid. Else if 0 in sums_set, valid?
                # Wait. If sums_set contains 0, it's either empty or non-empty.
                # If non-empty, it's a violation.
                # If empty, it's not.
                # So we need to know if 0 is achievable by a non-empty subset.
                
                # Let's compute non-empty sums.
                non_empty_sums = set()
                for d in diffs:
                    non_empty_sums.add(d)
                    # This loop is O(N * |sums|)
                    for s in list(non_empty_sums):
                        non_empty_sums.add(s + d)
                if 0 in non_empty_sums:
                    valid = False
        
        if valid:
            # Check 2: For each a in S, is a the mean of a subset of S U {x} excluding a?
            for a in S:
                diffs_a = [s - a for s in S if s != a]
                target = a - x
                if not diffs_a:
                    continue
                
                sums_a = subset_sums_bitset(diffs_a)
                
                if target in sums_a:
                    # Check if the subset is non-empty if target is 0.
                    if target == 0:
                        # Need non-empty subset summing to 0.
                        non_empty_sums_a = set()
                        for d in diffs_a:
                            non_empty_sums_a.add(d)
                            for s in list(non_empty_sums_a):
                                non_empty_sums_a.add(s + d)
                        if 0 in non_empty_sums_a:
                            valid = False
                            break
                    else:
                        # Any subset summing to target is non-empty.
                        valid = False
                        break
        
        if valid:
            S.append(x)
            if len(S) >= max_card:
                break
    
    return S
# EVOLVE-BLOCK-END
