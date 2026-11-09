"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
from math import gcd

P = {'n': 256, 'maximum_size': 30}

def construct():
    # We need to construct a set of distinct integers in 1..256 such that:
    # 1. All nonempty subset sums are distinct (no two different subsets have the same sum).
    #    Actually, the rule says: "no two distinct resulting values may divide one another. 
    #    Equal sum values are counted once. Two different subsets with the same sum still make the set invalid"
    #    So, all nonempty subset sums must be distinct.
    # 2. For any two distinct subset sums s1, s2 (where s1 != s2), s1 does not divide s2 and s2 does not divide s1.
    # 3. Maximize the size of the set.
    #
    # A classic way to ensure all subset sums are distinct is to use a superincreasing sequence.
    # Let a1, a2, ..., ak be such that a_i > sum(a_1..a_{i-1}) for all i.
    # Then all subset sums are distinct.
    #
    # Additionally, we need no two subset sums to divide each other.
    # If we use powers of 2: 1, 2, 4, 8, ..., 2^(k-1), then subset sums range from 1 to 2^k - 1.
    # But 1 divides everything, so that fails.
    #
    # Let's think differently. If we use a superincreasing sequence where each term is larger than the sum of all previous,
    # and we choose them such that no subset sum divides another.
    #
    # Consider the sequence: a_i = M^i for some large M. Then subset sums are distinct.
    # But do any divide each other? Unlikely if M is large enough.
    #
    # Let's try: a_i = 2^(i) * p_i where p_i are distinct primes? No, that might not help.
    #
    # Actually, a simpler approach: use a superincreasing sequence where each element is greater than the sum of all previous,
    # and also ensure that no element divides any subset sum formed by smaller elements.
    #
    # Let's try: a_1 = 2, a_2 = 5, a_3 = 13, a_4 = 33, a_5 = 81, ...
    # Check: 
    #   a_1 = 2. Subset sums: {2}
    #   a_2 = 5 > 2. Subset sums: {2, 5, 7}. None divides another? 2 doesn't divide 5 or 7. 5 doesn't divide 2 or 7. OK.
    #   a_3 = 13 > 2+5=7. Subset sums: {2, 5, 7, 13, 15, 17, 19}. 
    #     Check divisibility: 
    #       2 divides 15? No. 2 divides 17? No. 2 divides 19? No.
    #       5 divides 15? Yes! 15 is a subset sum (2+5+... wait, 2+5=7, 5+13=18, 2+13=15). 
    #       So 5 divides 15. Invalid.
    #
    # So we need to be more careful.
    #
    # Let's try: a_1 = 3, a_2 = 7, a_3 = 19, a_4 = 51, ...
    #   a_1 = 3. Sums: {3}
    #   a_2 = 7 > 3. Sums: {3, 7, 10}. 
    #     3 divides 10? No. 3 divides 7? No. 7 divides 3? No. 7 divides 10? No. OK.
    #   a_3 = 19 > 3+7=10. Sums: {3, 7, 10, 19, 22, 26, 29, 36}
    #     Check: 
    #       3 divides 22? No. 3 divides 26? No. 3 divides 29? No. 3 divides 36? Yes! 36 = 3*12. Invalid.
    #
    # The problem is that subset sums can be multiples of smaller elements.
    #
    # Alternative: use a sequence where each element is much larger than the sum of all previous, 
    # and also ensure that no subset sum is a multiple of any smaller element.
    #
    # Let's try: a_i = 2 * (3^(i-1)) 
    #   a_1 = 2, a_2 = 6, a_3 = 18, a_4 = 54, a_5 = 162
    #   a_1 = 2. Sums: {2}
    #   a_2 = 6 > 2. Sums: {2, 6, 8}. 
    #     2 divides 6? Yes! Invalid.
    #
    # How about: a_i = 3^(i) 
    #   a_1 = 3, a_2 = 9, a_3 = 27, a_4 = 81, a_5 = 243
    #   a_1 = 3. Sums: {3}
    #   a_2 = 9 > 3. Sums: {3, 9, 12}. 
    #     3 divides 9? Yes! Invalid.
    #
    # It seems hard to avoid divisibility with simple geometric sequences.
    #
    # Let's think about the condition more carefully. 
    # We need all subset sums to be pairwise non-dividing.
    # This means for any two distinct subset sums s1 < s2, s1 does not divide s2.
    #
    # One way to achieve this is to make all subset sums fall into a range where no number divides another.
    # For example, if all subset sums are in the range (N/2, N], then no two can divide each other 
    # (since if s1 < s2 and s1 divides s2, then s2 >= 2*s1 > N, contradiction).
    #
    # So if we can make all nonempty subset sums lie in (N/2, N] for some N, we are done.
    # But the smallest subset sum is min(a_i), and the largest is sum(a_i).
    # So we need min(a_i) > N/2 and sum(a_i) <= N.
    # This implies min(a_i) > sum(a_i)/2, which is impossible for n >= 2 (since min(a_i) <= sum(a_i)/n <= sum(a_i)/2 for n>=2).
    #
    # So that approach doesn't work directly.
    #
    # Let's try a different approach: use random search with backtracking.
    # We'll try to build a set incrementally, checking the constraints at each step.
    #
    # Given the time limit (60s), we can try a randomized heuristic:
    # 1. Start with an empty set.
    # 2. Try adding numbers from a pool of candidates (e.g., large numbers in 1..256).
    # 3. After adding a number, check if the new set violates the constraints.
    # 4. If valid, keep it; otherwise, try another candidate.
    # 5. Repeat until we can't add more or we reach a good size.
    #
    # To check validity, we need to compute all subset sums and check:
    #   a. All subset sums are distinct.
    #   b. No two subset sums divide each other.
    #
    # For a set of size k, there are 2^k - 1 subset sums. For k=10, that's 1023 sums, which is manageable.
    # For k=15, that's 32767 sums, still manageable.
    # For k=20, that's 1 million sums, might be slow but possibly OK in 60s if optimized.
    #
    # Let's implement a helper to check if a set is valid.
    
    def is_valid_set(s):
        """Check if a set of integers satisfies the constraints."""
        k = len(s)
        if k == 0:
            return True
        # Compute all nonempty subset sums
        subset_sums = set()
        # Use bitmasks for small k
        for mask in range(1, 1 << k):
            total = 0
            for i in range(k):
                if mask & (1 << i):
                    total += s[i]
            if total in subset_sums:
                return False  # Duplicate subset sum
            subset_sums.add(total)
        # Check divisibility
        sums_list = list(subset_sums)
        n_sums = len(sums_list)
        for i in range(n_sums):
            for j in range(i + 1, n_sums):
                a, b = sums_list[i], sums_list[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    return False
        return True
    
    # We'll use a greedy randomized approach:
    # Start with a large number, then try to add smaller numbers that don't break the constraints.
    # Actually, it's better to start with small numbers and add larger ones, ensuring superincreasing property.
    
    # Let's try a different strategy: use a superincreasing sequence but with careful choice to avoid divisibility.
    # We'll use: a_1 = x_1, a_2 = x_2, ..., where x_{i+1} > sum(x_1..x_i) and x_{i+1} is chosen to avoid divisibility issues.
    
    # For simplicity, let's try a fixed sequence that we know works for small k, and extend it.
    # From literature or experimentation, sequences like:
    #   [3, 7, 19, 51, 135, 355, ...] where a_{i+1} = 3*a_i + 2 or similar.
    
    # Let's try: a_1 = 3, a_{i+1} = 3*a_i + 2
    #   a_1 = 3
    #   a_2 = 11
    #   a_3 = 35
    #   a_4 = 107
    #   a_5 = 323
    #   a_6 = 971 > 256, so max 5 elements.
    
    # Check validity for [3, 11, 35, 107, 323]:
    # This is too slow to do by hand. Let's just run the is_valid_set function.
    
    # Actually, let's just use a randomized search with backtracking for k up to 10-12.
    
    # Pool of candidates: numbers in 1..256
    # We'll try to build a set of size k by randomly selecting candidates and checking validity.
    
    # To speed up, we can use the fact that if we have a superincreasing sequence, subset sums are distinct.
    # So we only need to check divisibility.
    
    # Let's try: start with a random number, then repeatedly add a number larger than the current sum,
    # and check if the new set is valid.
    
    best_set = []
    
    # Try multiple random restarts
    for _ in range(1000):
        current_set = []
        # Start with a random number in [10, 100]
        current_set.append(random.randint(10, 100))
        
        # Try to add more elements
        while True:
            # Current sum
            curr_sum = sum(current_set)
            # Next element must be > curr_sum to ensure superincreasing (and thus distinct subset sums)
            # And must be <= 256
            if curr_sum >= 256:
                break
            # Try a random candidate > curr_sum
            added = False
            for _ in range(100):
                cand = random.randint(curr_sum + 1, 256)
                new_set = current_set + [cand]
                if is_valid_set(new_set):
                    current_set = new_set
                    added = True
                    break
            if not added:
                break
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
    
    # Ensure best_set is not empty
    if not best_set:
        best_set = [31, 32, 34]  # Fallback to original
    
    # Convert to list of Python ints
    return list(best_set)

# EVOLVE-BLOCK-END
