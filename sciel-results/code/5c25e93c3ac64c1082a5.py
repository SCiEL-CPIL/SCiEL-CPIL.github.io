"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find 12 distinct positive integers <= 1048576 such that all 4096 subset sums
    are distinct, minimizing the largest integer.
    
    Strategy:
    1. Use a greedy construction with backtracking to find a valid set.
    2. The key insight is that we need a "superincreasing" like sequence but not necessarily powers of 2.
    3. Actually, the minimal maximum for n=12 distinct subset sums is related to the concept of 
       "Sidon sequences" or more specifically, we need a set where all subset sums are unique.
       
    The greedy approach: start with small numbers and try to add the smallest possible next number
    that maintains the distinct subset sum property.
    
    For n elements, we need 2^n distinct subset sums. The smallest possible maximum element
    for n=12 is known to be around 2^11 = 2048 for powers of 2, but we can do better.
    
    Actually, let's think about this more carefully. If we have elements a_1 < a_2 < ... < a_12,
    and all subset sums are distinct, then a_1 must be 1 (otherwise we can't form sum 1).
    Then a_2 must be at least 2 (otherwise {a_2} and {a_1, ...} might collide, but more precisely,
    the subset sums of {a_1} are {0, a_1}. Adding a_2, the new sums are {a_2, a_1+a_2}. 
    These must not collide with {0, a_1}, so a_2 != 0, a_2 != a_1, a_1+a_2 != 0, a_1+a_2 != a_1.
    Since a_1=1, a_2 != 1 and a_2+1 != 1 => a_2 != 0. So a_2 >= 2.
    
    In general, if we have k elements with distinct subset sums, the set of subset sums has size 2^k.
    When adding a_{k+1}, the new subset sums are the old ones plus {s + a_{k+1} for s in old_sums}.
    These two sets must be disjoint. So a_{k+1} must be larger than the maximum of the old subset sums,
    OR more precisely, the set {s + a_{k+1}} must not intersect {s}.
    
    The simplest sufficient condition is a_{k+1} > sum of all previous elements, which gives powers of 2.
    But this is not necessary. We just need a_{k+1} such that no two subset sums collide.
    
    A better approach: use a randomized search or a more clever greedy.
    
    Let me implement a greedy that at each step picks the smallest possible value that works.
    """
    start_time = time.time()
    deadline = 80.0  # 80 seconds, leaving 10s margin
    
    def is_valid_subset_sums(elements):
        """Check if all subset sums are distinct."""
        n = len(elements)
        sums = set()
        sums.add(0)
        for e in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + e)
            if new_sums & sums:
                return False
            sums |= new_sums
        return True
    
    # Greedy construction: start with [1], then add smallest possible next element
    def greedy_construct(n):
        elements = [1]
        for _ in range(n - 1):
            # Try the smallest possible next value
            # The current max subset sum is sum(elements)
            max_sum = sum(elements)
            # Try values from 2 upward
            candidate = 2
            while True:
                # Check if adding candidate maintains distinct subset sums
                test_elements = elements + [candidate]
                if is_valid_subset_sums(test_elements):
                    elements.append(candidate)
                    break
                candidate += 1
                # Safety: don't go too high
                if candidate > 1048576:
                    return None
            # Check deadline
            if time.time() > deadline:
                return None
        return elements
    
    # Try greedy first
    result = greedy_construct(12)
    if result is not None and max(result) <= 1048576:
        return result
    
    # If greedy fails, try a randomized local search
    # Start with a random valid set and try to minimize the max
    # For now, just return a known valid set if greedy fails
    
    # Fallback: powers of 2
    return [1 << i for i in range(12)]

# EVOLVE-BLOCK-END
