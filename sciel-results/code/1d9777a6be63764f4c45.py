"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

P = {'n': 256, 'maximum_size': 60}

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Repeated sums within the same
    cardinality are allowed.
    
    Strategy: Use a greedy approach with dynamic programming to maximize set size.
    We want to ensure that for any k < l, the minimum sum of a k-subset is greater
    than the maximum sum of an l-subset? No, that's too strong.
    
    Actually, the condition is: if |S1| != |S2|, then sum(S1) != sum(S2).
    This means the set of subset sums for cardinality k must be disjoint from the
    set of subset sums for cardinality l for all k != l.
    
    Let A_k be the set of all subset sums of size k. We need A_k ∩ A_l = ∅ for k != l.
    
    To maximize the size of the original set, we should pick numbers that spread out
    the subset sums as much as possible, ensuring separation between cardinalities.
    
    A simple sufficient condition: If we pick numbers x_1 < x_2 < ... < x_n,
    and if the maximum sum of any k-subset is less than the minimum sum of any (k+1)-subset,
    then all cardinality classes are separated.
    
    Max sum of k-subset: sum of the k largest elements.
    Min sum of (k+1)-subset: sum of the (k+1) smallest elements.
    
    We need: sum_{i=n-k+1}^{n} x_i < sum_{i=1}^{k+1} x_i for all k from 0 to n-1.
    
    For k=0: 0 < x_1 (always true for positive integers).
    For k=n-1: sum_{i=2}^{n} x_i < x_1 + x_2 + ... + x_n - x_n? No.
    Max sum of (n-1)-subset: total sum - x_1.
    Min sum of n-subset: total sum.
    So we need total sum - x_1 < total sum, which is always true.
    
    The critical case is usually around the middle or small k.
    Let's check k=1:
    Max sum of 1-subset: x_n
    Min sum of 2-subset: x_1 + x_2
    Condition: x_n < x_1 + x_2
    
    If we choose a geometric sequence, this holds easily.
    For example, powers of 2: 1, 2, 4, 8, ...
    x_n = 2^(n-1)
    x_1 + x_2 = 1 + 2 = 3
    For n=3, x_3=4, 4 < 3 is FALSE.
    So powers of 2 do NOT satisfy the "max k < min k+1" condition for k=1, n=3.
    Wait, subsets of size 1: {1}, {2}, {4}. Sums: 1, 2, 4.
    Subsets of size 2: {1,2}, {1,4}, {2,4}. Sums: 3, 5, 6.
    Subsets of size 3: {1,2,4}. Sum: 7.
    All sums are distinct! So the condition "max k < min k+1" is sufficient but not necessary.
    The actual requirement is just disjointness of the sum sets.
    
    However, finding a large set where the sum sets for different cardinalities are disjoint is hard.
    But note: if we use superincreasing sequences, all subset sums are distinct, which implies
    cardinality-separated sums (since distinct sums implies no overlap between any sets).
    A superincreasing sequence x_1 < x_2 < ... < x_n satisfies x_i > sum_{j<i} x_j.
    Then all 2^n subset sums are distinct.
    This is a very strong condition.
    
    Can we find a superincreasing sequence of length 60 with values in 1..256?
    x_1 = 1
    x_2 > 1 => 2
    x_3 > 1+2=3 => 4
    x_4 > 1+2+4=7 => 8
    ...
    x_k = 2^(k-1)
    x_60 = 2^59, which is way larger than 256.
    So we cannot have a superincreasing sequence of length 60 in 1..256.
    
    We need a weaker condition. We only need disjointness between different cardinalities.
    
    Let's try to construct a set greedily.
    Start with an empty set.
    Add numbers one by one.
    For each new number, check if adding it preserves the property.
    The property is: for all k, the set of sums of k-subsets is disjoint from the set of sums of l-subsets for l != k.
    
    This is computationally expensive to check directly for large sets.
    However, we can use a heuristic:
    If we ensure that the range of sums for cardinality k is disjoint from the range of sums for cardinality l, we are safe.
    Range of sums for cardinality k: [min_sum_k, max_sum_k]
    If we ensure max_sum_k < min_sum_{k+1} for all k, then all ranges are disjoint.
    This is the sufficient condition mentioned earlier.
    
    Let's see how large a set we can build with max_sum_k < min_sum_{k+1}.
    This implies:
    sum of k largest < sum of (k+1) smallest.
    
    Let the set be S = {s_1 < s_2 < ... < s_m}.
    Condition for k:
    s_m + s_{m-1} + ... + s_{m-k+1} < s_1 + s_2 + ... + s_{k+1}
    
    For k=1: s_m < s_1 + s_2
    For k=2: s_m + s_{m-1} < s_1 + s_2 + s_3
    ...
    
    Let's try to maximize m.
    We want to pick small numbers to allow more of them.
    Let s_1 = 1, s_2 = 2.
    Then s_m < 3 => s_m can be 2? No, distinct. So s_m must be < 3.
    If s_1=1, s_2=2, then s_m < 3 means s_m can only be 1 or 2, but they are already used.
    So with s_1=1, s_2=2, we can't add any third number?
    Wait, if m=2, k=1: s_2 < s_1 + s_2 => 2 < 1+2 => 2<3 True.
    So {1, 2} works.
    Can we add 3?
    S = {1, 2, 3}.
    k=1: max sum = 3. min sum of 2-subset = 1+2=3. 3 < 3 is False.
    So {1, 2, 3} fails the sufficient condition.
    Does it fail the actual condition?
    Sums of size 1: 1, 2, 3.
    Sums of size 2: 1+2=3, 1+3=4, 2+3=5.
    3 is in both! So it fails.
    
    Try {1, 2, 4}.
    k=1: max sum = 4. min sum of 2-subset = 1+2=3. 4 < 3 False.
    Actual check:
    Size 1: 1, 2, 4.
    Size 2: 3, 5, 6.
    Size 3: 7.
    All distinct. So {1, 2, 4} works.
    
    Try to add 5 to {1, 2, 4}. S = {1, 2, 4, 5}.
    Size 1: 1, 2, 4, 5.
    Size 2: 3, 5, 6, 6, 7, 9. (1+2=3, 1+4=5, 1+5=6, 2+4=6, 2+5=7, 4+5=9)
    5 is in Size 1 and Size 2. Fails.
    
    Try {1, 2, 4, 8}.
    Size 1: 1, 2, 4, 8.
    Size 2: 3, 5, 9, 6, 10, 12.
    Size 3: 7, 11, 13, 14.
    Size 4: 15.
    All distinct?
    Size 1: {1,2,4,8}
    Size 2: {3,5,9,6,10,12}
    Size 3: {7,11,13,14}
    Size 4: {15}
    No overlaps. Works.
    
    It seems powers of 2 work.
    How many powers of 2 fit in 256?
    1, 2, 4, ..., 128, 256.
    2^0 to 2^8.
    2^8 = 256.
    So we can have {1, 2, 4, 8, 16, 32, 64, 128, 256}.
    Size 9.
    
    Can we do better?
    The current program returns 8 numbers: 128..135.
    Score 1.0.
    
    We need to maximize the size.
    Let's try to find a larger set.
    The condition is that the set of subset sums for each cardinality are disjoint.
    
    Let's try a greedy algorithm that adds the smallest possible number that doesn't create a conflict.
    We maintain the set of sums for each cardinality.
    When adding a new number x, the new sums for cardinality k are:
    old_sums[k] ∪ {s + x for s in old_sums[k-1]}
    We need to ensure that for all k, new_sums[k] is disjoint from new_sums[l] for l != k.
    
    This is feasible for small sets.
    Let's implement this.
    
    Steps:
    1. Start with empty set. sums = [set()] for k=0..max_size. sums[0] = {0}.
    2. Iterate candidate numbers from 1 to 256.
    3. For each candidate, check if adding it preserves the disjointness property.
    4. If yes, add it.
    5. Stop when we have 60 numbers or no more candidates.
    
    We want to maximize the count.
    Using small numbers first might allow more numbers?
    Or large numbers?
    Powers of 2 used small numbers.
    
    Let's try adding numbers in increasing order.
    """
    max_val = 256
    max_size = 60
    
    # sums[k] is a set of sums for subsets of size k
    sums = [set() for _ in range(max_size + 1)]
    sums[0].add(0)
    
    selected = []
    
    for x in range(1, max_val + 1):
        # Check if we can add x
        # New sums[k] = sums[k] ∪ {s + x for s in sums[k-1]}
        new_sums = [set() for _ in range(max_size + 1)]
        valid = True
        
        # Calculate new sums
        for k in range(len(selected) + 1, 0, -1):
            # Subsets of size k in the new set are either old subsets of size k
            # or old subsets of size k-1 plus x
            new_sums[k] = sums[k].copy()
            for s in sums[k-1]:
                new_sums[k].add(s + x)
        
        # Check disjointness
        for k in range(len(selected) + 2):
            for l in range(k + 1, len(selected) + 2):
                if new_sums[k] & new_sums[l]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            selected.append(x)
            sums = new_sums
            if len(selected) >= max_size:
                break
    
    return selected

# EVOLVE-BLOCK-END
