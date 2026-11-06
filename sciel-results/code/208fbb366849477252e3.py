"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _is_cap_set_incremental(existing_set, vec, n):
    """
    Check if adding vec to existing_set keeps it a cap set.
    A cap set has no 3-term arithmetic progressions.
    In F_3^n, a 3-AP is a, a+d, a+2d where d != 0.
    Equivalently, for any two distinct elements a, b in the set,
    the third element c = 2a - b (mod 3) must not be in the set,
    and c must be distinct from a and b (which is true if a != b).
    
    Actually, the condition for a set S to be a cap set is:
    For all distinct a, b in S, the element c = 2*a - b (mod 3) is not in S.
    Note that if a = b, then c = a, which is already in S, but we only
    consider distinct pairs. Also, the 3-AP can be written as a, b, c where
    a + c = 2b (mod 3). So for any distinct a, b, c in S, we must not have
    a + c = 2b.
    
    To check if adding vec keeps S a cap set:
    For every pair (a, b) in S (distinct), check that vec != 2a - b and vec != 2b - a.
    Also, for every a in S, check that vec != 2a - vec (which is always true since 2vec != 2a - vec => 3vec = 2a => 0 = 2a => a=0, but vec != a generally).
    Wait, let's rephrase.
    
    A set S is a cap set if it contains no 3 distinct elements a, b, c such that a + c = 2b.
    When adding vec, we need to ensure no new 3-AP is formed involving vec.
    A 3-AP involving vec can be:
    1. vec is the middle term: a + c = 2*vec, where a, c in S.
       This means a + c = 2*vec => c = 2*vec - a.
       So for each a in S, check if (2*vec - a) is in S. If so, and (2*vec - a) != a, then it's a 3-AP.
       Actually, if a = 2*vec - a, then 2a = 2*vec => a = vec, but vec is not in S yet.
       So we just need to check: for each a in S, is (2*vec - a) in S?
    2. vec is an end term:
       a + vec = 2b => vec = 2b - a, where a, b in S, a != b.
       So for each pair (a, b) in S with a != b, check if vec == 2b - a.
       Similarly, vec + a = 2b => a = 2b - vec. This is symmetric to the above.
    
    So the check is:
    - For each a in S: if (2*vec - a) in S and (2*vec - a) != a, then NOT a cap set.
      But since vec is not in S, 2*vec - a = a implies 2*vec = 2a => vec = a, which is false.
      So we just check: for each a in S, is (2*vec - a) in S?
    - For each pair (a, b) in S with a != b: if vec == 2b - a, then NOT a cap set.
    
    We can combine these. Let S be the existing set.
    For each a in S:
        c = (2*vec - a) % 3 (element-wise)
        If c in S: return False
    For each a in S:
        For each b in S where a != b:
            if vec == (2*b - a) % 3: return False
    
    Actually, the second condition is redundant with the first if we also check symmetrically?
    Let's see: if vec == 2b - a, then a + vec = 2b. This is a 3-AP with b as middle.
    In the first condition, we check for middle term vec.
    So we need to check for vec being middle AND vec being an end.
    
    Alternatively, a simpler way:
    For each a in S:
        For each b in S:
            if a == b: continue
            c = (2*b - a) % 3  # This is the third element if a, b, c form AP with b as middle? No.
            # 3-AP: x, y, z with x + z = 2y.
            # If we have a and b, the third element to form AP with b as middle is 2b - a.
            # The third element to form AP with a as middle is 2a - b.
            # So for each pair (a, b) with a != b, check if (2a - b) in S or (2b - a) in S.
            # But we are adding vec. So we need to check if vec is one of these third elements.
            # i.e., is vec == 2a - b or vec == 2b - a?
    
    Let's just do:
    for a in S:
        for b in S:
            if a == b: continue
            if vec == (2*a - b) % 3 or vec == (2*b - a) % 3:
                return False
    return True
    
    But this is O(|S|^2). We can optimize.
    
    Actually, the condition "vec == 2a - b" for some a, b in S, a != b, is equivalent to:
    For each a in S, is (2a - vec) in S? (since b = 2a - vec)
    And for each b in S, is (2b - vec) in S? (since a = 2b - vec)
    Wait, 2a - b = vec => b = 2a - vec. So for each a in S, check if (2a - vec) in S.
    Similarly, 2b - a = vec => a = 2b - vec. So for each b in S, check if (2b - vec) in S.
    These are the same check: for each x in S, check if (2x - vec) in S.
    But we must ensure that the pair is distinct. If 2x - vec = x, then x = vec, which is not in S.
    So we just need to check: for each x in S, is (2x - vec) in S?
    
    Wait, is that sufficient?
    If there exist a, b in S, a != b, such that vec = 2a - b, then b = 2a - vec.
    So if we check for each a in S whether (2a - vec) is in S, we will catch it.
    Similarly for vec = 2b - a, we get a = 2b - vec, so checking for each b in S whether (2b - vec) is in S catches it.
    But these are the same loop: for each x in S, check if (2x - vec) in S.
    
    However, we also need to check if vec is the middle term.
    If vec is the middle term, then a + c = 2vec for some a, c in S, a != c.
    This means c = 2vec - a. So for each a in S, check if (2vec - a) in S.
    
    So the full check is:
    1. For each a in S: if (2*vec - a) in S, return False.
    2. For each a in S: if (2*a - vec) in S, return False.
    
    Note that condition 2 is equivalent to: for each b in S, if (2*b - vec) in S, return False.
    Which is the same as condition 1 but with vec and a swapped? No.
    
    Let's test with a small example.
    S = {(0,0), (1,1)}. Is this a cap set?
    Check 3-AP: (0,0), (1,1), (2,2). (0,0)+(2,2) = (2,2) = 2*(1,1). So yes, it's a 3-AP.
    So S is NOT a cap set.
    
    Now, suppose we have S = {(0,0)} and we want to add vec = (1,1).
    Condition 1: For a=(0,0), 2*vec - a = (2,2) - (0,0) = (2,2). Is (2,2) in S? No.
    Condition 2: For a=(0,0), 2*a - vec = (0,0) - (1,1) = (2,2) mod 3 = (2,2). Is (2,2) in S? No.
    So we would say it's okay to add. But S U {(1,1)} = {(0,0), (1,1)} is not a cap set.
    
    What went wrong?
    The 3-AP is (0,0), (1,1), (2,2). None of these is vec=(1,1) as an end or middle in the sense I described?
    Wait, vec=(1,1) IS in the set after adding. The 3-AP involves vec.
    In the 3-AP (0,0), (1,1), (2,2), vec=(1,1) is the middle term.
    So condition 1 should have caught it: a=(0,0), 2*vec - a = (2,2). Is (2,2) in S? S only has (0,0). So no.
    But wait, the 3-AP is (0,0), (1,1), (2,2). The middle is (1,1)=vec.
    The ends are (0,0) and (2,2).
    For vec to be the middle, we need a and c in S such that a + c = 2vec.
    Here a=(0,0), c=(2,2). But c=(2,2) is NOT in S. S only has (0,0).
    So actually, {(0,0), (1,1)} IS a cap set?
    Let's check: A cap set has no 3 distinct elements a,b,c with a+c=2b.
    The set has only 2 elements. So no 3 distinct elements. So it IS a cap set.
    My earlier claim was wrong. {(0,0), (1,1)} is a cap set.
    The 3-AP (0,0), (1,1), (2,2) requires all three to be in the set.
    
    Okay, so the check is correct.
    
    Let's re-verify:
    To check if S U {vec} is a cap set, given S is a cap set:
    We need to ensure no 3-AP is formed that includes vec.
    Any such 3-AP has vec as one of the three elements.
    Case 1: vec is the middle. Then there exist a, c in S such that a + c = 2vec.
        => c = 2vec - a. So for some a in S, (2vec - a) is in S.
    Case 2: vec is an end. Then there exist a, b in S such that a + vec = 2b (vec is end, b is middle) or vec + a = 2b (same thing).
        => a = 2b - vec. So for some b in S, (2b - vec) is in S.
        Note: a must be distinct from b. If a = b, then 2b - vec = b => b = vec, which is not in S.
    
    So the check is:
    for a in S:
        if (2*vec - a) in S: return False
        if (2*a - vec) in S: return False
    return True
    
    This is O(|S|) per check, which is good.
    """
    # Use tuples for hashing
    vec_tuple = tuple(vec)
    for a in existing_set:
        # Check if vec is middle: a + c = 2vec => c = 2vec - a
        c = tuple((2*vec_tuple[i] - a[i]) % 3 for i in range(n))
        if c in existing_set:
            return False
        # Check if vec is end: a + vec = 2b => b = (a + vec) / 2? No.
        # a + vec = 2b => 2b = a + vec => b = (a + vec) * inv2 mod 3.
        # inv2 mod 3 is 2, since 2*2=4=1 mod 3.
        # So b = (a + vec) * 2 mod 3 = 2a + 2vec mod 3.
        # Wait, earlier I said a = 2b - vec => b = (a + vec)/2.
        # Let's stick to: for some b in S, a = 2b - vec.
        # So for each b in S, check if (2b - vec) is in S.
        # But we are looping over a. Let's just loop over b separately or combine.
        # Actually, the condition "for some b in S, (2b - vec) in S" is independent of a.
        # So we can check it once outside the loop, or inside.
        # Let's do it inside for simplicity, but it's redundant.
        # Better: check once outside.
    # Check if vec is an end: for some b in S, (2b - vec) in S
    for b in existing_set:
        a = tuple((2*b[i] - vec_tuple[i]) % 3 for i in range(n))
        if a in existing_set:
            return False
    return True

def priority(vector, n):
    # Uniform priority: greedy insertion in lexicographic order.
    return 0.0

# EVOLVE-BLOCK-END
