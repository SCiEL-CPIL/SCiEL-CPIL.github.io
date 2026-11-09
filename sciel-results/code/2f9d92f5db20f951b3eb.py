import time

N = 18

def _merge(a, b, offset_a=0, offset_b=0):
    """
    Generates comparators for merging two sorted sequences of sizes a and b.
    offset_a and offset_b are the starting wire indices for the two sequences.
    Returns a list of (i, j) comparators.
    """
    if a == 0 or b == 0:
        return []
    
    comparators = []
    
    # Step 1: Compare (offset_a + i, offset_b + i) for i in range(a)
    for i in range(a):
        comparators.append((offset_a + i, offset_b + i))
        
    # Step 2: Recursively merge the two halves
    # Top half: sizes floor(a/2) and floor(a/2) ?? 
    # Actually, the structure is:
    # The top half contains the smallest elements.
    # The sizes of the sub-merges depend on the parity of a and b.
    
    a1 = a // 2
    b1 = b // 2
    
    # The top merge handles the upper parts?
    # Standard Batcher merge:
    # 1. (i, a+i)
    # 2. Merge(a/2, a/2) on first a/2 items of A and first a/2 items of B?
    #    No, it merges the result of the first layer.
    #    The first layer outputs:
    #    For i < a: min(A_i, B_i) at A_i, max(A_i, B_i) at B_i.
    #    We need to merge the mins and the maxes.
    #    The mins are at A_0..A_{a-1}.
    #    The maxes are at B_0..B_{a-1}.
    #    Wait, the recursive call is on the "upper" and "lower" halves of the combined array?
    
    # Let's stick to the recursive structure on indices.
    # Top half: indices offset_a to offset_a + a1 - 1 and offset_b to offset_b + b1 - 1?
    # No, the standard recurrence is:
    # M(a,b) = a comps + M(floor(a/2), floor(a/2)) + M(ceil(a/2), ceil(b/2))
    # This assumes the inputs are contiguous and aligned.
    # But the offsets matter.
    
    # Let's define the indices explicitly.
    # Layer 1:
    # (offset_a + i, offset_b + i) for i in [0, a)
    
    # The elements at offset_a + i are the mins.
    # The elements at offset_b + i are the maxes.
    
    # We now have two sequences to merge:
    # 1. The mins: located at offset_a + 0 .. offset_a + a - 1.
    #    But they are not sorted relative to each other yet?
    #    Actually, A is sorted, B is sorted.
    #    A_i < A_{i+1}. B_i < B_{i+1}.
    #    min(A_i, B_i) < min(A_{i+1}, B_{i+1})?
    #    Yes.
    #    So the mins are sorted.
    #    And the maxes are sorted.
    
    # So we just need to merge the two sorted lists of mins and maxes?
    # No, that's not how Batcher works.
    # Batcher merge merges the "top" half and "bottom" half of the *entire* set.
    
    # Let's use the indices in the final array.
    # The final array has size a+b.
    # The first layer compares (i, a+i).
    # Then we recursively merge:
    # M(floor(a/2), floor(a/2)) on the first half of the output?
    # No.
    
    # Correct recursive structure for Batcher Merge M(a,b) on wires [0, a+b):
    # 1. (i, a+i) for i=0..a-1
    # 2. M(floor(a/2), floor(a/2)) on wires [0, a) ? No.
    #    On wires corresponding to the "upper" and "lower" parts of the first a items and first b items?
    
    # Let's look at the wire indices.
    # Input A: 0..a-1
    # Input B: a..a+b-1
    # 
    # Layer 1: (i, a+i)
    # 
    # Now we split the array into two halves:
    # Top half: 0..(a+b)//2 - 1
    # Bottom half: (a+b)//2 .. a+b-1
    #
    # But the recursive calls are on the *sources*?
    #
    # Actually, the simplest way to implement this correctly is to generate the comparators for M(a,b) assuming wires 0..a+b-1, and then shift them by offsets.
    
    def _gen_merge_local(a, b):
        """Generates merge comparators for sizes a,b on wires 0..a+b-1"""
        if a == 0 or b == 0:
            return []
        
        comps = []
        # Layer 1
        for i in range(a):
            comps.append((i, a + i))
            
        # Recursion
        a1 = a // 2
        b1 = b // 2
        
        # Top merge: M(a1, b1) on wires 0..a1+b1-1
        # But wait, the wires are not 0..a1+b1-1 for the top part.
        # The top part consists of the smaller values.
        # In Batcher's scheme, the recursive calls are:
        # M(floor(a/2), floor(a/2)) on the first half of A and first half of B?
        # No.
        
        # Let's use the property that the network is symmetric.
        # M(a,b) =
        #   Layer: (i, a+i)
        #   M(a/2, a/2) on indices [0, a/2) and [a, a+a/2)?
        #   M(a/2+1, b/2) on indices [a/2, a) and [a+a/2, a+b)?
        
        # This is tricky with odd sizes.
        
        # Let's use the standard definition:
        # M(a,b) merges A[0..a-1] and B[0..b-1].
        # Step 1: (i, a+i) for i=0..a-1.
        # Step 2: Merge(A[0..a/2-1], B[0..a/2-1]) ?? No.
        
        # Correct Recurrence:
        # M(a,b) = a + M(a1, b1) + M(a2, b2)
        # where a1 = floor(a/2), a2 = ceil(a/2)
        # b1 = floor(b/2), b2 = ceil(b/2)
        # The first recursive call M(a1, b1) operates on the "top" half of the wires?
        # No, it operates on the first a1 elements of A and first b1 elements of B?
        # And the second M(a2, b2) on the rest?
        
        # Let's assume the standard implementation where we track offsets.
        
        # Top half of the merge result comes from the smaller elements.
        # The comparators in the first layer connect A_i and B_i.
        # The recursive M(a1, b1) should merge A[0..a1-1] and B[0..b1-1]?
        # No, that would ignore the cross-comparisons.
        
        # Actually, for M(a,a), it is symmetric.
        # M(a,a) = a comps + M(a/2, a/2) + M(a/2, a/2).
        # The first M(a/2, a/2) is on wires A[0..a/2-1] and B[0..a/2-1].
        # The second M(a/2, a/2) is on wires A[a/2..a-1] and B[a/2..a-1].
        
        # For M(a,b) with a < b:
        # M(a,b) = a comps + M(a1, b1) + M(a2, b2).
        # M(a1, b1) on A[0..a1-1] and B[0..b1-1].
        # M(a2, b2) on A[a1..a-1] and B[b1..b-1].
        
        # This seems correct.
        
        comps_top = _gen_merge_local(a1, b1)
        # Shift by 0? No, A starts at 0, B starts at a.
        # So A[0..a1-1] are wires 0..a1-1.
        # B[0..b1-1] are wires a..a+b1-1.
        # So we need to shift the second input of the top merge by 'a'.
        
        comps_bot = _gen_merge_local(a2, b2)
        # A[a1..a-1] are wires a1..a-1.
        # B[b1..b-1] are wires a+b1..a+b-1.
        # So we need to shift the first input by 'a1' and the second input by 'a+b1'.
        
        # Let's re-verify the indices for the local generation.
        # _gen_merge_local(x, y) returns comps for wires 0..x+y-1, where 0..x-1 is input A and x..x+y-1 is input B.
        
        # For top merge M(a1, b1):
        # Input A is A[0..a1-1] -> wires 0..a1-1.
        # Input B is B[0..b1-1] -> wires a..a+b1-1.
        # The local generator assumes A is 0..a1-1 and B is a1..a1+b1-1.
        # So we need to map local B index j to global a+j.
        # Local A index i to global i.
        
        for i, j in comps_top:
            # i is in [0, a1), j is in [a1, a1+b1)
            comps.append((i, a + (j - a1)))
            
        for i, j in comps_bot:
            # i is in [0, a2), j is in [a2, a2+b2)
            # Global A index: a1 + i
            # Global B index: a + b1 + (j - a2)
            comps.append((a1 + i, a + b1 + (j - a2)))
            
        return comps

    # We can't easily recurse inside a function that returns a list without passing the generator.
    # Let's make _gen_merge_local a standalone or closure.
    
    def gen_merge(a, b):
        """Generates merge comparators for sizes a,b on wires 0..a+b-1"""
        if a == 0 or b == 0:
            return []
        
        comps = []
        for i in range(a):
            comps.append((i, a + i))
            
        a1 = a // 2
        a2 = a - a1
        b1 = b // 2
        b2 = b - b1
        
        top_comps = gen_merge(a1, b1)
        for i, j in top_comps:
            # Local A: 0..a1-1, Local B: a1..a1+b1-1
            # Global A: 0..a1-1, Global B: a..a+b1-1
            comps.append((i, a + (j - a1)))
            
        bot_comps = gen_merge(a2, b2)
        for i, j in bot_comps:
            # Local A: 0..a2-1, Local B: a2..a2+b2-1
            # Global A: a1..a-1, Global B: a+b1..a+b-1
            comps.append((a1 + i, a + b1 + (j - a2)))
            
        return comps

    def gen_sort(n):
        """Generates sort network for n wires on wires 0..n-1"""
        if n <= 1:
            return []
        
        n1 = n // 2
        n2 = n - n1
        
        left_comps = gen_sort(n1)
        for i, j in left_comps:
            # Local A: 0..n1-1. Global A: 0..n1-1.
            comps_left = (i, j)
            # We just add them, but they are on wires 0..n1-1.
            # The right sort is on wires n1..n-1.
            
        right_comps = gen_sort(n2)
        for i, j in right_comps:
            # Local A: 0..n2-1. Global A: n1..n-1.
            pass
            
        merge_comps = gen_merge(n1, n2)
        
        all_comps = []
        # Left sort
        all_comps.extend(left_comps)
        # Right sort
        all_comps.extend([(n1 + i, n1 + j) for i, j in right_comps])
        # Merge
        all_comps.extend(merge_comps)
        
        return all_comps

    return gen_sort(N)

def construct():
    return _merge(N, N) # Wait, _merge is for merging two sorted lists. We need to sort an unsorted list.
    # So we need the sort function.
    
# Let's rewrite cleanly.

def construct():
    """
    Returns a list of (i, j) comparator pairs that sort N=18 wires.
    Uses Batcher's Odd-Even Merge Sort.
    """
    N = 18
    
    def gen_merge(a, b):
        """
        Generates comparators for merging two sorted sequences of sizes a and b.
        The sequence of size 'a' is on wires [0, a).
        The sequence of size 'b' is on wires [a, a+b).
        """
        if a == 0 or b == 0:
            return []
        
        comps = []
        # Layer 1: Compare (i, a+i)
        for i in range(a):
            comps.append((i, a + i))
            
        a1 = a // 2
        a2 = a - a1
        b1 = b // 2
        b2 = b - b1
        
        # Top merge: M(a1, b1)
        # Inputs: A[0..a1-1] and B[0..b1-1]
        # Local wires for this merge: 0..a1+b1-1
        # Global wires: A[0..a1-1] is 0..a1-1. B[0..b1-1] is a..a+b1-1.
        top_comps = gen_merge(a1, b1)
        for i, j in top_comps:
            # i is in [0, a1), j is in [a1, a1+b1)
            # Map j to global: a + (j - a1)
            comps.append((i, a + (j - a1)))
            
        # Bottom merge: M(a2, b2)
        # Inputs: A[a1..a-1] and B[b1..b-1]
        # Local wires for this merge: 0..a2+b2-1
        # Global wires: A[a1..a-1] is a1..a-1. B[b1..b-1] is a+b1..a+b-1.
        bot_comps = gen_merge(a2, b2)
        for i, j in bot_comps:
            # i is in [0, a2), j is in [a2, a2+b2)
            # Map i to global: a1 + i
            # Map j to global: a + b1 + (j - a2)
            comps.append((a1 + i, a + b1 + (j - a2)))
            
        return comps

    def gen_sort(n):
        """
        Generates a sorting network for n wires on wires [0, n).
        """
        if n <= 1:
            return []
        
        n1 = n // 2
        n2 = n - n1
        
        # Sort left half
        left_comps = gen_sort(n1)
        
        # Sort right half (shifted by n1)
        right_comps_raw = gen_sort(n2)
        right_comps = [(n1 + i, n1 + j) for i, j in right_comps_raw]
        
        # Merge the two sorted halves
        # Left half is on wires [0, n1), Right half on [n1, n).
        # gen_merge(a, b) expects A on [0, a) and B on [a, a+b).
        # So we call gen_merge(n1, n2).
        merge_comps = gen_merge(n1, n2)
        
        return left_comps + right_comps + merge_comps

    return gen_sort(N)

# Sanity check count
# S(18) should be 94.
if __name__ == "__main__":
    net = construct()
    print(len(net))
