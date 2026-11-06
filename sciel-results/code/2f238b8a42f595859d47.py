import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    n = 256
    sys.setrecursionlimit(10000)
    
    # Build the 3-AP conflict graph
    # Two numbers conflict if they're part of a 3-term AP with some third number in [1, n]
    # Actually, we need: a set is 3-AP-free if NO three distinct elements form a 3AP
    # So vertices are 1..n, and edge (a,b) exists if there's a c in [1,n] such that
    # {a,b,c} forms a 3-term AP. But that's not quite right either.
    
    # Let me think more carefully. A 3-term AP is (a, a+d, a+2d) with d>0.
    # A set S is 3-AP-free if no three distinct elements of S form a 3AP.
    # So the "conflict" is between any two elements that could be part of a 3AP together with a third.
    # Actually, for maximum independent set formulation:
    # - Vertex i represents including i in our set
    # - We need to exclude any triple that forms a 3AP
    # - This is actually a 3-uniform hypergraph independent set, not a graph!
    
    # Hmm, this is trickier. Let me think of another approach.
    # Actually, we can model it as: for each potential 3AP (a, b, c), we cannot include all three.
    # This is a constraint satisfaction problem.
    
    # Alternative: use the fact that if we include a and c (with c-a even), we cannot include (a+c)/2.
    # So conflicts are pairwise: if a and c are both in S and c-a is even, then (a+c)/2 cannot be in S.
    # But this isn't a simple graph independent set either because the constraint involves triples.
    
    # Let me reconsider. The cleanest formulation:
    # Build a graph where vertices are 1..n
    # Edge between i and j if i and j can coexist in a 3AP with some k
    # Wait, that's the opposite of what we want.
    
    # Actually, for 3-AP-free sets, the constraint is:
    # For any a < b < c in S, we must NOT have b-a == c-b (i.e., a+b != 2c is not the right check)
    # We must NOT have a + c == 2b
    
    # So the forbidden patterns are triples {a, b, c} where a+c=2b.
    # This is a 3-uniform constraint, not pairwise.
    
    # However, we can reformulate: if we include both a and c (a<c, c-a even), then we CANNOT include (a+c)/2.
    # So the "conflict" between a and c (when c-a is even) is that they jointly forbid (a+c)/2.
    
    # For branch-and-bound, let's use a different approach:
    # Process elements in some order, and at each step decide include/exclude.
    # Use memoization or bounds based on which elements are still available.
    
    # Actually, let me try a smarter local search with better initialization and multi-move swaps.
    
    def is_3ap_free(S):
        """Check if S contains no 3-term arithmetic progression."""
        s = set(S)
        for a in s:
            for b in s:
                if a >= b:
                    continue
                c = 2*b - a
                if c in s and c != a and c != b:
                    return False
        return True
    
    def greedy_3free():
        """Greedy: add smallest available number that doesn't create a 3AP."""
        S = set()
        for x in range(1, n+1):
            # Check if adding x creates a 3AP
            valid = True
            for y in S:
                # x, y, z form 3AP? 
                # Case 1: x is middle: exists a,c in S with a+c=2x
                # Case 2: x is first: exists b,c in S with x+c=2b
                # Case 3: x is last: exists a,b in S with a+x=2b
                # Simpler: check all pairs in S
                pass
            # More efficient check
            for y in S:
                if y > x:
                    # x could be first term: need z where x+z=2y, z=2y-x
                    z = 2*y - x
                    if z in S and z != x and z != y:
                        valid = False
                        break
                    # x could be middle: need a where a+x=2y... no wait
                    # If x and y are in S, check if there's a z such that {x,y,z} is 3AP
                    # z = 2y-x (x first, y middle) or z = 2x-y (y first, x middle) or z = x+y (neither is middle... no)
                    # Actually for three numbers a<b<c to be 3AP: b=(a+c)/2, i.e., a+c=2b
                    # So for pair (x,y) with x<y:
                    #   If x is first, y is middle: third term = 2y-x, need 2y-x > y i.e. y>x (true)
                    #   If y is first, x is middle: third term = 2x-y, need 2x-y < x i.e. x<y (true)  
                    #   If x is middle, y is last: third term = 2x-y, need 2x-y < x i.e. x<y (true)
                    #   If y is middle, x is last: third term = 2y-x, need 2y-x > y i.e. y>x (true)
                    # So for pair (x,y), the possible third terms are 2y-x and 2x-y
                    z1 = 2*y - x
                    if z1 in S and z1 != x and z1 != y:
                        valid = False
                        break
                    z2 = 2*x - y
                    if z2 in S and z2 != x and z2 != y:
                        valid = False
                        break
                else:  # y < x
                    z1 = 2*x - y
                    if z1 in S and z1 != x and z1 != y:
                        valid = False
                        break
                    z2 = 2*y - x
                    if z2 in S and z2 != x and z2 != y:
                        valid = False
                        break
            if valid:
                S.add(x)
        return S
    
    # Better approach: use bitmasks for fast 3AP checking
    # Represent set as a 256-bit integer
    
    def can_add(mask, x):
        """Check if x can be added to the set represented by mask without creating a 3AP."""
        # For each y in the current set, check if adding x creates a 3AP
        # 3AP: a, b, c with a+c=2b, all distinct
        # If we add x, check:
        # 1. x as middle: for any a,c in set, a+c=2x
        # 2. x as first: for any b,c in set with b>x, x+c=2b => c=2b-x
        # 3. x as last: for any a,b in set with b<x, a+x=2b => a=2b-x
        
        # Efficient: iterate over bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            y = lb.bit_length() - 1  # 0-indexed, so actual value is y+1
            y_val = y + 1
            m ^= lb
            
            if y_val == x:
                continue
                
            # Check if x, y_val, and some z form a 3AP
            # Case A: x is middle, y_val is one end: z = 2*x - y_val
            z = 2*x - y_val
            if z != x and z != y_val and 1 <= z <= n:
                if (mask >> (z-1)) & 1:
                    return False
            
            # Case B: y_val is middle, x is one end: z = 2*y_val - x
            z = 2*y_val - x
            if z != x and z != y_val and 1 <= z <= n:
                if (mask >> (z-1)) & 1:
                    return False
            
            # Case C: both x and y_val are ends: middle = (x+y_val)//2
            if (x + y_val) % 2 == 0:
                mid = (x + y_val) // 2
                if mid != x and mid != y_val and 1 <= mid <= n:
                    if (mask >> (mid-1)) & 1:
                        return False
        
        return True
    
    def mask_sum_recip(mask):
        """Compute sum of 1/i for i in mask."""
        total = Fraction(0)
        m = mask
        while m:
            lb = m & (-m)
            i = lb.bit_length()  # 1-indexed
            total += Fraction(1, i)
            m ^= lb
        return total
    
    # Start with greedy
    mask = 0
    for x in range(1, n+1):
        if can_add(mask, x):
            mask |= (1 << (x-1))
    
    best_mask = mask
    best_score = mask_sum_recip(mask)
    
    # Local search with bitmask operations
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements
        for x in range(1, n+1):
            if (mask >> (x-1)) & 1:
                continue
            if can_add(mask, x):
                new_mask = mask | (1 << (x-1))
                new_score = best_score + Fraction(1, x)
                if new_score > best_score:
                    mask = new_mask
                    best_score = new_score
                    best_mask = mask
                    improved = True
        
        # Try swaps: remove x, add y where y < x (prefer smaller y for better reciprocal)
        if time.time() > deadline - 5:
            break
        
        # Get list of elements in mask
        elements_in = [i for i in range(1, n+1) if (mask >> (i-1)) & 1]
        elements_out = [i for i in range(1, n+1) if not (mask >> (i-1)) & 1]
        
        # Sort by reciprocal value (smaller number = higher reciprocal)
        for y in elements_out:
            if time.time() > deadline - 3:
                break
            for x in elements_in:
                if x <= y:
                    continue
                # Try swapping x out, y in
                new_mask = (mask & ~(1 << (x-1))) | (1 << (y-1))
                # Check if valid
                if can_add(new_mask & ~(1 << (y-1)), y):  # Check y can be added to mask without x
                    new_score = best_score - Fraction(1, x) + Fraction(1, y)
                    if new_score > best_score:
                        mask = new_mask
                        best_score = new_score
                        best_mask = mask
                        improved = True
                        elements_in = [i for i in range(1, n+1) if (mask >> (i-1)) & 1]
                        elements_out = [i for i in range(1, n+1) if not (mask >> (i-1)) & 1]
                        break
        
        # Try double swaps: remove x1, x2, add y1, y2
        if time.time() > deadline - 2:
            break
        
        elements_in = [i for i in range(1, n+1) if (mask >> (i-1)) & 1]
        elements_out = [i for i in range(1, n+1) if not (mask >> (i-1)) & 1]
        
        # Try removing two large elements and adding two small ones
        for i in range(len(elements_in)):
            if time.time() > deadline - 1:
                break
            x1 = elements_in[i]
            for j in range(i+1, len(elements_in)):
                x2 = elements_in[j]
                # Try adding y1, y2 where y1 < x1 and y2 < x2
                for y1 in elements_out:
                    if time.time() > deadline - 0.5:
                        break
                    if y1 >= x1:
                        continue
                    for y2 in elements_out:
                        if y2 >= x2 or y2 == y1:
                            continue
                        # Check if we can add both y1 and y2 to mask without x1, x2
                        base_mask = mask & ~(1 << (x1-1)) & ~(1 << (x2-1))
                        if can_add(base_mask, y1) and can_add(base_mask | (1 << (y1-1)), y2):
                            new_score = best_score - Fraction(1, x1) - Fraction(1, x2) + Fraction(1, y1) + Fraction(1, y2)
                            if new_score > best_score:
                                mask = base_mask | (1 << (y1-1)) | (1 << (y2-1))
                                best_score = new_score
                                best_mask = mask
                                improved = True
                                elements_in = [i for i in range(1, n+1) if (mask >> (i-1)) & 1]
                                elements_out = [i for i in range(1, n+1) if not (mask >> (i-1)) & 1]
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Convert mask to sorted list
    result = [i for i in range(1, n+1) if (mask >> (i-1)) & 1]
    return sorted(result)
