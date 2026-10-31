import hashlib
import time

def priority(vector, n):
    # Z-order (Morton) code priority
    # Each element is 0, 1, or 2.
    # We represent each element with 2 bits.
    # We interleave the bits of all elements to form a single integer.
    
    # For n elements, we have 2n bits.
    # We can construct the integer by iterating over bits.
    
    # Optimization: Precompute bit patterns for 0, 1, 2
    # 0 -> 00
    # 1 -> 01
    # 2 -> 10
    # Note: 3 is not used.
    
    # Let's just compute the interleaved integer.
    # bit_position starts at 0.
    # For each coordinate i:
    #   take the 2 bits of vector[i]
    #   place them at positions 2*i and 2*i+1?
    #   No, standard Morton code interleaves bits from different axes.
    #   Axis 0 bit 0 goes to pos 0.
    #   Axis 1 bit 0 goes to pos 1.
    #   Axis 0 bit 1 goes to pos 2.
    #   Axis 1 bit 1 goes to pos 3.
    
    # So for coordinate i, bit j (0 or 1) goes to position 2*i + j?
    # Let's check n=2.
    # Coord 0: bits b01, b00.
    # Coord 1: bits b11, b10.
    # Morton: b01 (pos 0), b11 (pos 1), b00 (pos 2), b10 (pos 3).
    # So pos = 2*i + j?
    # i=0, j=1 (MSB) -> pos 1? No.
    # Usually, we interleave starting from LSB.
    # Let's define the bits of v_i as b_{i,1} (MSB) and b_{i,0} (LSB).
    # v_i = 2*b_{i,1} + b_{i,0}.
    # Morton index bit at position 2*i + k?
    # Let's just use a simple hash if Z-order is too complex to get right quickly.
    
    # Actually, let's try a different structural idea:
    # Priority based on the sum of coordinates.
    # Vectors with sum close to n/2?
    
    # Let's stick to the random hash but use a faster, high-quality one.
    # The prompt says SHA256 gave 1.039.
    # Maybe BLAKE2b is faster and allows the evaluator to run more iterations?
    # No, the evaluator runs one greedy pass.
    # The time limit is 300s.
    # If n is small, it's fast.
    
    # Let's try a priority that is the sum of the coordinates.
    # This groups vectors by weight.
    # Greedy will pick all weight 0, then weight 1, etc.
    # This might be bad.
    
    # Let's try: priority = hash(vector)
    # But use a different hash.
    
    # What if we use the vector itself as a seed for a random number generator?
    # No, we need a scalar.
    
    # Let's try the "Ellenberg-Gijswijt" insight.
    # The largest cap sets are related to the size of the largest subset of $\{0,1,2\}^n$ with no 3-AP.
    # A simple lower bound is $c^n$ where $c \approx 2.217$.
    # The greedy algorithm with random order should find a set of size roughly $c^n$?
    # Actually, random greedy usually finds smaller sets than optimal.
    
    # Let's try to improve the random order by using a **seed** derived from the vector.
    # This is what SHA256 does.
    
    # Is there a better structural ordering?
    # How about **Gray Code** ordering?
    # Gray code changes one bit at a time.
    # This might help locality.
    
    # Let's implement a simple Gray Code based priority.
    # Convert vector to integer, then to Gray code.
    # But the mapping from $\{0,1,2\}^n$ to integer is not unique.
    # We can map $v$ to an integer in base 3.
    # Then convert that integer to Gray code.
    # Then use the Gray code integer as priority.
    
    def vec_to_int(v, n):
        num = 0
        for i in range(n):
            num = num * 3 + v[i]
        return num

    # Convert base-3 integer to Gray code?
    # Gray code is defined for binary integers.
    # We can treat the base-3 integer as a binary integer and apply Gray code.
    # This creates a permutation of the vectors.
    
    # Let's try this.
    num = vec_to_int(vector, n)
    # Apply Gray code transformation
    gray = num ^ (num >> 1)
    return float(gray)
