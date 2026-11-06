"""
Constructs n=12 distinct positive integers such that all subset sums are distinct.
The goal is to minimize the largest integer.

Mathematical Insight:
To ensure all 2^12 subset sums are distinct, the sequence must be superincreasing,
i.e., each element must be strictly greater than the sum of all preceding elements.
To minimize the largest element, we choose the smallest possible value at each step:
a_1 = 1
a_i = 1 + sum(a_0 ... a_{i-1})
This results in powers of 2: 1, 2, 4, ..., 2^(n-1).
For n=12, the maximum value is 2^11 = 2048, which is the theoretical minimum.
"""

# EVOLVE-BLOCK-START
def construct():
    n = 12
    max_val = 1048576
    
    result = []
    current_sum = 0
    
    for _ in range(n):
        # The next number must be at least current_sum + 1 to ensure 
        # superincreasing property and thus distinct subset sums.
        next_val = current_sum + 1
        
        # Ensure we stay within the bound (though for n=12 this is not an issue)
        if next_val > max_val:
            # Fallback or error handling, though theoretically unreachable for n=12
            # with this optimal construction
            raise ValueError("Exceeded maximum value bound")
            
        result.append(next_val)
        current_sum += next_val
        
    return result
# EVOLVE-BLOCK-END
