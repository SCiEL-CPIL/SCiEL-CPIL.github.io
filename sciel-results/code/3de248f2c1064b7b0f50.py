"""
Rewritten program: maximize pairwise-noncoprime subset of [1,255] containing 255.

Strategy: 
255 = 3 * 5 * 17.
The set must contain 255, so every element $a$ in the set must satisfy $\gcd(a, 255) > 1$.
This implies $a$ must be divisible by 3, 5, or 17.
To maximize the size while maintaining the pairwise non-coprime property for all distinct pairs,
we select all multiples of the smallest prime factor of 255, which is 3.
Any two distinct multiples of 3 share the factor 3, so their gcd is at least 3.
The size of this set is $\lfloor 255/3 \rfloor = 85$.
Mixing in multiples of 5 or 17 that are not multiples of 3 would require removing many multiples of 3
to maintain the pairwise gcd > 1 property (since e.g., gcd(3, 5) = 1), resulting in a smaller set.
Thus, the set of all multiples of 3 is optimal.
"""
# EVOLVE-BLOCK-START
def construct():
    # Return all multiples of 3 in [1, 255]
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
