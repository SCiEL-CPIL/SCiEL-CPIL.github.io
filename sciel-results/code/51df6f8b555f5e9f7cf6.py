"""Improved program: returns all multiples of 3 in [1,255].

255 = 3 * 5 * 17. All multiples of 3 share the prime factor 3, so they are
pairwise non-coprime. Since 3 is the smallest prime factor of 255, the set of
multiples of 3 is the largest such clique: any element in a valid set must be
divisible by 3, 5, or 17 to share a factor with 255, and if the set contains
any number not divisible by 3 (e.g., a pure multiple of 5), it cannot coexist
with numbers like 3 or 7*3=21 that are only divisible by 3 among {3,5,17}.
The maximum is floor(255/3) = 85.
"""
# EVOLVE-BLOCK-START
def construct():
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
