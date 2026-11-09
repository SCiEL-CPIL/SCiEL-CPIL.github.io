"""Direct construction for 'Large sets with distinct subset products'."""

import time

def construct():
    # The key insight: if all numbers are prime powers, then subset products
    # are distinct iff for each prime p, the set of exponents assigned to items
    # with that prime factor is "subset-sum-free" (no element equals the sum of
    # a subset of the other elements).
    
    # For each prime in [2, 40], we choose a maximal subset-sum-free set of exponents.
    # 
    # p=2: max exponent 5 (32<=40). Exponents {1,2,4} -> sums: 1,2,3,4,5,6,7. All distinct. Size 3.
    # p=3: max exponent 3 (27<=40). {1,2} -> sums: 1,2,3. Size 2. ({1,2,3} fails since 1+2=3)
    # p=5: max exponent 2 (25<=40). {1,2} -> sums: 1,2,3. Size 2.
    # p=7: max exponent 1. {1}. Size 1.
    # p=11: max exponent 1. {1}. Size 1.
    # p=13: max exponent 1. {1}. Size 1.
    # p=17: max exponent 1. {1}. Size 1.
    # p=19: max exponent 1. {1}. Size 1.
    # p=23: max exponent 1. {1}. Size 1.
    # p=29: max exponent 1. {1}. Size 1.
    # p=31: max exponent 1. {1}. Size 1.
    # p=37: max exponent 1. {1}. Size 1.
    
    # Total: 3 + 2 + 2 + 9*1 = 16 items.
    
    # The specific numbers:
    # p=2: 2^1=2, 2^2=4, 2^4=16
    # p=3: 3^1=3, 3^2=9
    # p=5: 5^1=5, 5^2=25
    # p=7: 7
    # p=11: 11
    # p=13: 13
    # p=17: 17
    # p=19: 19
    # p=23: 23
    # p=29: 29
    # p=31: 31
    # p=37: 37
    
    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify: all distinct, all in [2, 40], all Python ints
    assert len(result) == len(set(result)), "duplicates found"
    assert all(isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 for x in result)
    
    # Verify subset products are distinct (optional but safe for small n=16)
    # For n=16, we have 2^16 = 65536 subset products to check.
    prods = set()
    for mask in range(1 << 16):
        prod = 1
        for i in range(16):
            if mask & (1 << i):
                prod *= result[i]
        if prod in prods:
            raise ValueError("Collision detected in subset products")
        prods.add(prod)
    
    return list(result)
