def construct():
    # Select integers in [2, 40] with distinct subset products.
    # Strategy: Use prime powers where exponents form a superincreasing sequence
    # for each prime, ensuring distinct subset sums of exponents.
    
    # Prime 2: powers 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32
    # Exponents {1,2,4} are superincreasing: 1>0, 2>1, 4>1+2=3
    # Subset sums: 0,1,2,3,4,5,6,7 (all distinct) -> 3 numbers
    # Using {1,2,4} gives numbers: 2, 4, 16
    
    # Prime 3: powers 3^1=3, 3^2=9, 3^3=27
    # Exponents {1,2}: 1>0, 2>1. Subset sums: 0,1,2,3 -> 2 numbers: 3, 9
    
    # Prime 5: powers 5^1=5, 5^2=25
    # Exponents {1,2}: subset sums 0,1,2,3 -> 2 numbers: 5, 25
    
    # Prime 7: 7^1=7 -> 1 number: 7
    # Primes 11,13,17,19,23,29,31,37: only first power -> 8 numbers
    
    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify: all subset products must be distinct
    from itertools import combinations
    
    def all_subset_products(nums):
        prods = set()
        for r in range(len(nums) + 1):
            for combo in combinations(nums, r):
                prod = 1
                for x in combo:
                    prod *= x
                if prod in prods:
                    return False
                prods.add(prod)
        return True
    
    # Safety check (should always pass by construction)
    assert all_subset_products(result), "Subset products not distinct!"
    
    # Ensure all elements are Python int in range [2, 40]
    result = [int(x) for x in result]
    assert all(isinstance(x, int) and 2 <= x <= 40 for x in result)
    assert len(set(result)) == len(result), "Elements must be distinct"
    
    return result
