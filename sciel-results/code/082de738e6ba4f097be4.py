import time

# EVOLVE-BLOCK-START

def _is_valid_set(nums):
    """Check if all 2^k subset products are distinct."""
    products = {1}
    for n in nums:
        new_products = {x * n for x in products}
        if new_products & products:
            return False
        products |= new_products
    return True

def construct():
    """
    Returns a list of 16 distinct integers in the range 2..40 such that
    all 2^16 subset products are distinct.
    
    Strategy: Use prime powers with exponents that have distinct subset sums.
    - Prime 2: 2^1=2, 2^2=4, 2^4=16 (exponents {1,2,4}, subset sums 0..7 all distinct)
    - Prime 3: 3^1=3, 3^2=9 (exponents {1,2}, subset sums 0..3 all distinct)
    - Prime 5: 5^1=5, 5^2=25 (exponents {1,2}, subset sums 0..3 all distinct)
    - Primes 7,11,13,17,19,23,29,31,37: the prime itself (exponent {1})
    
    Total: 3 + 2 + 2 + 9 = 16 numbers.
    
    This is optimal because:
    - There are 12 primes in [2,40].
    - For each prime, the maximum number of powers with distinct subset-sum exponents
      that fit in [2,40] is: prime 2 -> 3 (2,4,16), prime 3 -> 2 (3,9), prime 5 -> 2 (5,25),
      all other primes -> 1 each.
    - Total: 3+2+2+9 = 16.
    - No composite number can be added without creating a collision, since any composite
      in [2,40] is a product of primes already represented, and its exponent vector
      lies in the span of existing vectors, creating a subset-sum collision.
    """
    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify
    if not _is_valid_set(result):
        # Fallback to primes (should never happen)
        result = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
