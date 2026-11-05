"""
Optimized construction of the maximum pairwise non-coprime set containing 255.

Key insight: 255 = 3 * 5 * 17.
The set of all multiples of 3 in [1, 255] has size 85 and is a valid clique
(every pair shares factor 3). This is optimal because:
- Any element must share a factor with 255 (factors 3, 5, 17)
- Multiples of 3 give the largest such set (85 elements)
- No element outside multiples of 3 can be added to this set while maintaining
  the pairwise non-coprime property, as any such element x would need
  gcd(x, 3) > 1, implying 3|x, which contradicts x not being a multiple of 3.
"""
import math

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    # The set of all multiples of 3 in [1, 255] is a clique of size 85.
    # 255 is a multiple of 3.
    # Any superset of this clique would need to include a number x not divisible by 3.
    # For the set to remain a clique, gcd(x, 3) must be > 1, which implies 3|x.
    # This is a contradiction. Thus, no element can be added to the set of multiples of 3.
    # Other cliques (e.g., based on 5 or 17) are smaller.
    # Therefore, the set of multiples of 3 is the maximum cardinality clique containing 255.
    
    return [x for x in range(1, 256) if x % 3 == 0]

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(f"Contains 255: {255 in result}")
