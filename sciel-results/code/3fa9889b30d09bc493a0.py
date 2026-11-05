"""
Constructs a set of 20 distinct positive integers <= 10000 
minimizing max(|A+A|, |A*A|).

Strategy:
Use an Arithmetic Progression (AP) {1, 2, ..., 20}.
- Sumset A+A for {1..n} is {2, ..., 2n}, size 2n-1.
  For n=20, |A+A| = 39. This is the theoretical minimum.
- Productset A*A for {1..n} has size approx n^2 / log(n).
  For n=20, |A*A| is roughly 130-150.
- Alternative: Geometric Progression (GP) minimizes A*A to 2n-1=39,
  but maximizes A+A to n(n+1)/2 = 210.
  Also, a GP with ratio 2 requires max element 2^19 > 10000,
  so it is infeasible for n=20 within the limit.
  
Thus, the AP is the optimal choice under the constraint, 
as it minimizes the bottleneck (Sumset) and keeps Productset reasonable.
"""

# EVOLVE-BLOCK-START
def construct():
    # Return the first 20 positive integers
    return list(range(1, 21))
# EVOLVE-BLOCK-END
