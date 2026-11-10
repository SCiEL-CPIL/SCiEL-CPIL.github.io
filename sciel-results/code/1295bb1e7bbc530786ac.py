"""
Rewritten program: maximize pairwise-noncoprime subset of [1,255] containing 255.

Strategy:
The problem asks for a subset S of [1, 255] containing 255 such that gcd(a, b) > 1 
for all distinct a, b in S, maximizing |S|.

255 = 3 * 5 * 17.
The set of all multiples of the smallest prime factor, 3, forms a valid clique.
Size = floor(255 / 3) = 85.

Any attempt to include numbers not divisible by 3 requires removing all multiples of 3 
that are coprime to the added number. 
- Adding a multiple of 5 (not 3) forces removal of 68 multiples of 3, allowing addition of 34 multiples of 5 (not 3). Net loss.
- Adding a multiple of 17 (not 3) forces removal of 80 multiples of 3, allowing addition of 10 multiples of 17 (not 3). Net loss.

Thus, the set of multiples of 3 is optimal.
"""

def construct():
    # The optimal set is all multiples of 3 in [1, 255].
    # This set has size 85.
    # It contains 255.
    # Any pair has gcd >= 3.
    return list(range(3, 256, 3))
