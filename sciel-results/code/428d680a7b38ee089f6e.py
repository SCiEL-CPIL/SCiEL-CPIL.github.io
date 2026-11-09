"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def priority(vector, n):
    """
    Assign priority to vectors in {0,1,2}^n to maximize cap set size via greedy insertion.
    
    Strategy: 
    - A cap set is a subset of {0,1,2}^n with no three distinct elements forming an arithmetic progression (a, b, c) where a+c=2b.
    - In F_3^n, this means no three distinct points lie on a line (x, x+v, x+2v for v != 0).
    - Greedy approach: add vectors in descending priority order, keeping the set a cap set.
    
    Key insight for priority:
    - Vectors with more 'balanced' coordinates (mix of 0,1,2) tend to be more "central" and conflict with more potential lines.
    - Vectors at the "corners" (all 0s, all 1s, all 2s, or with extreme coordinate distributions) are safer.
    - However, for maximizing size, we want to add vectors that have minimal conflict with previously added vectors.
    
    Since this is a static priority (computed without knowing the current set), we need a heuristic that generally leads to larger cap sets.
    
    Known result: The maximum cap set size in {0,1,2}^n grows as ~2.756^n.
    
    Heuristic: Prioritize vectors that have a "simple" structure. Specifically, vectors with fewer distinct coordinate values or extreme values might be safer.
    
    Let's try: priority = - (number of 1s) * weight1 - (number of 2s) * weight2 + (number of 0s) * weight0
    
    Actually, let's think about it differently. In the greedy algorithm, the order matters. We want to add "safe" vectors first.
    
    A vector with all coordinates equal (00...0, 11...1, 22...2) is always safe to add (it only forms lines with itself if we consider degenerate cases, but cap sets require distinct points).
    
    Let's prioritize vectors based on their "distance" from the center. Vectors that are more "extreme" (closer to corners of the hypercube) might be safer.
    
    Alternative: Use a priority based on the sum of coordinates. Vectors with extreme sums (all 0s or all 2s) are safer.
    
    Let's try: priority = -abs(sum(vector) - n) 
    This would prioritize vectors with sum close to 0 or 2n (extremes) over those with sum close to n (center).
    
    Wait, actually, vectors with all 0s have sum 0, all 1s have sum n, all 2s have sum 2n.
    The center is sum = n.
    
    Let's try prioritizing vectors with sum far from n.
    
    priority = -abs(sum(vector) - n)
    
    But this might not be optimal. Let's think about another approach.
    
    Another idea: prioritize vectors that have many 0s or many 2s (extreme values) over those with many 1s.
    
    Let's use: priority = count_0s - count_1s + count_2s * 0.5
    Or simply: priority = count_0s + count_2s - count_1s
    
    This would prioritize vectors with more 0s and 2s over 1s.
    
    Let's go with: priority = (count_0s + count_2s) - count_1s
    
    This gives higher priority to vectors with more extreme coordinates (0 or 2) and lower priority to vectors with more 1s.
    """
    count_0 = vector.count(0)
    count_1 = vector.count(1)
    count_2 = vector.count(2)
    return (count_0 + count_2) - count_1
# EVOLVE-BLOCK-END
