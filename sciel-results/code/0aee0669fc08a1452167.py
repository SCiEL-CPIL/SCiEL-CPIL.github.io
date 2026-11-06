"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _validate(color_map):
    """Check every K4 spans at least 5 distinct colours."""
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        cols = set()
        for x, y in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            if x > y:
                x, y = y, x
            cols.add(color_map[(x, y)])
        if len(cols) < 5:
            return False
    return True

def _validate_fast(colors_list):
    """Fast validation using the flat list of 120 colours in lexicographic order."""
    # Precompute index: for pair (i,j) with 0<=i<j<16, index = i*16 - i*(i+1)//2 + j - i - 1
    # Actually let's just use a dict for speed, or compute index directly
    # index for (i,j): sum_{k=0}^{i-1} (15-k) + (j-i-1) = i*16 - i*(i+1)//2 + j - i - 1
    # Let's verify: i=0, j=1: 0 + 0 = 0. Correct.
    # i=0, j=2: 0 + 1 = 1. Correct.
    # i=1, j=2: 15 + 0 = 15. Correct.
    
    def idx(i, j):
        if i > j:
            i, j = j, i
        return i * 16 - i * (i + 1) // 2 + j - i - 1

    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        # Get the 6 edges
        e1 = colors_list[idx(a, b)]
        e2 = colors_list[idx(a, c)]
        e3 = colors_list[idx(a, d)]
        e4 = colors_list[idx(b, c)]
        e5 = colors_list[idx(b, d)]
        e6 = colors_list[idx(c, d)]
        
        # Check if at least 5 distinct
        # We can do this without a set for speed, but set is fine for 6 elements
        cols = {e1, e2, e3, e4, e5, e6}
        if len(cols) < 5:
            return False
    return True

def construct():
    t0 = time.time()
    deadline = 110  # safety margin before 120s kill

    # Strategy: Try to find a 15-colouring using a 1-factorization of K16.
    # If it fails validation, try to repair or fallback.
    
    # 1. Standard 1-factorization of K16
    # Vertices 0..14 on a cycle, 15 is center.
    # Factor i (i=0..14):
    #   Edge (15, i)
    #   Edges (i+j mod 15, i-j mod 15) for j=1..7
    
    color_map = {}
    for i in range(15):
        color_map[(15, i)] = i
        color_map[(i, 15)] = i
        for j in range(1, 8):
            a = (i + j) % 15
            b = (i - j) % 15
            if a > b:
                a, b = b, a
            color_map[(a, b)] = i

    # Check if this specific 1-factorization is valid
    if _validate(color_map):
        result = []
        for i in range(16):
            for j in range(i + 1, 16):
                result.append(color_map[(i, j)])
        return result

    # The standard 1-factorization often fails the "5 colours in K4" condition.
    # Let's analyze why.
    # In a 1-factorization, any K4 has 6 edges.
    # If 3 edges form a matching, they have 3 different colours.
    # If 2 edges share a vertex, they have different colours (since it's a matching decomposition).
    # Wait, in a 1-factorization, edges sharing a vertex are in different factors.
    # So in a K4, the 6 edges are distributed among the 15 factors.
    # Can 3 edges in a K4 have the same colour? No, because a factor is a matching.
    # Can 2 edges in a K4 have the same colour? No, for the same reason.
    # Therefore, every K4 in a 1-factorization uses 6 distinct colours?
    # Let's re-verify.
    # A 1-factor is a perfect matching.
    # In a K4, the 6 edges can be partitioned into 3 perfect matchings.
    # If the global 1-factorization restricts to these 3 matchings on the K4,
    # then the 3 edges of each matching have distinct colours (globally distinct factors).
    # But do the edges within one matching on the K4 have the same global colour?
    # No, each edge in the global factorization has a unique factor ID.
    # So, if the 3 edges of a matching in the K4 all belong to the same global factor, they share a colour.
    # But a global factor is a perfect matching on 16 vertices.
    # It contains exactly one edge for each vertex.
    # So in a K4, a global factor can contain at most 2 edges (if they are disjoint) or 1 edge.
    # It cannot contain 3 edges because that would require 6 vertices or share a vertex.
    # So, in a K4, any two edges sharing a vertex have different colours.
    # Any two disjoint edges might have the same colour if they are in the same factor.
    # So a K4 can have at most 3 pairs of disjoint edges with the same colour.
    # The 6 edges of a K4 can be partitioned into 3 pairs of disjoint edges.
    # If all 3 pairs have the same colour within the pair, we have 3 colours total.
    # This is the worst case.
    # The condition requires at least 5 colours.
    # So we need to avoid having 3 pairs of disjoint edges sharing colours.
    
    # The standard 1-factorization might have this property.
    # Let's try to modify the factorization or use a different structure.
    
    # Alternative: Use (i+j) mod 17 which is known to be valid (17 colours).
    # Can we reduce 17 to 15 or 16?
    
    # Let's try a randomized search for a 15-colouring.
    # We can start with the 1-factorization and swap edges between factors to break monochromatic matchings in K4s.
    
    # Given the time limit and complexity, let's first check if the standard 1-factorization actually works.
    # My previous reasoning suggests it might use only 3 colours in some K4s.
    # Let's actually run the validation on the standard 1-factorization to be sure.
    
    # If it fails, we fall back to the 17-colouring which is guaranteed valid.
    # The task says "Minimize the number of colours used".
    # 17 is a valid upper bound.
    # Can we do better?
    # The bound is 13 <= f(16) <= 15.
    # So 15 is a theoretical possibility.
    
    # Let's try a simple local search to improve the 1-factorization.
    # But for now, let's stick to the known valid 17-colouring if the 15-colouring fails.
    
    # Wait, the inspiration code had a fallback to (i+j)%17.
    # The current program uses (i+j)%17.
    # The inspiration tries 1-factorization first.
    
    # Let's combine: Try 1-factorization. If valid, return it (15 colours).
    # If not, return (i+j)%17 (17 colours).
    
    # But wait, if the 1-factorization is invalid, we should try to fix it or use a different method.
    # However, finding a valid 15-colouring is hard.
    # Let's just return the 17-colouring if the 15-colouring fails.
    
    # Actually, let's check if the standard 1-factorization is valid.
    # I will not run it here, but I will assume it might fail.
    # So the fallback is crucial.
    
    # Let's add a check: if time is running out, return the 17-colouring.
    
    # For now, the safest bet is the 17-colouring.
    # But the task asks to minimize.
    # If I can find a 15-colouring, that's a huge improvement (score 17/15 = 1.13 vs 1.0).
    
    # Let's try to generate a 1-factorization and check it.
    # If it's invalid, we return 17.
    
    # There is a known result that some 1-factorizations of K16 satisfy the property.
    # Let's try a few different 1-factorizations.
    
    # One way to generate different 1-factorizations is to permute the vertices.
    # But that just relabels the colours, it doesn't change the structure.
    # We need a different structure.
    
    # Let's try the "cyclic" 1-factorization which is the one I described.
    # If it fails, maybe a random 1-factorization works?
    # Generating a random 1-factorization is non-trivial.
    
    # Given the constraints, I will stick with the current program's logic but add the 1-factorization attempt as a quick check.
    # If it passes, great. If not, fallback.
    
    # Let's refine the fallback to be the 17-colouring.
    
    result_17 = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Check if 1-factorization is valid
    if _validate(color_map):
        result = []
        for i in range(16):
            for j in range(i + 1, 16):
                result.append(color_map[(i, j)])
        return result
    
    # Fallback
    return result_17

# EVOLVE-BLOCK-END
