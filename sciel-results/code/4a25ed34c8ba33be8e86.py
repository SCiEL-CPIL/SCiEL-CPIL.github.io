"""Optimized program using bitwise integer packing for constraint checking."""

def construct():
    # We need to assign colors to positions 1..128 (index 0..127 in our list)
    # Constraint: Every nonconstant four-term progression (a, a+d, a+2d, a+3d) 
    # with d>0 and a+3d <= 128 must use at least 3 distinct colors.
    # This is equivalent to: it is NOT the case that all 4 are the same color,
    # AND it is NOT the case that exactly 2 colors are used (i.e., the set of colors has size <= 2).
    # Wait, "at least 3 distinct colors" means the set of colors in the 4 positions has size >= 3.
    # So we must avoid:
    # 1. All 4 same (size 1)
    # 2. Exactly 2 distinct colors (size 2)
    
    # The seed program uses a greedy approach. Let's analyze the seed's constraints:
    # The seed checks:
    # 1. all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
    #    This checks that for any d such that x-1-3d >= 0 (i.e., the full progression fits),
    #    the set {color, c1, c2, c3} has size >= 3.
    # 2. all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
    #    This checks that for any d such that x-1-2d >= 0, it is not the case that both
    #    colors[x-1-d] == color AND colors[x-1-2d] == color.
    #    Why? If colors[x-1-d] == color and colors[x-1-2d] == color, and we pick color for x-1,
    #    then we have a 3-term progression with all same color. But the constraint is on 4-term progressions.
    #    Actually, let's re-read the task: "Every nonconstant four-term progression of positions must use AT LEAST three distinct colors."
    #    A four-term progression is (a, a+d, a+2d, a+3d).
    #    If we are at position x (1-indexed), we are choosing color[x].
    #    The progressions ending at x are (x-3d, x-2d, x-d, x) for d=1,2,...
    #    The progressions starting at x are (x, x+d, x+2d, x+3d) but these are not fully determined yet.
    #    The progressions where x is the 2nd or 3rd term are also not fully determined.
    #    So the only fully determined progressions involving x that we can check immediately are those ending at x.
    #    Wait, the seed program seems to check something else.
    #    Let's look at the seed's second condition:
    #    any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1)
    #    This means: NOT (colors[x-1-d]==color AND colors[x-1-2d]==color).
    #    If this holds, then we don't have three consecutive terms in an arithmetic progression with the same color.
    #    Why is this necessary?
    #    If we have c[x-2d] == c[x-d] == c[x], then consider the 4-term progression (x-3d, x-2d, x-d, x).
    #    If c[x-3d] is either c[x] or some other color, we might violate the constraint.
    #    Actually, if c[x-2d] == c[x-d] == c[x] = C, and c[x-3d] = D.
    #    If D == C, then all 4 are C (size 1) -> Violation.
    #    If D != C, then colors are {C, D} (size 2) -> Violation.
    #    So, having three consecutive terms in a 4-AP with the same color is ALWAYS a violation, regardless of the 4th term.
    #    Therefore, the condition "no three consecutive terms in any 4-AP can have the same color" is a necessary condition.
    #    The seed's first condition ensures that when we place the 4th term, the set size is >= 3.
    #    But the seed's second condition prevents the formation of a 3-term monochromatic subsequence that would inevitably lead to a violation when the 4th term is placed (or already exists).
    #    Actually, the 4-term progression (a, a+d, a+2d, a+3d) involves 4 positions.
    #    If we are placing position x, the progressions that are "completed" are those where x is the last term: (x-3d, x-2d, x-d, x).
    #    For these, we need |{c[x-3d], c[x-2d], c[x-d], c[x]}| >= 3.
    #    This is equivalent to:
    #    - Not all 4 same.
    #    - Not exactly 2 distinct.
    #    The "not exactly 2 distinct" condition is complex.
    #    However, if we ensure that no 3 of the 4 are the same, does that guarantee size >= 3?
    #    If 3 are the same, say C, C, C, D. If D==C, size 1. If D!=C, size 2. Both are violations.
    #    So, "no 3 of the 4 are the same" is a necessary condition.
    #    Is it sufficient?
    #    If no 3 are the same, the possible distributions of 4 items into colors are:
    #    - 4 distinct: size 4 (OK)
    #    - 2, 2: size 2 (VIOLATION)
    #    - 2, 1, 1: size 3 (OK)
    #    - 1, 1, 1, 1: size 4 (OK)
    #    So, "no 3 same" is NOT sufficient. We also need to avoid the 2,2 split.
    #    The 2,2 split means the colors are {A, A, B, B}.
    #    This happens if c[x-3d]==c[x-d] and c[x-2d]==c[x] (and A!=B), or c[x-3d]==c[x-2d] and c[x-d]==c[x] (and A!=B), etc.
    
    # Let's stick to the seed's logic which seems to work (score 1.0).
    # The directive asks to use bitwise integer packing.
    # Instead of storing a list of colors, we can store for each position a bitmask of which colors are "forbidden" or "used"?
    # No, the colors are chosen from 0..127.
    # The idea is to evaluate all 128 candidate colors in parallel.
    
    # Let's implement the greedy algorithm with bitmasks.
    # For each position i (0-indexed), we want to find a color c in 0..127 such that:
    # 1. For all d such that i-3d >= 0:
    #    The set {c, colors[i-d], colors[i-2d], colors[i-3d]} has size >= 3.
    # 2. For all d such that i-2d >= 0:
    #    It is not the case that colors[i-d] == c AND colors[i-2d] == c.
    
    # We can precompute for each position i and each d, the constraints on c.
    # Condition 2: c != colors[i-d] OR c != colors[i-2d].
    # This forbids c if c == colors[i-d] == colors[i-2d].
    # So if colors[i-d] == colors[i-2d] == K, then c != K.
    
    # Condition 1: |{c, A, B, D}| >= 3 where A=colors[i-d], B=colors[i-2d], D=colors[i-3d].
    # This is violated if the set size is 1 or 2.
    # Size 1: c=A=B=D. So c cannot be A if A=B=D.
    # Size 2: The set {c, A, B, D} has exactly 2 elements.
    # This happens if:
    # - c is one of {A,B,D} and {A,B,D} has size 2, and c is one of the two values.
    #   Specifically, if {A,B,D} = {X,Y} with X!=Y.
    #   If c=X, set is {X,Y} (size 2) -> Violation.
    #   If c=Y, set is {X,Y} (size 2) -> Violation.
    #   If c=Z (Z!=X,Y), set is {X,Y,Z} (size 3) -> OK.
    #   So if {A,B,D} has size 2, c cannot be in {A,B,D}.
    # - c is not in {A,B,D} but {A,B,D} has size 1? No, if {A,B,D}={X}, then c!=X gives size 2 {X,c}. Violation. c==X gives size 1. Violation.
    #   So if {A,B,D} has size 1, NO color c is allowed?
    #   Wait, if A=B=D=X.
    #   If c=X, set is {X} (size 1) -> Violation.
    #   If c!=X, set is {X,c} (size 2) -> Violation.
    #   So if the previous 3 terms are all the same color, it is IMPOSSIBLE to place the 4th term without violation.
    #   This confirms that Condition 2 (no 3 same) is essential to avoid unsolvable states.
    
    # So the constraints on c for a given i and d are:
    # Let A=colors[i-d], B=colors[i-2d], D=colors[i-3d].
    # Forbidden colors for this d:
    # - If A==B==D: All colors are forbidden (but this state should be prevented by Condition 2 from previous steps).
    # - If |{A,B,D}| == 2: c cannot be in {A,B,D}.
    # - If |{A,B,D}| == 3: c can be anything?
    #   If c is in {A,B,D}, say c=A. Set is {A,B,D} (size 3) -> OK.
    #   If c not in {A,B,D}, set is {A,B,D,c} (size 4) -> OK.
    #   So no restriction if |{A,B,D}| == 3.
    # - If |{A,B,D}| == 4: Impossible since only 3 elements.
    
    # So, for each d, we compute a bitmask of forbidden colors.
    # Then we OR all these bitmasks for all valid d.
    # Then we pick the smallest color not in the forbidden mask.
    
    # We also need to enforce Condition 2:
    # For d such that i-2d >= 0:
    # If colors[i-d] == colors[i-2d] == K, then c != K.
    # This adds K to the forbidden mask.
    
    # Algorithm:
    # colors = [0] * 128
    # for i in range(128):
    #   if i == 0:
    #     colors[0] = 0
    #     continue
    #   forbidden = 0
    #   # Check 4-term progressions ending at i
    #   for d in range(1, i//3 + 1):
    #     A = colors[i-d]
    #     B = colors[i-2*d]
    #     D = colors[i-3*d]
    #     if A == B == D:
    #       # This should not happen if we maintain the invariant.
    #       # But if it does, no color is valid. We might need to backtrack or choose differently earlier.
    #       # For now, assume it doesn't happen.
    #       pass
    #     else:
    #       # Compute set of {A,B,D}
    #       # If size 2, forbid all in {A,B,D}
    #       # If size 1, forbid all (impossible state)
    #       # If size 3, forbid none
    #       vals = {A, B, D}
    #       if len(vals) == 2:
    #         for v in vals:
    #           forbidden |= (1 << v)
    #       elif len(vals) == 1:
    #         # Forbid all colors?
    #         forbidden = (1 << 128) - 1
    #   # Check 3-term conditions (to prevent future 4-term violations)
    #   for d in range(1, i//2 + 1):
    #     A = colors[i-d]
    #     B = colors[i-2*d]
    #     if A == B:
    #       forbidden |= (1 << A)
    #   # Pick smallest color not in forbidden
    #   for c in range(128):
    #     if not (forbidden & (1 << c)):
    #       colors[i] = c
    #       break
    #   else:
    #     # No valid color found. This indicates a failure.
    #     # We might need a more sophisticated search.
    #     pass
    # return colors

    # Let's optimize the inner loop using bitmasks for the sets.
    # We can precompute masks for each color.
    
    colors = [0] * 128
    colors[0] = 0
    
    for i in range(1, 128):
        forbidden = 0
        
        # 4-term progressions ending at i: (i-3d, i-2d, i-d, i)
        # d can be 1 to i//3
        for d in range(1, i // 3 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            D = colors[i - 3 * d]
            
            # Check if A, B, D are all same
            if A == B and B == D:
                # This is a critical failure state.
                # We should have prevented this.
                # For now, mark all as forbidden.
                forbidden = (1 << 128) - 1
                break
            
            # Check if exactly 2 distinct
            # We can check this by seeing if (A^B) == 0 or (A^D) == 0 or (B^D) == 0
            # If any pair is equal, the set size is <= 2.
            # If set size is 2, we forbid the two values.
            # If set size is 1, we forbid all (handled above).
            
            # Let's determine the distinct values.
            # We can use a small local variable to track them.
            # Since we want speed, let's use bitwise tricks or simple checks.
            
            if A == B or A == D or B == D:
                # At least two are equal. Since not all three, exactly two are equal.
                # Find the two distinct values.
                if A == B:
                    v1 = A
                    v2 = D
                elif A == D:
                    v1 = A
                    v2 = B
                else: # B == D
                    v1 = B
                    v2 = A
                
                forbidden |= (1 << v1)
                forbidden |= (1 << v2)
            # If all distinct, no restriction from this d.
        
        # 3-term conditions: (i-2d, i-d, i)
        # If colors[i-2d] == colors[i-d], then c cannot be that color.
        for d in range(1, i // 2 + 1):
            A = colors[i - d]
            B = colors[i - 2 * d]
            if A == B:
                forbidden |= (1 << A)
        
        # Pick the smallest color not in forbidden
        # We can iterate 0..127
        found = False
        for c in range(128):
            if not (forbidden & (1 << c)):
                colors[i] = c
                found = True
                break
        
        if not found:
            # Fallback: pick 0 (should not happen if logic is correct)
            colors[i] = 0
    
    return colors
