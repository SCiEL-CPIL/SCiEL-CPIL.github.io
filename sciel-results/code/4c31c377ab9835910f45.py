import time
from itertools import combinations

def construct():
    """
    Construct the Witt design S(5,6,12) using a deterministic algebraic
    construction based on the extended binary Golay code / Mathieu group M12.
    
    The S(5,6,12) design is unique and has exactly 132 blocks. We construct
    it using the known explicit construction from the Golay code structure.
    
    Approach: Use the fact that the 132 blocks can be generated from the
    action of the Mathieu group M12 on 12 points. We use a known explicit
    set of generator blocks and their orbits.
    
    A very reliable method: Use the construction via the projective line
    over GF(11) plus infinity, with the specific group action.
    
    Actually, the most reliable approach for guaranteed correctness is to
    use the known explicit list or a verified algebraic construction.
    
    Here we use a construction based on the following:
    - The 12 points are labeled 0 through 11
    - We use the known fact that S(5,6,12) can be constructed from
      the weight-6 words in the [12,6,4] code (the "half" of the Golay code)
    
    The [12,6,4] code C is the set of all vectors in F_2^12 that can be
    obtained by the following construction from the Golay code.
    
    A cleaner approach: Use the explicit construction from the icosahedral
    symmetry. The 12 points correspond to the 12 vertices of an icosahedron,
    and the 132 blocks correspond to specific 6-vertex subsets.
    
    Most practical: Use a known explicit construction. The blocks of
    S(5,6,12) can be generated as follows:
    
    Method 1: Start with the 11 points of PG(3,2) plus one extra point,
    but this doesn't directly give S(5,6,12).
    
    Method 2: Use the construction from the binary Golay code.
    The extended Golay code G24 has 24 coordinates. The S(5,6,12) design
    is related to the "halves" of the Golay code.
    
    Let me use a different, very clean construction:
    
    The S(5,6,12) design can be constructed from the S(4,5,11) design.
    The S(4,5,11) design has 66 blocks (5-subsets of 11 points).
    S(5,6,12) is obtained by adding a 12th point and taking:
    - For each 5-block B of S(4,5,11): the 6-set B ∪ {∞}
    - For each 5-block B of S(4,5,11): the 6-set (complement of B in the 11 points)
    
    Wait, that gives 66 + 66 = 132 blocks! Let me verify:
    - S(4,5,11) is a Steiner system with parameters (v=11, k=5, t=4, λ=1)
    - It has C(11,4)/C(5,4) = 330/5 = 66 blocks
    - Each block is a 5-subset of {0,1,...,10}
    - Adding point 11 (∞):
      - Type A: B ∪ {11} for each 5-block B → 66 blocks
      - Type B: {0,...,10} \ B for each 5-block B → 66 blocks (complements are also 6-subsets)
    
    Wait, the complement of a 5-subset of 11 points is a 6-subset of 11 points.
    So Type B gives 6-subsets of {0,...,10} (not including point 11).
    
    Let's check the S(5,6,12) property:
    - We need: every 5-subset of {0,...,11} is contained in exactly one block.
    
    For a 5-subset S of {0,...,11}:
    - If 11 ∈ S: Let S' = S \ {11}, a 4-subset of {0,...,10}.
      In S(4,5,11), there's a unique 5-block B containing S'.
      Then S = S' ∪ {11} ⊂ B ∪ {11}, which is a Type A block.
      Could S be in another block? 
      - Type A: B' ∪ {11} where S' ⊂ B'. Since S' is a 4-subset and B' is a 5-block,
        the unique 5-block containing S' is B, so B' = B. Unique.
      - Type B: C where C = {0,...,10} \ B' for some 5-block B'. 
        S ⊂ C means S' ∪ {11} ⊂ C. But C ⊂ {0,...,10}, so 11 ∉ C. Contradiction.
      So exactly one block. ✓
      
    - If 11 ∉ S: S is a 5-subset of {0,...,10}.
      - Type A: B ∪ {11} where S ⊂ B ∪ {11}. Since 11 ∉ S, S ⊂ B. But B is a 5-set
        and S is a 5-set, so S = B. So S is in B ∪ {11} where B = S.
        But B must be a 5-block of S(4,5,11). Is S a 5-block? Not necessarily.
        So S is in Type A block B ∪ {11} iff S itself is a 5-block of S(4,5,11).
      - Type B: C = {0,...,10} \ B' for some 5-block B'. S ⊂ C means S ∩ B' = ∅.
        So B' is a 5-block disjoint from S.
        
      So S is in:
      - Type A block S ∪ {11} iff S is a 5-block of S(4,5,11)
      - Type B blocks {0,...,10} \ B' for each 5-block B' disjoint from S
      
      For S(5,6,12), we need exactly one block containing S.
      
      Case 1: S is a 5-block of S(4,5,11).
        Then S is in S ∪ {11} (Type A).
        Is S in any Type B block? S ⊂ {0,...,10} \ B' means S ∩ B' = ∅.
        In S(4,5,11), how many 5-blocks are disjoint from a given 5-block S?
        Total 5-subsets of {0,...,10} \ S (which has 6 elements): C(6,5) = 6.
        But not all 6 are 5-blocks. In S(4,5,11), the number of 5-blocks disjoint
        from a given 5-block... 
        
        Actually, this construction might not work directly. Let me reconsider.
        
    Let me use a different, verified construction.
    
    The most reliable approach: hardcode the 132 blocks using a known
    explicit construction, or use the algebraic construction via GF(4) or GF(11).
    
    Here's a clean construction using the field GF(11):
    
    Points: Z_11 ∪ {∞} = {0, 1, 2, ..., 10, ∞}
    We label ∞ as 11.
    
    The blocks of S(5,6,12) can be described using the Mathieu group M12.
    
    Actually, let me use the most practical approach that is guaranteed to work:
    Use a known explicit list of the 132 blocks, generated from the algebraic
    construction.
    
    Here's a verified construction:
    
    The 132 blocks of S(5,6,12) can be generated from the following:
    - 11 "base" blocks from the S(4,5,11) design
    - Plus their translates under Z_11
    
    The S(4,5,11) design can be constructed as follows:
    Points: Z_11
    Blocks: For each a ∈ Z_11, the 5-set {a, a+1, a+3, a+4, a+5} mod 11
    Wait, that's only 11 blocks. We need 66.
    
    Let me use the standard construction of S(4,5,11):
    The 66 blocks of S(4,5,11) on Z_11 are:
    - 11 blocks: {i, i+1, i+3, i+4, i+5} + 11i for i = 0,...,10 (mod 11)
      Wait, that's not right either.
    
    Let me just use a different approach entirely. I'll use the fact that
    the Witt design S(5,6,12) can be constructed from the extended Golay code,
    and implement a direct algebraic construction.
    
    The extended binary Golay code G24 ⊂ F_2^24 is a [24,12,8] code.
    The S(5,6,12) design is related to the "halves" of the Golay code.
    
    A specific construction:
    1. Identify the 12 points with the cosets of a subgroup in M24.
    2. The blocks are the supports of certain codewords.
    
    This is getting too complex. Let me use the most practical approach:
    
    Use a known explicit construction based on the following:
    
    The 132 blocks of S(5,6,12) on points {0,1,...,11} can be constructed
    as follows:
    
    Define the 11 "base" 5-subsets of {0,1,...,10} that form part of S(4,5,11):
    These are the 5-subsets that are the supports of weight-5 words in a
    certain code.
    
    Actually, the simplest reliable method: use a precomputed list.
    The 132 blocks of S(5,6,12) are well-known. Let me generate them
    using a verified algorithm.
    
    Here's a clean approach using the "doubling" construction:
    
    Step 1: Construct S(4,5,11) on points {0,1,...,10}.
    Step 2: Add point 11.
    Step 3: The 132 blocks of S(5,6,12) are:
      - For each 5-block B of S(4,5,11): B ∪ {11}
      - For each 5-block B of S(4,5,11): ({0,...,10} \ B)
    
    Let me verify this more carefully.
    
    S(4,5,11) has 66 blocks. Let's call them B_1, ..., B_66.
    Type A blocks: A_i = B_i ∪ {11} for i=1,...,66 (66 blocks)
    Type B blocks: C_i = {0,...,10} \ B_i for i=1,...,66 (66 blocks)
    
    Total: 132 blocks. ✓
    
    Now check: every 5-subset of {0,...,11} is in exactly one block.
    
    Case 1: 5-subset S contains 11.
    S = S' ∪ {11} where S' ⊂ {0,...,10}, |S'|=4.
    - Type A: S ⊂ A_i = B_i ∪ {11} iff S' ⊂ B_i.
      In S(4,5,11), there's a unique B_i containing S'. So S is in exactly one Type A block.
    - Type B: S ⊂ C_i = {0,...,10} \ B_i. But C_i ⊂ {0,...,10}, so 11 ∉ C_i.
      Thus S ⊄ C_i for any i.
    Total: exactly 1 block. ✓
    
    Case 2: 5-subset S ⊂ {0,...,10}.
    - Type A: S ⊂ A_i = B_i ∪ {11} iff S ⊂ B_i (since 11 ∉ S).
      |S|=5, |B_i|=5, so S = B_i. Thus S is in A_i iff S is one of the 5-blocks.
    - Type B: S ⊂ C_i = {0,...,10} \ B_i iff S ∩ B_i = ∅.
    
    So S is in:
    - A_S (if S is a 5-block of S(4,5,11))
    - C_i for each 5-block B_i disjoint from S
    
    We need exactly one block total.
    
    If S is a 5-block: S is in A_S. How many B_i are disjoint from S?
    In S(4,5,11), the number of 5-blocks disjoint from a given 5-block S:
    The complement {0,...,10} \ S has 6 elements. The 5-blocks disjoint from S
    are 5-subsets of these 6 elements that are also 5-blocks of S(4,5,11).
    
    In S(4,5,11), for any 4-subset T of {0,...,10}, there's a unique 5-block containing T.
    The 5-subsets of {0,...,10} \ S are: there are C(6,5)=6 of them.
    Each is a 5-subset of the 6-element complement.
    For each such 5-subset U, is U a 5-block of S(4,5,11)?
    U is a 5-block iff for every 4-subset T ⊂ U, T is contained in a unique 5-block,
    and that 5-block is U. But by the S(4,5,11) property, each 4-subset T is in
    exactly one 5-block. So U is a 5-block iff all 5 of its 4-subsets are contained
    in U as their unique 5-block.
    
    This is getting complicated. Let me just check: in S(4,5,11), how many 5-blocks
    are disjoint from a given 5-block?
    
    Total 5-blocks: 66.
    5-blocks intersecting S: For each 4-subset T ⊂ S, the unique 5-block containing T
    intersects S in at least 4 elements. There are C(5,4)=5 such 4-subsets.
    Each 5-block containing a 4-subset of S is counted. But some 5-blocks may contain
    multiple 4-subsets of S.
    
    Actually, a 5-block B intersects S in 0,1,2,3,4, or 5 elements.
    - If |B ∩ S| = 5: B = S (1 block, if S is a 5-block)
    - If |B ∩ S| = 4: B contains a 4-subset of S. There are C(5,4)=5 such 4-subsets,
      each in a unique 5-block. So at most 5 such blocks (could be fewer if some coincide,
      but they can't since they contain different 4-subsets... actually they could contain
      the same 5-block if two different 4-subsets of S are in the same 5-block, which
      means that 5-block contains both 4-subsets, hence contains their union which is S,
      so it's S itself. So the 5 blocks with |B ∩ S|=4 are distinct from S and from each other.)
      Wait, if B contains two different 4-subsets T1 and T2 of S, then B contains T1 ∪ T2 = S,
      so B = S. So no 5-block other than S can contain two different 4-subsets of S.
      Thus there are exactly 5 5-blocks with |B ∩ S| = 4 (one for each 4-subset of S).
    - If |B ∩ S| ≤ 3: B is disjoint from at least 2 elements of S.
    
    Total 5-blocks intersecting S: 1 (S itself) + 5 (with |B∩S|=4) + (those with |B∩S|≤3).
    
    This is getting too involved. Let me just trust the construction and verify
    computationally.
    
    Actually, I recall that this construction is standard and correct. The S(5,6,12)
    design is indeed obtained from S(4,5,11) by this "doubling" method. Let me
    just implement it and verify.
    
    First, I need to construct S(4,5,11). The standard construction:
    
    S(4,5,11) on points Z_11:
    The 66 blocks can be generated as follows:
    - Base blocks: {0,1,3,4,5}, {0,1,3,7,9}, {0,2,3,5,6}, {0,2,4,5,7}, {0,2,4,6,9}
    - Take all translates mod 11: for each base block B and each a ∈ Z_11, B+a mod 11.
    
    That gives 5 × 11 = 55 blocks. We need 66. So we need more base blocks.
    
    Actually, the standard construction uses the following:
    The 66 blocks of S(4,5,11) are:
    - 11 blocks from each of 6 orbits under Z_11, but with some orbits having
      smaller size due to symmetry.
    
    Let me use a different, verified construction of S(4,5,11):
    
    The S(4,5,11) design is the unique Steiner system with these parameters.
    It can be constructed as the set of 5-subsets of Z_11 that are the supports
    of weight-5 codewords in a certain [11,6,5] code.
    
    Alternatively, here's a concrete construction:
    
    The 66 blocks of S(4,5,11) on {0,1,...,10} are:
    
    For a = 0,1,...,10:
      {a, a+1, a+3, a+4, a+5} mod 11
      {a, a+1, a+3, a+7, a+9} mod 11  
      {a, a+2, a+3, a+5, a+6} mod 11
      {a, a+2, a+4, a+5, a+7} mod 11
      {a, a+2, a+4, a+6, a+9} mod 11
    
    That's 5 × 11 = 55 blocks. We're missing 11 more.
    
    The remaining 11 blocks are:
      {a, a+1, a+4, a+6, a+10} mod 11 for a = 0,...,10
    
    Let me verify: 55 + 11 = 66. ✓
    
    Let me check that these form S(4,5,11). I'll verify computationally in the code.
    
    Now, with S(4,5,11) constructed, the S(5,6,12) blocks are:
    - For each 5-block B of S(4,5,11): sorted(B ∪ {11})
    - For each 5-block B of S(4,5,11): sorted({0,...,10} \ B)
    
    Let me implement this and verify.
    """
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Construct S(4,5,11) on points 0..10
    # Base blocks and their translates mod 11
    base_blocks = [
        (0, 1, 3, 4, 5),
        (0, 1, 3, 7, 9),
        (0, 2, 3, 5, 6),
        (0, 2, 4, 5, 7),
        (0, 2, 4, 6, 9),
        (0, 1, 4, 6, 10),
    ]
    
    s4511_blocks = set()
    for base in base_blocks:
        for a in range(11):
            block = tuple(sorted((x + a) % 11 for x in base))
            s4511_blocks.add(block)
    
    # Verify we have 66 distinct blocks
    assert len(s4511_blocks) == 66, f"Expected 66 S(4,5,11) blocks, got {len(s4511_blocks)}"
    
    # Verify S(4,5,11) property: every 4-subset is in exactly one block
    all_4subsets = list(combinations(range(11), 4))
    assert len(all_4subsets) == 330
    
    count_per_4subset = {}
    for block in s4511_blocks:
        for sub4 in combinations(block, 4):
            count_per_4subset[sub4] = count_per_4subset.get(sub4, 0) + 1
    
    # Every 4-subset should appear exactly once
    for sub4 in all_4subsets:
        assert count_per_4subset.get(sub4, 0) == 1, f"4-subset {sub4} appears {count_per_4subset.get(sub4, 0)} times"
    
    # Construct S(5,6,12) blocks
    s5612_blocks = set()
    
    # Type A: B ∪ {11}
    for B in s4511_blocks:
        block = tuple(sorted(list(B) + [11]))
        s5612_blocks.add(block)
    
    # Type B: {0,...,10} \ B
    full_set = frozenset(range(11))
    for B in s4511_blocks:
        comp = tuple(sorted(full_set - set(B)))
        s5612_blocks.add(comp)
    
    # Verify we have 132 distinct blocks
    assert len(s5612_blocks) == 132, f"Expected 132 S(5,6,12) blocks, got {len(s5612_blocks)}"
    
    # Verify S(5,6,12) property: every 5-subset is in exactly one block
    all_5subsets = list(combinations(range(12), 5))
    assert len(all_5subsets) == 792
    
    count_per_5subset = {}
    for block in s5612_blocks:
        for sub5 in combinations(block, 5):
            count_per_5subset[sub5] = count_per_5subset.get(sub5, 0) + 1
    
    # Every 5-subset should appear exactly once
    for sub5 in all_5subsets:
        cnt = count_per_5subset.get(sub5, 0)
        assert cnt == 1, f"5-subset {sub5} appears {cnt} times"
    
    # Convert to list of lists
    result = [list(b) for b in s5612_blocks]
    
    return result
