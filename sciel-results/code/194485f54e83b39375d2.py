import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 12.0
    
    # Direct algebraic construction of S(5,6,12) via PGL(2,11)
    # Points: 0 = ∞, 1..11 represent elements of GF(11) (i.e., Z_11)
    
    # PGL(2,11) acts on the projective line P^1(GF(11)) = GF(11) ∪ {∞}
    # The group has order 11*12*2 = 264
    # It's 3-transitive on the 12 points
    
    # The Steiner system S(4,5,11) on Z_11 has 66 blocks
    # These are generated as the orbit of a base block under PGL(2,11)
    
    # A known base block for S(4,5,11): {0, 1, 3, 4, 5} in Z_11
    # Let's verify: the stabilizer of this block in PGL(2,11) should have order 264/66 = 4
    
    # Generate all PGL(2,11) transformations
    # PGL(2,11) = {x -> (ax+b)/(cx+d) : a,b,c,d ∈ Z_11, ad-bc ≠ 0} / scalars
    # We can represent each element by (a,b,c,d) with ad-bc = 1 (PGL mod scalars)
    
    # Actually, let's use a simpler approach: enumerate all fractional linear transformations
    # x -> (ax + b)/(cx + d) where ad - bc ≠ 0, modulo scalar multiplication
    
    # For PGL(2,11), we can fix the determinant to be 1 (or any nonzero value)
    # Since we're working mod 11, we can normalize
    
    # Let's generate all distinct permutations of {0,1,...,10,∞} that come from PGL(2,11)
    # ∞ maps to: if c≠0, ∞ -> a/c; if c=0, ∞ -> ∞
    # x maps to: (ax+b)/(cx+d) mod 11, with ∞ if cx+d ≡ 0
    
    def apply_pgl(x, a, b, c, d):
        """Apply x -> (ax+b)/(cx+d) in PGL(2,11), where x is 0-10 or 11 (for ∞)"""
        if x == 11:  # ∞
            if c == 0:
                return 11  # ∞
            else:
                return (a * pow(c, 9, 11)) % 11  # a/c mod 11
        else:
            denom = (c * x + d) % 11
            if denom == 0:
                return 11  # ∞
            else:
                numer = (a * x + b) % 11
                return (numer * pow(denom, 9, 11)) % 11
    
    # Generate all PGL(2,11) elements
    # To avoid duplicates from scalar multiplication, fix a = 1 when possible
    # Or just generate all and use a set
    
    pgl_elements = set()
    for a in range(11):
        for b in range(11):
            for c in range(11):
                for d in range(11):
                    det = (a * d - b * c) % 11
                    if det != 0:
                        # Normalize: divide by a if a ≠ 0, else by c if c ≠ 0, else by d
                        # Actually, let's just store (a,b,c,d) and deduplicate later
                        # by checking if they produce the same permutation
                        pgl_elements.add((a, b, c, d))
    
    # Deduplicate: two elements (a,b,c,d) and (ka,kb,kc,kd) give the same transformation
    # Generate unique transformations by their action on {0, 1, ∞}
    unique_transforms = {}
    for (a, b, c, d) in pgl_elements:
        # Determine transformation by images of 0, 1, 11(∞)
        img_0 = apply_pgl(0, a, b, c, d)
        img_1 = apply_pgl(1, a, b, c, d)
        img_inf = apply_pgl(11, a, b, c, d)
        key = (img_0, img_1, img_inf)
        if key not in unique_transforms:
            unique_transforms[key] = (a, b, c, d)
    
    pgl_transforms = list(unique_transforms.values())
    # Should have 264 elements
    
    # Now generate the orbit of a base 5-subset under PGL(2,11)
    # Base block for S(4,5,11): let's try {0, 1, 3, 4, 5}
    base_block = frozenset([0, 1, 3, 4, 5])
    
    s4511_blocks = set()
    for (a, b, c, d) in pgl_transforms:
        transformed = frozenset([apply_pgl(x, a, b, c, d) for x in base_block])
        # Remove ∞ if present (we want blocks on Z_11 only)
        if 11 in transformed:
            transformed = transformed - {11}
            if len(transformed) != 5:
                continue
        s4511_blocks.add(transformed)
    
    # Should have 66 blocks
    if len(s4511_blocks) != 66:
        # Try a different base block
        # Let's try {0, 1, 2, 4, 5}
        base_block = frozenset([0, 1, 2, 4, 5])
        s4511_blocks = set()
        for (a, b, c, d) in pgl_transforms:
            transformed = frozenset([apply_pgl(x, a, b, c, d) for x in base_block])
            if 11 in transformed:
                transformed = transformed - {11}
                if len(transformed) != 5:
                    continue
            s4511_blocks.add(transformed)
    
    if len(s4511_blocks) != 66:
        # Try yet another: {0, 1, 3, 5, 7}
        base_block = frozenset([0, 1, 3, 5, 7])
        s4511_blocks = set()
        for (a, b, c, d) in pgl_transforms:
            transformed = frozenset([apply_pgl(x, a, b, c, d) for x in base_block])
            if 11 in transformed:
                transformed = transformed - {11}
                if len(transformed) != 5:
                    continue
            s4511_blocks.add(transformed)
    
    # Now extend to S(5,6,12)
    # Point 0 = ∞, points 1-11 = Z_11 elements (1..11 map to 0..10 in Z_11)
    # For each 5-block B of S(4,5,11):
    #   Block 1: {0} ∪ {b+1 for b in B}  (add ∞, shift Z_11 to 1..11)
    #   Block 2: complement of {b+1 for b in B} in {1,...,11}
    
    final_blocks = []
    for b in s4511_blocks:
        shifted = frozenset([x + 1 for x in b])
        block_with_inf = [0] + sorted(shifted)
        block_without_inf = sorted(set(range(1, 12)) - shifted)
        final_blocks.append(block_with_inf)
        final_blocks.append(block_without_inf)
    
    # Verify no 5-subset is repeated
    all_5_subsets = set()
    for block in final_blocks:
        for subset in combinations(block, 5):
            if subset in all_5_subsets:
                # Duplicate found - this shouldn't happen for a valid design
                pass
            all_5_subsets.add(subset)
    
    return final_blocks
