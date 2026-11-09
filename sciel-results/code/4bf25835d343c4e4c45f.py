    def is_bad(c0, c1, c2, c3):
        # Returns True if distinct count < 3
        # 1 distinct: all equal
        if c0 == c1 == c2 == c3:
            return True
        # 2 distinct:
        # Pairs equal:
        # 01, 02, 03, 12, 13, 23
        # If any pair equal, and not all equal, we might have 2 or 3.
        # To have 2 distinct, the set of values must have size 2.
        # This happens if:
        # - 3 are same, 1 different
        # - 2 are same, 2 are same (and the two pairs are different)
        # - 2 are same, 2 are different (one pair, two singles) -> Wait, 2 same, 2 singles means 3 distinct.
        
        # Let's just count distinct manually for speed
        d = 1
        if c1 != c0: d += 1
        if c2 != c0 and c2 != c1: d += 1
        if c3 != c0 and c3 != c1 and c3 != c2: d += 1
        return d < 3
