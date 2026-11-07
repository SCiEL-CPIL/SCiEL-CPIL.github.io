    def is_valid_addition(S, x):
        # 1. Check if x is a base in a violation
        # Triple (x, b, c) with b, c in S.
        # Sums: x+b, x+c, b+c.
        # Need x+b in S, x+c in S, b+c in S.
        # (Note: x+b, x+c, b+c cannot be x because b,c >= 1)
        for b in S:
            if x + b > 128: continue
            if x + b not in S: continue
            for c in S:
                if c == b: continue
                if x + c > 128: continue
                if x + c not in S: continue
                if b + c > 128: continue
                if b + c not in S: continue
                # Found violation where x is a base
                return False
        
        # 2. Check if x is a sum in a violation
        # Triple (a, b, c) with a, b, c in S.
        # One of the sums is x.
        # Case 2a: x = a+b.
        # Need a+c in S and b+c in S.
        # Iterate a in S. Let b = x - a.
        # If b in S and b != a:
        #   Check if there exists c in S, c != a, c != b such that a+c in S and b+c in S.
        
        for a in S:
            b = x - a
            if b <= 0 or b > 128: continue
            if b not in S: continue
            if a == b: continue
            # Found pair (a, b) summing to x.
            # Now find c.
            # To avoid duplicate work (a,b) vs (b,a), we can assume a < b.
            if a > b: continue
            
            # Check for c
            for c in S:
                if c == a or c == b: continue
                if a + c > 128: continue
                if a + c not in S: continue
                if b + c > 128: continue
                if b + c not in S: continue
                # Found violation where x is a sum (a+b)
                return False
        return True
