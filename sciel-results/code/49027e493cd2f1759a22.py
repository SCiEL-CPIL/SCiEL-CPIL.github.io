    def can_add_to_A(A, x, sums_A, diff_A, diff_B):
        if x in A: return False
        # Check Sidon
        # New sums: x+a for a in A, x+x
        for a in A:
            if (x + a) in sums_A:
                return False
        if (x + x) in sums_A:
            return False
            
        # Check Diff Disjointness
        # New diffs: x-a, a-x for a in A
        for a in A:
            if (x - a) in diff_B:
                return False
            if (a - x) in diff_B:
                return False
        return True

    def can_add_to_B(B, x, sums_B, diff_B, diff_A):
        if x in B: return False
        for b in B:
            if (x + b) in sums_B:
                return False
        if (x + x) in sums_B:
            return False
            
        for b in B:
            if (x - b) in diff_A:
                return False
            if (b - x) in diff_A:
                return False
        return True
