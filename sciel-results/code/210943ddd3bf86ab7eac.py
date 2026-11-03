        def rank_perm_fast(p, n):
            # p is a tuple of ints 1..n
            used = 0
            rank = 0
            for i in range(n):
                val = p[i]
                # Count unused numbers smaller than val
                # Numbers are 1..n
                # Mask of numbers smaller than val: (1 << (val-1)) - 1
                mask_smaller = (1 << (val - 1)) - 1
                # Unused smaller: mask_smaller & ~used
                unused_smaller = mask_smaller & ~used
                # Count bits
                count = bin(unused_smaller).count('1')
                rank += count * math.factorial(n - i - 1)
                # Mark val as used
                used |= (1 << (val - 1))
            return rank
        