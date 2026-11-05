    def try_add(x, S, used_sums, pair_sums):
        new_sums = set()
        
        # 3x
        new_sums.add(3*x)
        
        # 2x + y
        for y in S:
            new_sums.add(2*x + y)
            
        # x + pair
        for p in pair_sums:
            new_sums.add(x + p)
            
        expected_count = 1 + len(S) + len(pair_sums)
        if len(new_sums) != expected_count:
            return False
            
        if new_sums & used_sums:
            return False
            
        # Valid
        used_sums.update(new_sums)
        S.append(x)
        
        # Update pair_sums
        # New pairs: x+x, x+y
        new_pairs = {2*x}
        for y in S[:-1]: # S[:-1] is the old S
            new_pairs.add(x + y)
        pair_sums.update(new_pairs)
        
        return True
