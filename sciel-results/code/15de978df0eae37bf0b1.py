        def solve(n_A, n_B, time_limit):
            # candidates_A and candidates_B are lists of numbers that are currently valid
            # to add to A and B respectively.
            # Initially, all numbers are valid?
            # No, we need to check against empty set. Empty set has no diffs.
            # So all 1..256 are valid initially.
            
            # But we must ensure A and B are disjoint?
            # "A and B need not themselves be disjoint."
            # So x can be in both A and B?
            # If x is in A, it uses diffs with A.
            # If x is in B, it uses diffs with B.
            # The diffs are global.
            # So if x is in A, it consumes diffs.
            # If we try to add x to B, we check diffs against B.
            # But the diffs consumed by A are in `used_diffs`.
            # So if x is in A, and we try to add x to B:
            # We check |x - b| for b in B.
            # These diffs must not be in `used_diffs`.
            # Since x is in A, diffs involving x and A are in `used_diffs`.
            # This doesn't directly block x from being in B, unless the diffs with B overlap with diffs in A or B.
            
            # So, we can have x in A and x in B.
            
            # Pruning:
            # If x is invalid for A (conflicts with A or used_diffs), we can remove it from candidates_A.
            # If x is invalid for B, remove from candidates_B.
            
            A = []
            B = []
            used_diffs = 0
            
            cand_A = list(range(1, 257))
            cand_B = list(range(1, 257))
            
            def is_valid_for_A(x):
                for a in A:
                    d = abs(x - a)
                    if used_diffs & (1 << d):
                        return False
                return True
                
            def is_valid_for_B(x):
                for b in B:
                    d = abs(x - b)
                    if used_diffs & (1 << d):
                        return False
                return True

            def backtrack():
                if time.time() > time_limit:
                    return False
                if len(A) == n_A and len(B) == n_B:
                    return True
                
                # Choose which set to extend
                # Heuristic: extend the one with fewer candidates?
                # Or just alternate/random.
                # Let's try to extend A if possible, else B.
                # But we need to explore.
                
                # To avoid deep recursion issues, we can use a stack or just recursion (depth 32 is fine).
                
                # Try A
                if len(A) < n_A:
                    # Filter candidates
                    # We can't easily filter in place without copying.
                    # Since N is small, just iterate.
                    for x in cand_A:
                        if not is_valid_for_A(x):
                            continue
                        
                        # Add x to A
                        A.append(x)
                        # Update used_diffs
                        for a in A:
                            if a != x:
                                used_diffs |= (1 << abs(x - a))
                                
                        if backtrack():
                            return True
                            
                        # Undo
                        A.pop()
                        for a in A:
                            used_diffs &= ~(1 << abs(x - a))
                            
                # Try B
                if len(B) < n_B:
                    for x in cand_B:
                        if not is_valid_for_B(x):
                            continue
                        
                        B.append(x)
                        for b in B:
                            if b != x:
                                used_diffs |= (1 << abs(x - b))
                                
                        if backtrack():
                            return True
                            
                        B.pop()
                        for b in B:
                            used_diffs &= ~(1 << abs(x - b))
                
                return False

            for restart in range(500):
                if time.time() > time_limit: break
                random.seed(restart)
                random.shuffle(cand_A)
                random.shuffle(cand_B)
                A.clear()
                B.clear()
                used_diffs = 0
                if backtrack():
                    return A, B
            return None
        