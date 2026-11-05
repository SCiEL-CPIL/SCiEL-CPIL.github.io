    def try_swap_optimized(A_set, peak):
        A_list = sorted(A_set)
        m = len(A_list)
        
        # If m is large, sampling is better. If small, exhaustive is fine.
        # Let's use exhaustive for m <= 50.
        
        for x in A_list:
            # Compute counts without x
            A_minus_x = [a for a in A_list if a != x]
            counts_minus_x = get_counts(A_minus_x)
            
            # Check if removing x alone breaks coverage?
            # If counts_minus_x has 0s in target range, we MUST add a y that covers them.
            # But we are swapping, so we add y.
            
            # Try all y
            best_y = -1
            best_new_peak = peak
            
            for y in range(0, n+1):
                if y == x:
                    continue
                if y in A_set:
                    continue
                
                # New set is A_minus_x + {y}
                # New counts = counts_minus_x + counts_from_y
                # counts_from_y: y+a for a in A_minus_x, and 2y
                
                # We need to check if any target sum becomes 0.
                # And calculate new peak.
                
                # Efficient check:
                # 1. Identify sums that are 0 in counts_minus_x.
                # 2. These MUST be covered by y.
                # 3. If y doesn't cover all holes, skip.
                
                # Actually, it's easier to just compute the new counts and check.
                # But we can do it incrementally.
                
                # Let's just use the fast get_counts for the new set?
                # No, that's O(m^2).
                
                # Let's stick to the simpler `evaluate` for correctness first, 
                # and rely on the speed of numpy.
                # 3870 evals * 900 ops = 3.5M ops. 
                # 3.5M ops in Python is ~0.05s.
                # So 20 swaps take 1s.
                # We have 85s. So ~80 iterations.
                # This should be sufficient to find a good local minimum.
                
                new_A = A_set.copy()
                new_A.discard(x)
                new_A.add(y)
                new_peak, new_valid = evaluate(new_A)
                
                if new_valid and new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_y = y
            
            if best_y != -1:
                new_A = A_set.copy()
                new_A.discard(x)
                new_A.add(best_y)
                return new_A, best_new_peak
                
        return A_set, peak
