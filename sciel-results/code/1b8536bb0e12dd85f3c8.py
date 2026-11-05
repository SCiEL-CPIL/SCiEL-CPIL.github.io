    def greedy_init():
        A = []
        covered = np.zeros(num_targets, dtype=bool)
        
        for _ in range(k):
            best_x = 0
            best_gain = -1
            # Try all x in 0..128
            for x in range(0, N + 1):
                # Calculate new sums
                # x + x
                # x + a for a in A
                new_sums = [x + x]
                for a in A:
                    new_sums.append(x + a)
                    
                gain = 0
                for s in new_sums:
                    if min_sum <= s <= max_sum:
                        idx = s - min_sum
                        if not covered[idx]:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            A.append(best_x)
            # Update covered
            s = best_x + best_x
            if min_sum <= s <= max_sum:
                covered[s - min_sum] = True
            for a in A[:-1]: # Exclude self? No, pairs are unordered, but we just added x.
                 # Wait, the pair (x, x) is already handled.
                 # Pairs (x, a) for a in previous A.
                 s = best_x + a
                 if min_sum <= s <= max_sum:
                     covered[s - min_sum] = True
                     
        return A
