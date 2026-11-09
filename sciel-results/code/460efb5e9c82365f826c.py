    def evaluate_fast(S, target_range_set):
        n = len(S)
        num_diffs = n * (n - 1) // 2
        diff_counts = {}
        valid_sidon = True
        # Build diff counts
        for i in range(n):
            si = S[i]
            for j in range(i+1, n):
                d = S[j] - si
                if d in diff_counts:
                    diff_counts[d] += 1
                    if diff_counts[d] > 1:
                        valid_sidon = False
                else:
                    diff_counts[d] = 1
        
        if not valid_sidon:
            # Count how many diffs are duplicated
            # Or just return a high penalty
            # Number of "extra" diffs = total pairs - unique diffs
            # But we want to guide towards uniqueness.
            # Penalty: sum of (count-1) for all counts > 1
            pen = sum(c-1 for c in diff_counts.values() if c > 1)
            return 1000000 * pen + max(S)
            
        # Check coverage
        missing = 0
        for d in target_range_set:
            if d not in diff_counts:
                missing += 1
                
        if missing > 0:
            return 10000 * missing + max(S)
            
        return max(S)
