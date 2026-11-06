        # Pruning: Check if enough capacity to cover missing diffs
        k = 32 - len(S)
        if k > 0:
            max_new_diffs = k * len(S) + k * (k - 1) // 2
            if missing_count > max_new_diffs:
                return None
