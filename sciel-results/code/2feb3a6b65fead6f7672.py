    def polish(A, B, time_limit=2.0):
        """Try to insert skipped elements into existing sets."""
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        all_diffs = set()

        for i in range(len(A)):
            for j in range(len(A)):
                sums_A.add(A[i] + A[j])
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(abs(A[i] - A[j]))

        for i in range(len(B)):
            for j in range(len(B)):
                sums_B.add(B[i] + B[j])
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(abs(B[i] - B[j]))

        all_diffs = diffs_A | diffs_B

        start = time.time()
        used = set(A) | set(B)
        remaining = [x for x in range(1, n + 1) if x not in used]
        
        # MRV Heuristic: Sort by descending number of conflicts
        def count_conflicts(x):
            c = 0
            for a in A:
                if abs(x - a) in all_diffs:
                    c += 1
            for b in B:
                if abs(x - b) in all_diffs:
                    c += 1
            return c

        # Only sort if time permits and list is not too small
        if len(remaining) > 5:
            remaining.sort(key=count_conflicts, reverse=True)
        else:
            random.shuffle(remaining)

        for x in remaining:
            if time.time() - start > time_limit:
                break
            # Try the smaller set first
            try_A_first = len(A) <= len(B)
            sets_to_try = [(A, sums_A, diffs_A), (B, sums_B, diffs_B)] if try_A_first else [(B, sums_B, diffs_B), (A, sums_A, diffs_A)]
            
            added = False
            for target_set, target_sums, target_diffs in sets_to_try:
                if len(target_set) >= max_size:
                    continue
                valid = True
                for elem in target_set:
                    d = abs(x - elem)
                    if d in all_diffs:
                        valid = False
                        break
                if valid:
                    if 2 * x in target_sums:
                        valid = False
                    else:
                        for elem in target_set:
                            if x + elem in target_sums:
                                valid = False
                                break
                if valid:
                    for elem in target_set:
                        all_diffs.add(abs(x - elem))
                        target_sums.add(x + elem)
                    target_sums.add(2 * x)
                    target_set.append(x)
                    added = True
                    break

        return A, B
    