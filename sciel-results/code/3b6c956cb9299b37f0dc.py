<<<<<<< SEARCH
    def count_violations(S):
        """
        Count violations per element to help with removal.
        """
        S_list = list(S)
        counts = {x: 0 for x in S_list}
        if len(S_list) < 4:
            return counts
        
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            for m in range(start_m, end_m + 1):
                target = m * a
                others = [x for x in S_list if x != a]
                others_set = set(others)
                
                # Count how many triplets sum to target
                # We just need to know if a is involved in violations.
                # If a violation exists, increment count for a, b, c, d.
                # To avoid double counting the same triplet for different a's?
                # The definition is: a divides sum(b,c,d).
                # So (a,b,c,d) is a violation.
                
                found_any = False
                for i in range(len(others)):
                    b = others[i]
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d in others_set:
                            if not found_any:
                                # We could count all, but one is enough for "is there a violation"
                                # For weighting, let's just mark that a violation exists for these elements
                                # Actually, let's just return the first one to save time in removal phase
                                # But for "most problematic", we might want counts.
                                # Let's stick to finding the first violation for removal to keep it fast.
                                found_any = True
                                # Increment counts
                                counts[a] += 1
                                counts[b] += 1
                                counts[c] += 1
                                counts[d] += 1
                                break
                    if found_any:
                        break
        return counts
=======
    def count_violations(S):
        """
        Count violations per element to help with removal.
        """
        S_list = list(S)
        counts = {x: 0 for x in S_list}
        if len(S_list) < 4:
            return counts
        
        S_sorted = sorted(S_list)
        min_sum = S_sorted[0] + S_sorted[1] + S_sorted[2]
        max_sum = S_sorted[-1] + S_sorted[-2] + S_sorted[-3]
        
        for a in S_list:
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            if start_m > end_m:
                continue
            
            others = [x for x in S_list if x != a]
            
            # Build pair sum map for others
            pair_sum_map = {}
            for i in range(len(others)):
                b = others[i]
                for j in range(i + 1, len(others)):
                    c = others[j]
                    s = b + c
                    if s not in pair_sum_map:
                        pair_sum_map[s] = []
                    pair_sum_map[s].append((b, c))
            
            found_any = False
            for m in range(start_m, end_m + 1):
                target = m * a
                for d in others:
                    bc_sum = target - d
                    if bc_sum in pair_sum_map:
                        for (b, c) in pair_sum_map[bc_sum]:
                            if b != d and c != d:
                                if not found_any:
                                    found_any = True
                                    counts[a] += 1
                                    counts[b] += 1
                                    counts[c] += 1
                                    counts[d] += 1
                                break
                if found_any:
                    break
        return counts
>>>>>>> REPLACE

