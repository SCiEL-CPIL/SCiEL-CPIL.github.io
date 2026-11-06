import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_solution = None
    best_D = float('inf')
    
    # missing_mask: bit i is 1 if (i+1) is missing
    # 1..30 corresponds to bits 0..29
    
    def backtrack(S, used_diffs, missing_mask, k, D_limit):
        if time.time() > deadline:
            return None
            
        if len(S) == k:
            if missing_mask == 0:
                return S
            return None
            
        rem_elements = k - len(S)
        max_new_diffs = rem_elements * len(S) + rem_elements * (rem_elements - 1) // 2
        # Count bits in missing_mask
        missing_count = bin(missing_mask).count('1')
        if max_new_diffs < missing_count:
            return None
            
        last = S[-1]
        
        candidates = set()
        
        # Generate candidates from missing diffs
        # For each s in S, and each missing diff d:
        # x = s + d
        # Check if x is valid
        for s in S:
            # Iterate over set bits in missing_mask
            m = missing_mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                d = lsb.bit_length() # 1-based index
                # Actually bit_length gives position.
                # If lsb is 1 (bit 0), bit_length is 1. d=1.
                # If lsb is 2 (bit 1), bit_length is 2. d=2.
                # So d = lsb.bit_length()
                
                x = s + d
                if x > last and x <= D_limit:
                    # Check validity
                    valid = True
                    new_diffs = []
                    for s2 in S:
                        diff = x - s2
                        if diff in used_diffs:
                            valid = False
                            break
                        new_diffs.append(diff)
                    if valid:
                        candidates.add((x, new_diffs))
                m ^= lsb
                
        # If no candidates, try to add any valid x
        if not candidates:
            for x in range(last + 1, D_limit + 1):
                valid = True
                new_diffs = []
                for s in S:
                    diff = x - s
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                if valid:
                    candidates.add((x, new_diffs))
                    
        # Sort candidates by x
        candidates = sorted(candidates, key=lambda item: item[0])
        
        for x, new_diffs in candidates:
            new_used = used_diffs | set(new_diffs)
            
            # Update missing_mask
            new_missing_mask = missing_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    bit = 1 << (d - 1)
                    new_missing_mask &= ~bit
                    
            res = backtrack(S + [x], new_used, new_missing_mask, k, D_limit)
            if res:
                return res
                
        return None

    # Try increasing D
    for k in range(9, 16):
        for D in range(30, 150):
            if time.time() > deadline:
                if best_solution:
                    return best_solution
                return [0, 1]
            
            S = [0, 1]
            used_diffs = {1}
            # missing: 2..30
            # bit 1 (d=2) to bit 29 (d=30)
            # bit 0 (d=1) is covered
            missing_mask = 0
            for d in range(2, 31):
                missing_mask |= (1 << (d - 1))
                
            res = backtrack(S, used_diffs, missing_mask, k, D)
            if res:
                if D < best_D:
                    best_D = D
                    best_solution = res
                break # Found minimal D for this k
                
        if time.time() > deadline:
            break
            
    if best_solution:
        return best_solution
    else:
        # Fallback
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
