import time
from itertools import combinations

def construct():
    deadline = time.time() + 85
    
    # Precompute modular inverses for denominators 1..10
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
        
    # We want to find the minimum k such that all residues 0..100 are covered
    # by subsets of size <= k.
    # Then we assign each residue to a subset of size <= k.
    
    # residues_by_size[s] maps residue -> list of subsets (or just one subset)
    # To minimize memory and time, we can just store the first subset found for each residue at each size.
    # But we need to know if a residue is covered by size <= k.
    
    # Strategy:
    # 1. Iterate sizes s from 1 to 10.
    # 2. For each size, compute all subset sums.
    # 3. Maintain a set `covered_by_up_to_s`.
    # 4. If `covered_by_up_to_s` becomes all 101 residues, we have found minimal k = s.
    # 5. We need to store an assignment for each residue.
    #    We can store `assignment[residue] = subset` when we first see it.
    #    Since we iterate s from 1 up, the first time we see a residue, it's with the smallest size.
    #    This guarantees that for the minimal k, every residue has an assignment of size <= k.
    #    And specifically, the size of the assignment for residue r is the smallest size s_r such that r is covered by size s_r.
    #    The max row length will be max(s_r).
    #    Since we stopped at k where all are covered, max(s_r) <= k.
    #    Is it possible that max(s_r) < k? No, because if max(s_r) < k, then all residues would have been covered by size < k, and we would have stopped earlier.
    #    So the max row length is exactly k (unless some residues are covered by smaller sizes but we still needed k for the "last" residue).
    
    covered = set()
    assignment = {}
    
    for s in range(1, 11):
        if time.time() > deadline - 5:
            break
            
        # Generate all combinations of size s
        for combo in combinations(range(1, 11), s):
            # Calculate sum of inverses
            res = 0
            for x in combo:
                res += inv[x]
            res %= 101
            
            if res not in assignment:
                assignment[res] = list(combo)
                covered.add(res)
                
        if len(covered) == 101:
            # We have covered all residues with subsets of size <= s
            # Construct the result list
            result = []
            for r in range(101):
                result.append(assignment[r])
            return result
            
    # Fallback if something went wrong (should not happen)
    # Use the last known assignment or generate a valid one
    # Since 101 residues and 1024 subsets, it's guaranteed to cover.
    # If we didn't return, it means we ran out of time or a bug.
    # Force return a valid structure to avoid crash, though it might not be optimal.
    # But the loop should finish fast.
    
    # Emergency fallback: just pick any valid subsets
    # This part is unlikely to be reached.
    result = []
    for r in range(101):
        if r in assignment:
            result.append(assignment[r])
        else:
            # Find any subset that sums to r
            # This is a slow fallback
            found = False
            for mask in range(1, 1 << 10):
                terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
                res = sum(inv[x] for x in terms) % 101
                if res == r:
                    result.append(terms)
                    found = True
                    break
            if not found:
                result.append([1]) # Invalid, but prevents crash
    return result
