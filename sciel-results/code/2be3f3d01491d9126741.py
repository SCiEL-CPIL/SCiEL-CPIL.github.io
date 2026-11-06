import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Try to construct the Witt Design S(5,6,12) using PSL(2,11) action
    try:
        # Points are 0..10 and 11 (infinity)
        # Generate PSL(2,11) permutations
        perms = set()
        for a in range(11):
            for b in range(11):
                for c in range(11):
                    for d in range(11):
                        if (a * d - b * c) % 11 == 1:
                            # Check if this is a unique element in PSL(2,11)
                            # Since we are generating SL(2,11), M and -M give same perm.
                            # We can just collect all and dedupe.
                            P = [0] * 12
                            for x in range(11):
                                denom = (c * x + d) % 11
                                if denom == 0:
                                    P[x] = 11
                                else:
                                    # Inverse of denom mod 11
                                    inv_denom = pow(denom, 10, 11)
                                    P[x] = ((a * x + b) * inv_denom) % 11
                            # Handle x = infinity (11)
                            if c == 0:
                                P[11] = 11
                            else:
                                inv_c = pow(c, 10, 11)
                                P[11] = (a * inv_c) % 11
                            perms.add(tuple(P))
        
        # Expected 660 permutations
        if len(perms) != 660:
            raise ValueError(f"Generated {len(perms)} permutations, expected 660")
            
        # Base block candidate
        # Known block: {0, 1, 3, 4, 5, infinity}
        base_block = (0, 1, 3, 4, 5, 11)
        
        blocks = set()
        for P in perms:
            # Apply permutation to base block
            new_block = tuple(P[i] for i in base_block)
            blocks.add(frozenset(new_block))
            
        if len(blocks) == 132:
            # Verify validity (optional but safe)
            # Check that no 5-subset is repeated
            covered = set()
            valid = True
            for b in blocks:
                subs = list(combinations(sorted(b), 5))
                for s in subs:
                    if s in covered:
                        valid = False
                        break
                    covered.add(s)
                if not valid:
                    break
            if valid:
                return [list(b) for b in blocks]
            else:
                # Invalid block set? Should not happen if base block is correct
                pass
        
        # If generation failed or produced wrong count, fallback to search
        pass
        
    except Exception:
        pass

    # Fallback: Original logic (Backtracking or Greedy)
    # ... (keep the rest of the original code here) ...
    
    # Re-inserting the fallback logic from the original code
    n = 12
    all_blocks = list(combinations(range(n), 6))
    all_5_subs = list(combinations(range(n), 5))
    sub_to_idx = {s: i for i, s in enumerate(all_5_subs)}
    
    candidates_for_5 = [[] for _ in range(len(all_5_subs))]
    for b in all_blocks:
        for i in range(6):
            s = b[:i] + b[i+1:]
            idx = sub_to_idx[s]
            candidates_for_5[idx].append(b)

    covered = [False] * len(all_5_subs)
    
    def solve(covered, blocks):
        if time.time() > deadline:
            return None
            
        if len(blocks) == 132:
            return blocks
            
        target_idx = -1
        for i in range(len(covered)):
            if not covered[i]:
                target_idx = i
                break
                
        if target_idx == -1:
            return None

        for b in candidates_for_5[target_idx]:
            conflict = False
            for i in range(6):
                s = b[:i] + b[i+1:]
                if covered[sub_to_idx[s]]:
                    conflict = True
                    break
            
            if not conflict:
                for i in range(6):
                    s = b[:i] + b[i+1:]
                    covered[sub_to_idx[s]] = True
                
                res = solve(covered, blocks + [b])
                if res is not None:
                    return res
                
                for i in range(6):
                    s = b[:i] + b[i+1:]
                    covered[sub_to_idx[s]] = False
        return None

    first_block = (0, 1, 2, 3, 4, 5)
    for i in range(6):
        s = first_block[:i] + first_block[i+1:]
        covered[sub_to_idx[s]] = True
        
    result = solve(covered, [first_block])
    
    if result:
        return [list(b) for b in result]
    else:
        covered_g, blocks_g = set(), []
        for b in combinations(range(12), 6):
            subsets = list(combinations(b, 5))
            if not any(s in covered_g for s in subsets):
                blocks_g.append(list(b))
                covered_g.update(subsets)
        return blocks_g
