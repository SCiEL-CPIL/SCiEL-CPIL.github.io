import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # All 6-subsets
    all_blocks = list(combinations(range(12), 6))
    
    # Pre-filter blocks that are valid with the first two fixed blocks
    fixed = [
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 6, 7)
    ]
    
    covered = set()
    for b in fixed:
        for s in combinations(b, 5):
            covered.add(s)
            
    blocks = list(fixed)
    
    # Candidate blocks for the rest
    candidates = []
    for b in all_blocks:
        if b in fixed:
            continue
        # Check intersection with fixed blocks
        valid = True
        for fb in fixed:
            inter = len(set(b) & set(fb))
            if inter > 4:
                valid = False
                break
        if valid:
            # Check 5-subsets
            for s in combinations(b, 5):
                if s in covered:
                    valid = False
                    break
        if valid:
            candidates.append(b)
            
    # Sort candidates to improve pruning?
    # Lexicographic order is fine.
    
    def backtrack(idx, current_blocks, current_covered, current_candidates):
        if time.time() > deadline:
            return current_blocks
        
        if len(current_blocks) == 132:
            return current_blocks
            
        # Pick a candidate
        # Heuristic: pick the one that covers the most "rare" 5-subsets?
        # For now, just iterate.
        
        for i in range(len(current_candidates)):
            b = current_candidates[i]
            # Check against current_covered
            valid = True
            for s in combinations(b, 5):
                if s in current_covered:
                    valid = False
                    break
            
            if valid:
                new_covered = current_covered | set(combinations(b, 5))
                new_blocks = current_blocks + [b]
                
                # Filter candidates for next step
                new_candidates = []
                for cb in current_candidates[i+1:]:
                    # Check against new_block
                    inter = len(set(cb) & set(b))
                    if inter > 4:
                        continue
                    # Check 5-subsets
                    ok = True
                    for s in combinations(cb, 5):
                        if s in new_covered:
                            ok = False
                            break
                    if ok:
                        new_candidates.append(cb)
                        
                res = backtrack(idx + 1, new_blocks, new_covered, new_candidates)
                if res is not None and len(res) == 132:
                    return res
                    
        return None

    result = backtrack(0, blocks, covered, candidates)
    
    if result and len(result) == 132:
        return [list(b) for b in result]
    else:
        # Fallback to greedy
        covered_g, blocks_g = set(), []
        for b in all_blocks:
            subsets = list(combinations(b, 5))
            if not any(s in covered_g for s in subsets):
                blocks_g.append(list(b))
                covered_g.update(subsets)
        return blocks_g
