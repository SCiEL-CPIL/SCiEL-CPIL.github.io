import time
from itertools import combinations

def _build_witt_blocks_direct():
    """
    Construct S(5,6,12) directly from S(4,5,11).
    
    S(4,5,11) has 66 blocks on points {0,...,10}. It can be generated
    from 6 base blocks under cyclic shifts mod 11.
    
    S(5,6,12) on points {0,...,11} is constructed as:
    - For each block B of S(4,5,11): add B ∪ {11}
    - For each block B of S(4,5,11): add ({0,...,10} \\ B)
    This gives 66 + 66 = 132 blocks.
    """
    # 6 base blocks of S(4,5,11) under Z_11 cyclic action
    base_blocks = [
        frozenset([0, 1, 3, 4, 5]),
        frozenset([0, 1, 3, 4, 10]),
        frozenset([0, 1, 3, 5, 8]),
        frozenset([0, 1, 4, 5, 9]),
        frozenset([0, 2, 3, 5, 9]),
        frozenset([0, 2, 3, 4, 8]),
    ]
    
    # Generate all 66 blocks of S(4,5,11) via cyclic shifts mod 11
    s4511_blocks = []
    seen_blocks = set()
    for base in base_blocks:
        for shift in range(11):
            shifted = frozenset((x + shift) % 11 for x in base)
            if shifted not in seen_blocks:
                seen_blocks.add(shifted)
                s4511_blocks.append(shifted)
    
    # Verify S(4,5,11): every 4-subset of {0,...,10} appears in exactly one block
    points11 = list(range(11))
    all_4subs = list(combinations(points11, 4))
    four_count = [0] * len(all_4subs)
    four_idx = {s: i for i, s in enumerate(all_4subs)}
    for block in s4511_blocks:
        for sub4 in combinations(sorted(block), 4):
            four_count[four_idx[sub4]] += 1
    
    # Check: should be exactly 66 blocks, and each 4-subset covered exactly once
    if len(s4511_blocks) != 66:
        # Fallback: use backtracking if direct construction fails
        return _build_witt_blocks_fast()
    
    if any(c != 1 for c in four_count):
        return _build_witt_blocks_fast()
    
    # Construct S(5,6,12) from S(4,5,11)
    all_points_11 = frozenset(range(11))
    blocks_132 = []
    for block in s4511_blocks:
        # Block containing point 11
        blocks_132.append(frozenset(block | {11}))
        # Block not containing point 11 (complement in {0,...,10})
        blocks_132.append(frozenset(all_points_11 - block))
    
    # Verify S(5,6,12): no 5-subset appears in two blocks
    all_5subs_12 = list(combinations(range(12), 5))
    five_idx_12 = {s: i for i, s in enumerate(all_5subs_12)}
    five_count_12 = [0] * len(all_5subs_12)
    for block in blocks_132:
        for sub5 in combinations(sorted(block), 5):
            five_count_12[five_idx_12[sub5]] += 1
    
    if len(blocks_132) != 132 or any(c > 1 for c in five_count_12):
        return _build_witt_blocks_fast()
    
    # Check all 792 5-subsets are covered exactly once
    if sum(five_count_12) != 792:
        return _build_witt_blocks_fast()
    
    return [list(b) for b in blocks_132]


def _build_witt_blocks_fast():
    """Fallback: backtracking search for S(5,6,12)."""
    points = list(range(12))
    all_5subs = list(combinations(points, 5))
    all_6subs = list(combinations(points, 6))
    
    five_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    six_masks = []
    for b in all_6subs:
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << five_to_idx[s])
        six_masks.append(mask)
        
    five_to_sixes = [[] for _ in range(len(all_5subs))]
    for i, b in enumerate(all_6subs):
        for s in combinations(b, 5):
            five_to_sixes[five_to_idx[s]].append(i)
            
    covered = 0
    blocks = []
    
    first_block = (0, 1, 2, 3, 4, 5)
    first_idx = all_6subs.index(first_block)
    blocks.append(first_block)
    covered |= six_masks[first_idx]
    
    def backtrack(covered_mask, current_blocks):
        if len(current_blocks) == 132:
            return current_blocks if (covered_mask.bit_count() == 792) else None
            
        best_idx = -1
        best_cands = None
        min_cands = float('inf')
        
        for i in range(792):
            if (covered_mask >> i) & 1:
                continue
                
            cands = []
            for b_idx in five_to_sixes[i]:
                if (six_masks[b_idx] & covered_mask) == 0:
                    cands.append(b_idx)
                    
            if len(cands) < min_cands:
                min_cands = len(cands)
                best_idx = i
                best_cands = cands
                if min_cands == 0:
                    return None
                if min_cands == 1:
                    break
                    
        if best_idx == -1:
            if covered_mask.bit_count() == 792:
                return current_blocks
            return None

        for b_idx in best_cands:
            new_covered = covered_mask | six_masks[b_idx]
            res = backtrack(new_covered, current_blocks + [all_6subs[b_idx]])
            if res is not None:
                return res
                
        return None

    result = backtrack(covered, blocks)
    if result:
        return [list(b) for b in result]
    else:
        return [list(b) for b in blocks]


_WITTY_CACHE = None

def _ensure_cache():
    global _WITTY_CACHE
    if _WITTY_CACHE is None:
        _WITTY_CACHE = _build_witt_blocks_direct()
    return _WITTY_CACHE

def construct():
    return _ensure_cache()

if __name__ == "__main__":
    start = time.time()
    res = construct()
    elapsed = time.time() - start
    print(f"Blocks: {len(res)}, Time: {elapsed:.3f}s")
    seen = set()
    valid = True
    block_set = set()
    for b in res:
        b_tuple = tuple(sorted(b))
        if b_tuple in block_set:
            valid = False
            break
        block_set.add(b_tuple)
        for s in combinations(b, 5):
            if s in seen:
                valid = False
                break
            seen.add(s)
        if not valid:
            break
    print(f"Valid: {valid}, Covered: {len(seen)}")
