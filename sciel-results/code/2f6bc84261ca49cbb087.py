import time
from itertools import combinations

def _build_witt_blocks_fast():
    points = list(range(12))
    all_5subs = list(combinations(points, 5))
    all_6subs = list(combinations(points, 6))
    
    # Index mapping
    five_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # For each 6-subset, compute bitmask of 5-subsets it covers
    six_masks = []
    for b in all_6subs:
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << five_to_idx[s])
        six_masks.append(mask)
        
    # For each 5-subset, list of 6-subset indices that cover it
    five_to_sixes = [[] for _ in range(len(all_5subs))]
    for i, b in enumerate(all_6subs):
        for s in combinations(b, 5):
            five_to_sixes[five_to_idx[s]].append(i)
            
    covered = 0
    blocks = []
    
    # Symmetry breaking: fix first block to be (0,1,2,3,4,5)
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
        _WITTY_CACHE = _build_witt_blocks_fast()
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
