import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Map t-subsets to integers for fast set operations
    # v <= 24, so 5 bits per element is enough
    # t=3, so 15 bits total.
    def subset_to_id(sub):
        val = 0
        for i in range(t):
            val = (val << 5) | sub[i]
        return val

    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_subsets_ids = [subset_to_id(sub) for sub in all_t_subsets]
    uncovered_set = set(all_t_subsets_ids)
    
    best_blocks = None
    best_count = float('inf')
    
    def run_greedy():
        current_uncovered = set(uncovered_set)
        blocks = []
        
        while current_uncovered:
            if time.time() > deadline:
                return blocks
                
            anchor_id = min(current_uncovered)
            
            # Decode anchor
            anchor = []
            tmp = anchor_id
            for _ in range(t):
                anchor.append(tmp & 31)
                tmp >>= 5
            anchor = tuple(anchor)
            
            remaining_points = [x for x in range(v) if x not in anchor]
            
            best_block = None
            best_gain = -1
            
            # Iterate candidates
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(anchor + combo))
                
                gain = 0
                for sub in itertools.combinations(block, t):
                    if subset_to_id(sub) in current_uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                elif gain == best_gain and best_block is not None:
                    # Random tie-breaking
                    if random.random() < 0.5:
                        best_block = block

            if best_block is None:
                break
                
            blocks.append(list(best_block))
            
            for sub in itertools.combinations(best_block, t):
                sid = subset_to_id(sub)
                if sid in current_uncovered:
                    current_uncovered.discard(sid)
                    
        return blocks

    def remove_redundant(blocks):
        coverage_count = {sid: 0 for sid in all_t_subsets_ids}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                coverage_count[subset_to_id(sub)] += 1
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                b = blocks[i]
                can_remove = True
                for sub in itertools.combinations(b, t):
                    sid = subset_to_id(sub)
                    if coverage_count[sid] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for sub in itertools.combinations(b, t):
                        coverage_count[subset_to_id(sub)] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    attempts = 0
    while time.time() < deadline - 2.0:
        blocks = run_greedy()
        if not blocks:
            break
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()
            
        attempts += 1
        # If we found a very good solution, we can stop earlier?
        if best_count <= 76:
            if attempts > 5:
                break
                
    return best_blocks
