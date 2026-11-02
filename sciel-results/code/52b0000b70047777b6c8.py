import itertools
import random
import time
from collections import Counter

def construct(v, k, t):
    deadline = time.time() + 145
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)
    
    def get_tsub_indices(block):
        idxs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(tuple(ts))
            if idx is not None:
                idxs.append(idx)
        return idxs

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        cover_count = Counter()
        block_tsubs = []
        for block in blocks:
            tsubs = get_tsub_indices(block)
            block_tsubs.append(tsubs)
            for idx in tsubs:
                cover_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not any(cover_count[idx] == 1 for idx in block_tsubs[i]):
                    for idx in block_tsubs[i]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    changed = True
                    break
        return blocks

    def local_search(blocks):
        blocks = remove_redundant(blocks)
        # Maintain cover_count and block_tsubs for fast swap checks
        cover_count = Counter()
        block_tsubs = []
        for block in blocks:
            tsubs = get_tsub_indices(block)
            block_tsubs.append(tsubs)
            for idx in tsubs:
                cover_count[idx] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                B = blocks[i]
                B_set = set(B)
                B_tsubs = block_tsubs[i]
                
                # Try swapping p in B with q not in B
                for p in B:
                    for q in range(v):
                        if q in B_set:
                            continue
                        
                        # Calculate lost and gained t-subsets
                        # Lost: t-subsets containing p and t-1 from B\{p}
                        # Gained: t-subsets containing q and t-1 from B\{p}
                        
                        lost_idxs = set()
                        gained_idxs = set()
                        
                        rest_points = [x for x in B if x != p]
                        
                        for rest in itertools.combinations(rest_points, t - 1):
                            # Lost
                            ts_lost = tuple(sorted(rest + (p,)))
                            idx_lost = tsub_to_idx.get(ts_lost)
                            if idx_lost is not None:
                                lost_idxs.add(idx_lost)
                                
                            # Gained
                            ts_gained = tuple(sorted(rest + (q,)))
                            idx_gained = tsub_to_idx.get(ts_gained)
                            if idx_gained is not None:
                                gained_idxs.add(idx_gained)
                        
                        # Check validity:
                        # For any idx in lost_idxs, if cover_count[idx] == 1, 
                        # it means only B covers it. If we remove B's coverage, it's uncovered.
                        # Unless it is also in gained_idxs (which is impossible for p!=q).
                        valid = True
                        for idx in lost_idxs:
                            if cover_count[idx] == 1:
                                valid = False
                                break
                        
                        if valid:
                            # Apply swap
                            new_B = [x for x in B if x != p]
                            new_B.append(q)
                            new_B.sort()
                            
                            # Update data structures
                            # 1. Remove old B contributions
                            for idx in B_tsubs:
                                cover_count[idx] -= 1
                            
                            # 2. Add new B contributions
                            new_B_tsubs = get_tsub_indices(new_B)
                            for idx in new_B_tsubs:
                                cover_count[idx] += 1
                            
                            blocks[i] = new_B
                            block_tsubs[i] = new_B_tsubs
                            
                            # 3. Try to remove redundant blocks
                            # We can just call remove_redundant, but it rebuilds everything.
                            # Since we only changed one block, we can check if any block is now redundant.
                            # A block j is redundant if all its tsubs have count > 1.
                            # Note: The counts have been updated.
                            
                            changed = True
                            while changed:
                                changed = False
                                for j in range(len(blocks)):
                                    if not any(cover_count[idx] == 1 for idx in block_tsubs[j]):
                                        # Remove block j
                                        for idx in block_tsubs[j]:
                                            cover_count[idx] -= 1
                                        blocks.pop(j)
                                        block_tsubs.pop(j)
                                        changed = True
                                        break
                            
                            improved = True
                            break # Restart outer loop with new structure
                            
                            # If we don't break, we continue checking other swaps for the same block?
                            # No, the structure changed, so indices are invalid.
                            # We must restart.

        return blocks

    for restart in range(100):
        if time.time() > deadline:
            break

        uncovered = set(range(n_tsubs))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            seed_idx = rng.choice(list(uncovered))
            seed = all_tsubs[seed_idx]
            
            block = list(seed)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    
                    noise = rng.random() * 0.5
                    total_score = gain + noise
                    
                    if total_score > best_gain:
                        best_gain = total_score
                        best_point = x
                
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(rng.choice(remaining))
                        block_set.add(block[-1])
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    uncovered.discard(idx)

        if time.time() > deadline or uncovered:
            continue

        blocks = local_search(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            if best_count <= 335:
                break

    if best_blocks is None:
        # Fallback
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            seed_idx = min(uncovered)
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(remaining[0])
                        block_set.add(block[-1])
                    else:
                        break
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(tuple(ts))
                if idx is not None:
                    uncovered.discard(idx)
        best_blocks = blocks

    return best_blocks
