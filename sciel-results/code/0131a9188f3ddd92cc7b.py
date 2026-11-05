import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    def get_block_triplets(block):
        """Returns list of triplet IDs covered by this block."""
        tids = []
        for triplet in itertools.combinations(block, t):
            tids.append(triplet_to_id[triplet])
        return tids

    def generate_greedy_block(rng, seed_triplet=None):
        """Generate a block by greedily adding points to cover triplets."""
        if seed_triplet is None:
            seed_triplet = tuple(sorted(rng.sample(range(v), t)))
        
        block = list(seed_triplet)
        block_set = set(block)
        
        # Heuristic: Add points that cover the most "new" triplets relative to current block
        # We don't have global coverage info here, so we just maximize internal density?
        # Or just random completion?
        # Let's just complete randomly but ensure distinctness.
        # Actually, a better "structured" block is to pick a random point and ensure 
        # it forms many triplets with existing points.
        
        # Simple random completion
        while len(block) < k:
            candidates = [p for p in range(v) if p not in block_set]
            p = rng.choice(candidates)
            block.append(p)
            block_set.add(p)
            
        return tuple(sorted(block))

    def generate_candidate_pool(seed, num_blocks):
        """Generate a pool of blocks. Mix of random and greedy-seeded."""
        rng = random.Random(seed)
        candidates = []
        seen_blocks = set()
        
        # 50% random, 50% greedy-seeded (though greedy-seeded here is just random completion, 
        # so it's effectively random. Let's make the greedy-seeded ones actually try to be dense?)
        # For now, let's just use random. The global greedy will pick the best ones.
        
        while len(candidates) < num_blocks:
            block = tuple(sorted(rng.sample(range(v), k)))
            if block in seen_blocks:
                continue
            seen_blocks.add(block)
            tids = get_block_triplets(block)
            candidates.append((block, tids))
            
        return candidates

    def greedy_set_cover(candidates, deadline):
        """
        Global greedy set cover using a max-heap with lazy deletion.
        candidates: list of (block_tuple, triplet_id_list)
        """
        covered_count = np.zeros(n_triplets, dtype=np.int32)
        selected_blocks = []
        
        # Initialize heap with (initial_gain, index)
        heap = []
        for i, (block, tids) in enumerate(candidates):
            gain = len(tids)
            heapq.heappush(heap, (-gain, i))
            
        selected = [False] * len(candidates)
        
        while covered_count.sum() < n_triplets:
            if time.time() > deadline - 2:
                break
                
            valid_block_idx = -1
            valid_gain = -1
            
            while heap:
                neg_gain, idx = heapq.heappop(heap)
                if selected[idx]:
                    continue
                
                block, tids = candidates[idx]
                current_gain = 0
                for tid in tids:
                    if covered_count[tid] == 0:
                        current_gain += 1
                
                if current_gain == -neg_gain:
                    valid_block_idx = idx
                    valid_gain = current_gain
                    break
                else:
                    if current_gain > 0:
                        heapq.heappush(heap, (-current_gain, idx))
            
            if valid_block_idx == -1:
                break
                
            selected[valid_block_idx] = True
            selected_blocks.append(candidates[valid_block_idx][0])
            
            block, tids = candidates[valid_block_idx]
            for tid in tids:
                covered_count[tid] += 1
                
        return selected_blocks

    def polish(blocks, deadline):
        """Remove redundant blocks."""
        if not blocks:
            return blocks
            
        blocks = [tuple(sorted(b)) for b in blocks]
        block_triplets = [set(get_block_triplets(b)) for b in blocks]
        
        count = np.zeros(n_triplets, dtype=np.int32)
        for tids in block_triplets:
            for tid in tids:
                count[tid] += 1
                
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                redundant = True
                for tid in block_triplets[i]:
                    if count[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    changed = True
                    for tid in block_triplets[i]:
                        count[tid] -= 1
                        
        new_blocks = [blocks[i] for i in range(len(blocks)) if active[i]]
        return [list(b) for b in new_blocks]

    best_blocks = None
    best_count = float('inf')
    
    # Run global greedy with different candidate pools
    num_pools = 30 
    pool_size = 5000
    
    for pool_idx in range(num_pools):
        if time.time() > deadline - 10:
            break
            
        seed = pool_idx * 1000 + 42
        candidates = generate_candidate_pool(seed, pool_size)
        
        blocks = greedy_set_cover(candidates, deadline - 5)
        blocks = polish(blocks, deadline - 2)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = [list(range(k))] # Fallback invalid, but ensures return

    return best_blocks

if __name__ == "__main__":
    pass
