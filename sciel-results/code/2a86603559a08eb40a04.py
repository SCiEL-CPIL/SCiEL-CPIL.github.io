import itertools
import random
import time
import sys
import heapq
from collections import defaultdict

_cache = {}

def _precompute(v, k, t):
    if (v, k, t) in _cache:
        return _cache[(v, k, t)]
    
    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    t_sub_map = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    all_k_blocks = list(itertools.combinations(points, k))
    n_k = len(all_k_blocks)
    
    # block_to_subs: for each block, list of indices of t-subsets it covers
    block_to_subs = []
    # sub_to_blocks: for each t-subset, list of indices of blocks that cover it
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_k_blocks):
        subs_in_block = []
        for t_sub in itertools.combinations(block, t):
            s_idx = t_sub_map[t_sub]
            subs_in_block.append(s_idx)
            sub_to_blocks[s_idx].append(b_idx)
        block_to_subs.append(subs_in_block)
        
    _cache[(v, k, t)] = (all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k)
    return _cache[(v, k, t)]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 35s margin for safety
    
    all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    initial_gain = len(block_to_subs[0]) if n_k > 0 else 0
    
    def run_greedy(seed):
        """Run greedy with heap-based gain tracking and randomized tie-breaking."""
        random.seed(seed)
        
        # Initialize
        uncovered = bytearray([1] * n_t)  # 1 if uncovered
        remaining_uncovered = n_t
        
        # Gain tracking with lazy deletion
        # heap entries: (-gain, block_idx, version)
        gain = [initial_gain] * n_k
        version = [0] * n_k
        heap = [(-initial_gain, b, 0) for b in range(n_k)]
        heapq.heapify(heap)
        
        selected = []
        
        while remaining_uncovered > 0:
            # Pop until we find a valid max-gain block
            while heap:
                neg_g, b, ver = heap[0]
                if ver == version[b] and gain[b] == -neg_g:
                    break
                heapq.heappop(heap)
            
            if not heap:
                break
            
            neg_g, b_sel, ver = heapq.heappop(heap)
            max_gain = -neg_g
            
            if max_gain == 0:
                break
            
            # Collect all blocks with the same max gain for tie-breaking
            max_blocks = [b_sel]
            # Peek at other entries with same gain
            temp_heap = []
            while heap:
                ng, b, v2 = heap[0]
                if v2 == version[b] and gain[b] == max_gain:
                    heapq.heappop(heap)
                    max_blocks.append(b)
                    temp_heap.append((ng, b, v2))
                else:
                    break
            # Put back the non-max entries
            for item in temp_heap:
                heapq.heappush(heap, item)
            
            # Random tie-breaking
            if len(max_blocks) > 1:
                b_sel = random.choice(max_blocks)
            
            selected.append(all_k_blocks[b_sel])
            
            # Update uncovered and gains
            for s in block_to_subs[b_sel]:
                if uncovered[s]:
                    uncovered[s] = 0
                    remaining_uncovered -= 1
                    for b_prime in sub_to_blocks[s]:
                        if gain[b_prime] > 0:
                            gain[b_prime] -= 1
                            version[b_prime] += 1
                            heapq.heappush(heap, (-gain[b_prime], b_prime, version[b_prime]))
        
        return selected
    
    def run_dual_greedy(seed):
        """Dual-greedy: minimize maximum redundancy created by adding a block.
        
        Instead of maximizing new coverage, we prioritize blocks that create
        the least overlap with existing coverage, building a structurally dense
        core set. We track for each t-subset how many times it's covered.
        When adding a block, we prefer one where the maximum cover_count among
        its t-subsets is minimized (least redundancy).
        """
        random.seed(seed)
        
        # cover_count[s] = number of blocks covering t-subset s
        cover_count = [0] * n_t
        remaining_uncovered = n_t
        
        # For each block, track:
        # - max_redundancy: max cover_count among its t-subsets (before adding)
        # - unique_gain: number of t-subsets with cover_count == 0
        # We want to minimize max_redundancy, then maximize unique_gain
        
        # Precompute for each block the list of t-subsets it covers
        # block_to_subs already exists
        
        # Use a priority queue: (max_redundancy, -unique_gain, block_idx, version)
        # Lower max_redundancy is better, higher unique_gain is better
        
        max_red = [0] * n_k  # max cover_count in block's t-subsets
        uniq_gain = [initial_gain] * n_k  # count of t-subsets with cover_count==0
        version = [0] * n_k
        
        heap = []
        for b in range(n_k):
            # Compute initial max_red and uniq_gain
            subs = block_to_subs[b]
            mr = 0
            ug = 0
            for s in subs:
                if cover_count[s] > mr:
                    mr = cover_count[s]
                if cover_count[s] == 0:
                    ug += 1
            max_red[b] = mr
            uniq_gain[b] = ug
            heapq.heappush(heap, (mr, -ug, b, 0))
        
        selected = []
        
        while remaining_uncovered > 0:
            # Pop until we find a valid entry
            while heap:
                mr, neg_ug, b, ver = heap[0]
                if ver == version[b] and max_red[b] == mr and uniq_gain[b] == -neg_ug:
                    break
                heapq.heappop(heap)
            
            if not heap:
                break
            
            mr, neg_ug, b_sel, ver = heapq.heappop(heap)
            
            # Collect all blocks with same (max_red, uniq_gain) for tie-breaking
            max_red_val = mr
            uniq_gain_val = -neg_ug
            candidates = [b_sel]
            
            temp_heap = []
            while heap:
                mr2, neg_ug2, b2, ver2 = heap[0]
                if ver2 == version[b2] and max_red[b2] == max_red_val and uniq_gain[b2] == uniq_gain_val:
                    heapq.heappop(heap)
                    candidates.append(b2)
                    temp_heap.append((mr2, neg_ug2, b2, ver2))
                else:
                    break
            
            for item in temp_heap:
                heapq.heappush(heap, item)
            
            # Random tie-breaking
            if len(candidates) > 1:
                b_sel = random.choice(candidates)
            
            selected.append(all_k_blocks[b_sel])
            
            # Update cover counts and recompute metrics for affected blocks
            for s in block_to_subs[b_sel]:
                old_count = cover_count[s]
                cover_count[s] = old_count + 1
                if old_count == 0:
                    remaining_uncovered -= 1
                
                # Update all blocks that cover this t-subset
                for b_prime in sub_to_blocks[s]:
                    if b_prime == b_sel:
                        continue
                    # Recompute max_red and uniq_gain for b_prime
                    # This is expensive; do it lazily only when needed
                    # For now, mark version change and recompute on pop
                    # Actually, let's recompute properly
                    subs_p = block_to_subs[b_prime]
                    new_mr = 0
                    new_ug = 0
                    for sp in subs_p:
                        cc = cover_count[sp]
                        if cc > new_mr:
                            new_mr = cc
                        if cc == 0:
                            new_ug += 1
                    
                    if new_mr != max_red[b_prime] or new_ug != uniq_gain[b_prime]:
                        max_red[b_prime] = new_mr
                        uniq_gain[b_prime] = new_ug
                        version[b_prime] += 1
                        heapq.heappush(heap, (new_mr, -new_ug, b_prime, version[b_prime]))
        
        return selected
    
    def local_search(blocks, seed, max_iters=500):
        """Try to reduce block count by swapping or removing redundant blocks."""
        random.seed(seed)
        blocks = list(blocks)
        n = len(blocks)
        
        if n == 0:
            return blocks
        
        # Build coverage count for each t-subset
        cover_count = [0] * n_t
        for block in blocks:
            b_idx = all_k_blocks.index(block) if block in all_k_blocks else -1
            if b_idx >= 0:
                for s in block_to_subs[b_idx]:
                    cover_count[s] += 1
        
        # Check if any block is redundant (all its t-subsets are covered by other blocks)
        def is_redundant(block_idx_in_list):
            block = blocks[block_idx_in_list]
            b_idx = all_k_blocks.index(block)
            if b_idx < 0:
                return False
            for s in block_to_subs[b_idx]:
                if cover_count[s] <= 1:
                    return False
            return True
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try removing redundant blocks
            new_blocks = []
            for i, block in enumerate(blocks):
                if is_redundant(i):
                    # Remove this block
                    b_idx = all_k_blocks.index(block)
                    for s in block_to_subs[b_idx]:
                        cover_count[s] -= 1
                    improved = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
            if not blocks:
                break
            
            # Try swapping: replace one block with another that covers more uncovered
            # This is expensive, so do it sparingly
            if random.random() < 0.3 and len(blocks) > 1:
                i = random.randint(0, len(blocks) - 1)
                old_block = blocks[i]
                old_b_idx = all_k_blocks.index(old_block)
                
                # Find blocks that could replace it
                # A good replacement covers at least the same t-subsets that are only covered by old_block
                only_covered_by_old = []
                for s in block_to_subs[old_b_idx]:
                    if cover_count[s] == 1:
                        only_covered_by_old.append(s)
                
                if not only_covered_by_old:
                    # Old block is redundant, already handled
                    continue
                
                # Try a few random candidates
                for _ in range(20):
                    cand_idx = random.randint(0, n_k - 1)
                    cand_block = all_k_blocks[cand_idx]
                    if cand_block == old_block:
                        continue
                    
                    # Check if candidate covers all only_covered_by_old
                    cand_subs_set = set(block_to_subs[cand_idx])
                    if all(s in cand_subs_set for s in only_covered_by_old):
                        # Try the swap
                        # Remove old, add candidate
                        for s in block_to_subs[old_b_idx]:
                            cover_count[s] -= 1
                        for s in block_to_subs[cand_idx]:
                            cover_count[s] += 1
                        
                        # Check validity: all cover_count >= 1
                        valid = True
                        for s in range(n_t):
                            if cover_count[s] < 1:
                                valid = False
                                break
                        
                        if valid:
                            blocks[i] = cand_block
                            improved = True
                            break
                        else:
                            # Revert
                            for s in block_to_subs[cand_idx]:
                                cover_count[s] -= 1
                            for s in block_to_subs[old_b_idx]:
                                cover_count[s] += 1
        
        return blocks
    
    # Phase 1: Run multiple greedy attempts (mix of standard and dual-greedy)
    attempts = 0
    while time.time() < deadline - 20.0:  # Leave 20s for local search
        attempts += 1
        # Alternate between standard greedy and dual-greedy
        if attempts % 2 == 1:
            blocks = run_greedy(attempts)
        else:
            blocks = run_dual_greedy(attempts)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # Early stop if we hit theoretical lower bound
        # Lower bound: ceil(C(19,5) / C(8,5)) = ceil(11628/56) = 208
        if best_count <= 208:
            break
        
        # If we've done many attempts and not improving, slow down
        if attempts > 50 and attempts % 10 == 0:
            # Check if we're making progress
            pass
    
    # Phase 2: Local search on best solution
    if best_blocks is not None and time.time() < deadline - 5.0:
        # Run local search with multiple seeds
        for ls_seed in range(10):
            if time.time() > deadline - 5.0:
                break
            ls_blocks = local_search(best_blocks, ls_seed, max_iters=200)
            if len(ls_blocks) < best_count:
                best_count = len(ls_blocks)
                best_blocks = ls_blocks
        
        # Also try local search on second-best greedy result if available
        # (We didn't store it, so skip for simplicity)
        
        # Phase 2.5: Additional dual-greedy attempts with local search
        if time.time() < deadline - 10.0:
            for dg_seed in range(5):
                if time.time() > deadline - 10.0:
                    break
                dg_blocks = run_dual_greedy(1000 + dg_seed)
                if len(dg_blocks) < best_count:
                    best_count = len(dg_blocks)
                    best_blocks = dg_blocks
                    # Run local search on this new best
                    if time.time() < deadline - 5.0:
                        ls_blocks = local_search(best_blocks, 2000 + dg_seed, max_iters=100)
                        if len(ls_blocks) < best_count:
                            best_count = len(ls_blocks)
                            best_blocks = ls_blocks
    
    if best_blocks is None:
        # Fallback: should not happen, but return something
        return []
        
    return best_blocks
