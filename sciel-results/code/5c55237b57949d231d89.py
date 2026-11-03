import itertools
import heapq
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_id_map = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(k_subsets)
    
    block_covers = [[] for _ in range(num_k)]
    blocks_covering_t = [[] for _ in range(num_t)]
    
    for b, block in enumerate(k_subsets):
        for ts in itertools.combinations(block, t):
            tid = t_id_map[ts]
            block_covers[b].append(tid)
            blocks_covering_t[tid].append(b)
            
    # 2. Greedy
    def run_greedy(randomize_ties=False):
        current_counts = [len(covers) for covers in block_covers]
        is_t_covered = [False] * num_t
        uncovered_t_count = num_t
        
        heap = []
        for b in range(num_k):
            if current_counts[b] > 0:
                salt = random.randint(0, 10**9) if randomize_ties else 0
                heapq.heappush(heap, (-current_counts[b], salt, b))
                
        selected_blocks = []
        
        while uncovered_t_count > 0:
            if time.time() > deadline:
                return None
                
            valid_block = None
            while heap:
                neg_cnt, salt, b = heapq.heappop(heap)
                if -neg_cnt == current_counts[b]:
                    valid_block = b
                    break
            if valid_block is None:
                break
                
            selected_blocks.append(valid_block)
            
            for tid in block_covers[valid_block]:
                if not is_t_covered[tid]:
                    is_t_covered[tid] = True
                    uncovered_t_count -= 1
                    for b in blocks_covering_t[tid]:
                        current_counts[b] -= 1
                        if current_counts[b] > 0:
                            salt = random.randint(0, 10**9) if randomize_ties else 0
                            heapq.heappush(heap, (-current_counts[b], salt, b))
                            
        return selected_blocks

    best_blocks = run_greedy(randomize_ties=False)
    
    if best_blocks and time.time() < deadline - 10.0:
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            current_blocks = run_greedy(randomize_ties=True)
            if current_blocks and len(current_blocks) < len(best_blocks):
                best_blocks = current_blocks
                
    # 3. Local Search: Remove redundant blocks
    if best_blocks and time.time() < deadline - 5.0:
        # Convert list of block IDs to set for O(1) lookup
        block_set = set(best_blocks)
        
        # Calculate initial cover counts
        cover_count = [0] * num_t
        for b in best_blocks:
            for tid in block_covers[b]:
                cover_count[tid] += 1
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Iterate over a copy to allow modification
            for b in list(best_blocks):
                if time.time() > deadline - 2.0:
                    break
                
                # Try removing b
                newly_uncovered = []
                for tid in block_covers[b]:
                    cover_count[tid] -= 1
                    if cover_count[tid] == 0:
                        newly_uncovered.append(tid)
                    
                if not newly_uncovered:
                    # Safe to remove
                    best_blocks.remove(b)
                    block_set.discard(b)
                    improved = True
                else:
                    # Need to find a replacement
                    # Find block b' not in block_set that covers all newly_uncovered
                    # To be efficient, pick the one covering the most of them
                    best_b_prime = -1
                    max_covered = 0
                    
                    # Optimization: Only check blocks that cover at least one of the newly_uncovered
                    candidate_blocks = set()
                    for tid in newly_uncovered:
                        candidate_blocks.update(blocks_covering_t[tid])
                    
                    for b_prime in candidate_blocks:
                        if b_prime in block_set:
                            continue
                        # Count how many of newly_uncovered it covers
                        cnt = 0
                        for tid in newly_uncovered:
                            if tid in block_covers[b_prime]: # This is slow
                                cnt += 1
                                if cnt == len(newly_uncovered):
                                    break
                        if cnt > max_covered:
                            max_covered = cnt
                            best_b_prime = b_prime
                            
                    if max_covered == len(newly_uncovered):
                        # Swap
                        best_blocks.remove(b)
                        block_set.discard(b)
                        best_blocks.append(best_b_prime)
                        block_set.add(best_b_prime)
                        
                        # Update cover counts for the swap
                        # We already decremented for b.
                        # Now increment for b_prime
                        for tid in block_covers[best_b_prime]:
                            cover_count[tid] += 1
                        improved = True
                    else:
                        # Revert removal of b
                        for tid in newly_uncovered:
                            cover_count[tid] += 1
                            
    # Convert IDs back to lists
    return [k_subsets[b] for b in best_blocks]
