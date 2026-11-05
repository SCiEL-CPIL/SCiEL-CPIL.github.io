import itertools
import random
import time
import heapq
import math

def construct(v, k, t):
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    def get_block_triple_indices(block):
        return [sub_to_idx[tr] for tr in itertools.combinations(block, t)]

    def greedy_pool_construct(seed, pool_size=5000):
        random.seed(seed)
        uncovered = set(range(n_subs))
        blocks = []
        pool = []
        
        # Helper to add a block to the pool
        def add_block_to_pool(b):
            # Only add if it's not a duplicate in the pool (approximate check or set)
            # For speed, we might just add it. Duplicates are handled by greedy selection logic
            # or we can keep a set of tuples.
            pool.append(tuple(sorted(b)))
            
        # Initial pool
        for _ in range(pool_size):
            b = random.sample(range(v), k)
            add_block_to_pool(b)
            
        # To speed up greedy, we can precompute the triple indices for pool blocks
        # But pool changes. Let's compute on the fly or cache.
        # Given v=24, k=7, C(7,3)=35 triples per block.
        
        while uncovered:
            if not pool:
                # Generate new pool, focusing on uncovered triples
                new_pool = []
                for _ in range(pool_size // 2):
                    if uncovered:
                        # Pick a random uncovered triple
                        tr_idx = random.choice(list(uncovered))
                        tr = all_subs[tr_idx]
                        # Build a block around this triple
                        b = list(tr)
                        remaining = random.sample([x for x in range(v) if x not in b], k - t)
                        b.extend(remaining)
                        new_pool.append(tuple(sorted(b)))
                pool = new_pool
            
            best_block = None
            best_gain = -1
            best_pool_idx = -1
            
            # Find the best block in the pool
            # Optimization: If pool is large, this is O(Pool * 35). 
            # 5000 * 35 = 175k ops per iteration. 
            # Number of iterations ~ 76. Total ~ 13M ops. Acceptable.
            
            for idx, b in enumerate(pool):
                gain = 0
                for tr_idx in get_block_triple_indices(b):
                    if tr_idx in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early termination if we find a block that covers everything?
                            # Unlikely.
                            pass
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    best_pool_idx = idx
            
            if best_block is None or best_gain == 0:
                # Stuck. Force add a block for a random uncovered triple
                tr_idx = random.choice(list(uncovered))
                tr = all_subs[tr_idx]
                b = list(tr)
                remaining = random.sample([x for x in range(v) if x not in b], k - t)
                b.extend(remaining)
                best_block = tuple(sorted(b))
                best_pool_idx = -1 # Not in pool
                best_gain = 0
            
            # Add to blocks
            blocks.append(list(best_block))
            
            # Update uncovered
            for tr_idx in get_block_triple_indices(best_block):
                uncovered.discard(tr_idx)
            
            # Remove from pool
            if best_pool_idx != -1:
                pool.pop(best_pool_idx)
                
        return blocks

    def local_search(initial_blocks, time_limit):
        current_blocks = [tuple(b) for b in initial_blocks]
        current_count = len(current_blocks)
        
        # Maintain coverage counts
        cover_counts = [0] * n_subs
        for b in current_blocks:
            for tr_idx in get_block_triple_indices(b):
                cover_counts[tr_idx] += 1
        
        # Verify validity
        if any(c == 0 for c in cover_counts):
            return None, float('inf')
            
        best_blocks = current_blocks[:]
        best_count = current_count
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            
            # --- Move 1: Try to Remove a Redundant Block ---
            removed = False
            for i in range(len(current_blocks)):
                b = current_blocks[i]
                redundant = True
                for tr_idx in get_block_triple_indices(b):
                    if cover_counts[tr_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for tr_idx in get_block_triple_indices(b):
                        cover_counts[tr_idx] -= 1
                    current_blocks.pop(i)
                    current_count -= 1
                    removed = True
                    break
                    
            if removed:
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = current_blocks[:]
                continue
            
            # --- Move 2: Try a Swap ---
            if not current_blocks:
                break
                
            i = random.randint(0, len(current_blocks) - 1)
            b_old = current_blocks[i]
            
            critical_triples = []
            for tr_idx in get_block_triple_indices(b_old):
                if cover_counts[tr_idx] == 1:
                    critical_triples.append(tr_idx)
            
            if not critical_triples:
                # Redundant, should have been caught by Move 1? 
                # Maybe cover_counts changed? No, we just checked.
                # If critical_triples is empty, it IS redundant.
                for tr_idx in get_block_triple_indices(b_old):
                    cover_counts[tr_idx] -= 1
                current_blocks.pop(i)
                current_count -= 1
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = current_blocks[:]
                continue
            
            # Union of points in critical triples
            U = set()
            for tr_idx in critical_triples:
                U.update(all_subs[tr_idx])
                
            if len(U) > k:
                continue
            
            available_points = [p for p in range(v) if p not in U]
            remaining_slots = k - len(U)
            
            # Try to find a better replacement
            # Heuristic: Maximize overlap with "over-covered" triples?
            # Or just random.
            # Let's try a few random completions and pick the one that maximizes
            # the number of triples with count > 1 in the new block (increasing redundancy).
            
            best_new_block = None
            best_score = -1
            
            for _ in range(10):
                if remaining_slots > len(available_points):
                    break
                extra_points = random.sample(available_points, remaining_slots)
                b_new = tuple(sorted(list(U) + extra_points))
                
                # Calculate score: how many of b_new's triples are already covered by others?
                # i.e., if we remove b_old, does b_new cover triples that are "safe"?
                # Actually, we want b_new to cover triples that have high counts, 
                # so that removing b_old (which covers criticals) doesn't break them,
                # and b_new adds redundancy to others.
                
                # Simulate:
                # New counts = old counts - b_old + b_new
                # We care about:
                # 1. All criticals in b_old are covered by b_new (guaranteed by U)
                # 2. Minimize new criticals? Or maximize redundancy?
                
                # Let's just use random for now to keep it simple and fast, 
                # relying on the removal step to find gains.
                best_new_block = b_new
                break # Just take the first one for speed in this iteration
            
            if best_new_block:
                # Perform swap
                for tr_idx in get_block_triple_indices(b_old):
                    cover_counts[tr_idx] -= 1
                for tr_idx in get_block_triple_indices(best_new_block):
                    cover_counts[tr_idx] += 1
                current_blocks[i] = best_new_block

        return best_blocks, best_count

    start_time = time.time()
    total_time_limit = 110
    
    best_blocks = None
    best_count = float('inf')
    
    seed = 42
    while time.time() - start_time < total_time_limit:
        blocks = greedy_pool_construct(seed)
        
        ls_blocks, ls_count = local_search(blocks, time_limit=total_time_limit - (time.time() - start_time))
        
        if ls_blocks is not None and ls_count < best_count:
            best_count = ls_count
            best_blocks = ls_blocks[:]
            
        seed += 1
        
        if best_count <= 70:
            break
            
    if best_blocks is None:
        blocks = greedy_pool_construct(seed)
        return blocks
        
    return best_blocks
