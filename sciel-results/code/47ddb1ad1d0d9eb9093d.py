"""Covering design C(24,7,3) - optimized greedy with local search and time budget."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 138.0  # safety margin below 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def get_block_covered(block):
        """Get the set of t-subset indices covered by a block."""
        return frozenset(t_subset_to_idx[ts] for ts in itertools.combinations(block, t))
    
    def greedy_cover(seed, order_mode="random"):
        """Greedy covering with a given random seed and ordering mode."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        if not uncovered:
            return blocks
        
        # Choose initial block from an uncovered t-subset
        if order_mode == "random":
            start_idx = rng.choice(list(uncovered))
        else:
            start_idx = min(uncovered)
        
        block = list(all_t_subsets[start_idx])
        
        while len(block) < k:
            # Find the point that covers the most uncovered t-subsets
            best_x = None
            best_gain = -1
            
            candidates = [x for x in range(v) if x not in block]
            if order_mode == "random":
                rng.shuffle(candidates)
            
            for x in candidates:
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if t_subset_to_idx.get(ts) in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for ts in itertools.combinations(block, t):
            uncovered.discard(t_subset_to_idx[ts])
        
        # Check if we're done
        if not uncovered:
            return blocks
        
        # Continue with remaining uncovered
        while uncovered:
            if time.time() > deadline - 5.0:
                return blocks
            
            if order_mode == "random":
                start_idx = rng.choice(list(uncovered))
            else:
                start_idx = min(uncovered)
            
            block = list(all_t_subsets[start_idx])
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                candidates = [x for x in range(v) if x not in block]
                if order_mode == "random":
                    rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                uncovered.discard(t_subset_to_idx[ts])
        
        return blocks
    
    def local_search_optimize(blocks, seed):
        """
        Local search using block removal and point swaps to reduce block count.
        """
        rng = random.Random(seed)
        blocks = list(blocks)
        n = len(blocks)
        
        # Precompute coverage for each block
        block_covers = [get_block_covered(b) for b in blocks]
        
        # Count how many times each t-subset is covered
        cover_count = np.zeros(n_t, dtype=np.int32)
        for bc in block_covers:
            for idx in bc:
                cover_count[idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Pass 1: Remove redundant blocks
            removed_indices = set()
            indices = list(range(n))
            rng.shuffle(indices)
            
            for i in indices:
                if i in removed_indices:
                    continue
                if time.time() > deadline - 5.0:
                    break
                
                # Check if removing block i keeps all t-subsets covered
                can_remove = True
                for idx in block_covers[i]:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove block i
                    for idx in block_covers[i]:
                        cover_count[idx] -= 1
                    removed_indices.add(i)
                    improved = True
            
            if removed_indices:
                # Rebuild lists
                new_blocks = []
                new_block_covers = []
                for i in range(n):
                    if i not in removed_indices:
                        new_blocks.append(blocks[i])
                        new_block_covers.append(block_covers[i])
                blocks = new_blocks
                block_covers = new_block_covers
                n = len(blocks)
                continue # Restart loop with new structure
            
            # Pass 2: Try to swap points in blocks to create redundancies
            # We want to find a swap that is "safe" (doesn't break coverage of unique t-subsets)
            # and ideally helps cover t-subsets that are currently unique to *other* blocks,
            # making them removable in the next pass.
            
            # To make this efficient, we focus on blocks that are "critical" (contain unique t-subsets)
            # or just try random swaps on random blocks.
            
            # Identify critical t-subsets (count == 1)
            # A block is "locked" if it covers any t-subset with count 1.
            # We can only swap points in a block if the t-subsets it loses are covered by others.
            
            # Try a few random swaps
            num_swap_attempts = 50
            for _ in range(num_swap_attempts):
                if time.time() > deadline - 5.0:
                    break
                
                if n == 0:
                    break
                    
                i = rng.randint(0, n - 1)
                block_i = blocks[i]
                
                # Pick a point to remove
                x_idx = rng.randint(0, k - 1)
                x = block_i[x_idx]
                
                # Pick a point to add
                # We want to pick a point y not in block_i
                # Ideally y should help cover some t-subsets.
                # Let's just pick randomly first.
                candidates = [y for y in range(v) if y not in block_i]
                if not candidates:
                    continue
                y = rng.choice(candidates)
                
                # Simulate swap: replace x with y in block_i
                # New block: block_i' = (block_i - {x}) U {y}
                # t-subsets lost: those containing x and 2 other points from (block_i - {x})
                # t-subsets gained: those containing y and 2 other points from (block_i - {x})
                
                rest = [p for p in block_i if p != x]
                
                # Check safety: All t-subsets lost must be covered by count > 1 (so count >= 2 originally, 
                # but since we are removing x from this block, if count was 1, it becomes 0 -> invalid)
                # So, for every t-subset ts containing x and 2 from rest, cover_count[ts] must be > 1.
                
                safe = True
                lost_ts_indices = []
                gained_ts_indices = []
                
                # Precompute combinations of rest of size t-1
                # Since t=3, rest has size 6. We need pairs from rest.
                # Wait, t=3. Block size k=7.
                # If we remove x, we lose t-subsets {x, a, b} where a,b in rest.
                # If we add y, we gain t-subsets {y, a, b} where a,b in rest.
                
                for a, b in itertools.combinations(rest, t - 1):
                    # Lost t-subset
                    ts_lost = tuple(sorted((x, a, b)))
                    idx_lost = t_subset_to_idx.get(ts_lost)
                    if idx_lost is not None:
                        if cover_count[idx_lost] == 1:
                            safe = False
                            break
                        lost_ts_indices.append(idx_lost)
                    
                    # Gained t-subset
                    ts_gained = tuple(sorted((y, a, b)))
                    idx_gained = t_subset_to_idx.get(ts_gained)
                    if idx_gained is not None:
                        gained_ts_indices.append(idx_gained)
                
                if not safe:
                    continue
                
                # The swap is safe. Apply it?
                # It might not immediately reduce the block count, but it might change the structure
                # such that other blocks become redundant.
                # However, we should only accept the swap if it doesn't increase the number of "critical" blocks
                # or if it leads to a removal.
                # For simplicity, let's accept the swap if it is safe, and see if it helps.
                # But we need to be careful not to loop indefinitely.
                # Let's only accept if it reduces the number of "unique" t-subsets or keeps it same?
                # Actually, a safe swap preserves validity. 
                # Let's just do it and let the removal pass clean up.
                
                # Apply swap
                new_block_i = list(block_i)
                new_block_i[x_idx] = y
                new_block_i.sort()
                
                # Update data structures
                # Subtract old coverage
                for idx in lost_ts_indices:
                    cover_count[idx] -= 1
                # Add new coverage
                # Note: gained_ts_indices might contain duplicates if the same t-subset is generated multiple times?
                # No, combinations are unique. But (y, a, b) is unique for each pair (a,b).
                # However, if y was already in the block? No, we picked y not in block.
                # So the new t-subsets are distinct from the old ones (since one has x, one has y).
                # But wait, could a "gained" t-subset be the same as a "lost" one?
                # Only if x == y, which is not possible.
                # So we just increment.
                
                # We need to be careful: The set of t-subsets covered by the block changes.
                # Old set: S_old
                # New set: S_new
                # S_old \ S_new = Lost
                # S_new \ S_old = Gained
                
                for idx in gained_ts_indices:
                    cover_count[idx] += 1
                
                # Update block list
                blocks[i] = new_block_i
                block_covers[i] = get_block_covered(new_block_i)
                
                improved = True
                # Break out of swap loop to allow removal pass to run
                break
        
        return blocks
    
    # Strategy 1: Standard greedy (deterministic)
    best_blocks = None
    best_count = float('inf')
    
    # First, try the original greedy approach
    blocks = greedy_cover(0, "deterministic")
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = blocks
    
    # Try multiple random seeds with local search
    num_attempts = 30
    for i in range(num_attempts):
        if time.time() > deadline - 10.0:
            break
        blocks = greedy_cover(i + 1, "random")
        
        # Apply local search to try to remove redundant blocks
        blocks = local_search_optimize(blocks, i + 100)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Try a more sophisticated approach: greedy with lookahead
    def greedy_with_lookahead(seed):
        """Greedy with a small lookahead to choose better starting points."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        if not uncovered:
            return blocks
        
        while uncovered:
            if time.time() > deadline - 5.0:
                return blocks
            
            # Try a few candidate starting t-subsets and pick the best
            candidates = list(uncovered)
            rng.shuffle(candidates)
            num_to_try = min(5, len(candidates))
            
            best_start = None
            best_score = float('inf')
            
            for start_idx in candidates[:num_to_try]:
                # Simulate how many blocks this starting point would lead to
                # (approximate by checking how many other uncovered t-subsets share points)
                start_ts = all_t_subsets[start_idx]
                shared = 0
                for other_idx in list(uncovered)[:50]:
                    if set(start_ts) & set(all_t_subsets[other_idx]):
                        shared += 1
                # Prefer starts that share many points with other uncovered sets
                score = -shared
                if score < best_score:
                    best_score = score
                    best_start = start_idx
            
            if best_start is None:
                best_start = candidates[0]
            
            block = list(all_t_subsets[best_start])
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                uncovered.discard(t_subset_to_idx[ts])
        
        return blocks
    
    # Try lookahead greedy with local search
    for i in range(15):
        if time.time() > deadline - 10.0:
            break
        blocks = greedy_with_lookahead(i)
        blocks = local_search_optimize(blocks, i + 200)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Try a different strategy: start with a large random block and greedily add
    def random_greedy(seed):
        """Start with random blocks and greedily fill them."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        if not uncovered:
            return blocks
        
        # First, create a few random blocks
        num_initial = 3
        for _ in range(num_initial):
            if not uncovered:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random t-subset from uncovered as seed
            start_idx = rng.choice(list(uncovered))
            block = list(all_t_subsets[start_idx])
            
            # Greedily add points
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(t_subset_to_idx[ts])
        
        # Continue with standard greedy for remaining
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            
            start_idx = min(uncovered)
            block = list(all_t_subsets[start_idx])
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                uncovered.discard(t_subset_to_idx[ts])
        
        return blocks
    
    for i in range(10):
        if time.time() > deadline - 10.0:
            break
        blocks = random_greedy(i)
        blocks = local_search_optimize(blocks, i + 300)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Final check: ensure we have a valid covering
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    # Verify the covering is valid
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
    
    if covered != set(range(n_t)):
        # If invalid, fall back to a simple greedy that's guaranteed to work
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            start_idx = min(uncovered)
            block = list(all_t_subsets[start_idx])
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(t_subset_to_idx[ts])
        return blocks
    
    return best_blocks
