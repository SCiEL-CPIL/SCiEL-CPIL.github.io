import time
import itertools
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsubsets_set = set(all_tsubsets)
    
    # Map t-subset to index for faster lookup? 
    # For v=24, t=3, 2024 subsets. Tuple lookup is fine.
    
    def _greedy_construct(rng):
        """
        Greedy construction. Returns list of blocks.
        """
        uncovered = set(all_tsubsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            # Pick a random uncovered t-subset to target
            target = rng.choice(list(uncovered))
            block = list(target)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                if time.time() > deadline:
                    break
                best_x = -1
                best_gain = -1
                # Candidate points
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    # New t-subsets formed by adding x to (t-1)-subsets of block
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        return blocks

    def _remove_redundant(blocks, deadline):
        """
        Iteratively remove blocks that are redundant.
        Returns list of blocks and a structure for coverage counts.
        """
        # Initialize coverage counts
        cover_count = defaultdict(int)
        for b in blocks:
            for sub in itertools.combinations(b, t):
                cover_count[sub] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                b = blocks[i]
                redundant = True
                for sub in itertools.combinations(b, t):
                    if cover_count[sub] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for sub in itertools.combinations(b, t):
                        cover_count[sub] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
                    
        return blocks, cover_count

    def _local_search(blocks, cover_count, deadline, rng):
        """
        Local search using swaps and kicks.
        """
        # We need a way to quickly check if a new block is "good".
        # A block is good if it covers triples that are currently "tight" (count=1) 
        # and doesn't disrupt the structure too much, OR if it allows removal of another block.
        
        # Simple strategy: 
        # 1. Try to remove any block.
        # 2. If cannot remove, try to swap a "bad" block with a "better" one.
        
        best_blocks = blocks[:]
        best_count = len(best_blocks)
        
        # Precompute all possible blocks? No, too many (346k).
        # Generate candidates on the fly.
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try removals
            for i in range(len(best_blocks)):
                if time.time() > deadline:
                    break
                b = best_blocks[i]
                redundant = True
                for sub in itertools.combinations(b, t):
                    if cover_count[sub] <= 1:
                        redundant = False
                        break
                if redundant:
                    for sub in itertools.combinations(b, t):
                        cover_count[sub] -= 1
                    best_blocks.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try swaps
            # Pick a random block to potentially replace
            if not best_blocks:
                break
                
            i = rng.randint(0, len(best_blocks) - 1)
            old_block = best_blocks[i]
            
            # Find "critical" triples in old_block (those with count 1)
            critical_subs = []
            for sub in itertools.combinations(old_block, t):
                if cover_count[sub] == 1:
                    critical_subs.append(sub)
                    
            if not critical_subs:
                # No critical subs, but we couldn't remove it? 
                # That means we already checked it in step 1 and it wasn't redundant?
                # Wait, if no critical subs, it IS redundant. 
                # So if we reach here, critical_subs must be non-empty 
                # OR the block was just added and not checked?
                # Actually, if critical_subs is empty, the block is redundant.
                # So this case shouldn't happen if step 1 worked correctly.
                # But maybe we missed it? Let's force removal.
                for sub in itertools.combinations(old_block, t):
                    cover_count[sub] -= 1
                best_blocks.pop(i)
                improved = True
                continue

            # We need to replace old_block with a new_block that covers all critical_subs.
            # And ideally, the new_block covers other subs that are also critical, 
            # or covers subs with count > 1 (harmless) or count 1 (helpful).
            
            # Generate candidate new blocks.
            # Heuristic: Start with the critical subs and expand to size k.
            # This is expensive. 
            # Alternative: Try random blocks that contain all critical subs?
            # Number of critical subs is small usually.
            
            # Let's try a few random candidates constructed from critical subs.
            found_swap = False
            for _ in range(10): # Try 10 candidates
                if time.time() > deadline:
                    break
                    
                # Build a candidate block containing all critical subs
                # This is a set packing/covering problem itself.
                # Simple approach: Take the union of points in critical subs, 
                # then add random points to reach size k.
                
                cand_points = set()
                for sub in critical_subs:
                    cand_points.update(sub)
                    
                if len(cand_points) > k:
                    continue # Cannot fit all critical subs in one block if they are disjoint? 
                             # Actually, if they are disjoint, we can't cover them all with one block.
                             # In that case, a single swap is not enough. We might need a "kick" (multi-swap).
                             # For now, skip if too large.
                             
                # Fill remaining spots randomly
                cand_block = list(cand_points)
                while len(cand_block) < k:
                    x = rng.randint(0, v - 1)
                    if x not in cand_block:
                        cand_block.append(x)
                
                cand_block.sort()
                
                if cand_block == old_block:
                    continue
                    
                # Check if this candidate is valid (covers all critical subs)
                valid = True
                for sub in critical_subs:
                    if not all(x in cand_block for x in sub):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Evaluate the swap
                # 1. Remove old_block
                for sub in itertools.combinations(old_block, t):
                    cover_count[sub] -= 1
                # 2. Add cand_block
                for sub in itertools.combinations(cand_block, t):
                    cover_count[sub] += 1
                    
                # 3. Check if any triple becomes uncovered (count 0)
                # We only need to check triples in old_block (which decreased) 
                # and triples in cand_block (which increased, so safe).
                # Triples in old_block that were count 1 are now count 0?
                # We ensured cand_block covers critical_subs (which were count 1 in old_block).
                # Are there other count 1 subs in old_block? 
                # Yes, critical_subs were *all* count 1 subs in old_block.
                # So if cand_block covers them, they are now count >= 1 (from cand_block).
                # What about subs in old_block that were count > 1? They become count-1 >= 1. Safe.
                # What about subs in cand_block? They increase. Safe.
                
                # So the swap is valid!
                best_blocks[i] = cand_block
                found_swap = True
                improved = True
                break
                
            if not found_swap and time.time() > deadline:
                break

        return best_blocks

    # Main logic
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10: # Leave 10s for final polish
            break
            
        rng = random.Random(100 + r)
        
        # Phase 1: Greedy
        blocks = _greedy_construct(rng)
        
        # Phase 2: Remove redundant
        blocks, cover_count = _remove_redundant(blocks, deadline - 5)
        
        # Phase 3: Local Search
        blocks = _local_search(blocks, cover_count, deadline - 2, rng)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
    # Final polish: try to remove more
    if best_blocks:
        # Rebuild cover count for the best blocks
        cover_count = defaultdict(int)
        for b in best_blocks:
            for sub in itertools.combinations(b, t):
                cover_count[sub] += 1
        best_blocks, _ = _remove_redundant(best_blocks, deadline)
        
    # Ensure distinct blocks
    if best_blocks is None:
        # Fallback
        rng = random.Random(42)
        best_blocks = _greedy_construct(rng)
        
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        b_sorted = sorted(b)
        key = tuple(b_sorted)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b_sorted)
            
    return unique_blocks
