import itertools
import random
import time
import sys

def construct(v, k, t):
    # Precompute all t-subsets and their IDs
    t_subs = list(itertools.combinations(range(v), t))
    sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    num_subs = len(t_subs)
    
    # Generate all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block index to list of t-subset IDs it covers
    block_to_subs = []
    # Map t-subset ID to list of block indices covering it
    sub_to_blocks = [[] for _ in range(num_subs)]
    
    for i, blk in enumerate(all_blocks):
        subs = [sub_to_id[sub] for sub in itertools.combinations(blk, t)]
        block_to_subs.append(subs)
        for sid in subs:
            sub_to_blocks[sid].append(i)

    start_time = time.time()
    timeout = 115.0
    
    best_size = float('inf')
    best_selected = None
    
    # Penalty for uncovered subsets. Must be large enough to prioritize coverage over size.
    # Max possible uncovered is num_subs. Max size is num_blocks.
    # If we use penalty P, cost = P * uncovered + size.
    # We want to prefer covering over adding blocks.
    # Adding a block costs 1. Uncovering a subset costs P.
    # If P > 1, we will always try to cover.
    # To ensure we don't pick 75000 blocks when 335 suffice, P should be large.
    # But if P is too large, the search space becomes "cover first, then minimize".
    # Let's use P = 10.
    PENALTY = 10
    
    def greedy_init():
        """Initialize with a greedy set cover to get a valid starting point."""
        uncovered = set(range(num_subs))
        selected = set()
        cover_count = [0] * num_subs
        
        while uncovered:
            # Pick a random uncovered subset to break symmetry and speed up
            target_sid = random.choice(list(uncovered))
            candidates = sub_to_blocks[target_sid]
            
            best_block = -1
            best_gain = -1
            # Randomize tie-breaking
            for b_idx in candidates:
                gain = 0
                for sid in block_to_subs[b_idx]:
                    if sid in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block != -1:
                selected.add(best_block)
                for sid in block_to_subs[best_block]:
                    cover_count[sid] += 1
                    if cover_count[sid] == 1:
                        uncovered.discard(sid)
        
        return selected, cover_count, len(uncovered) # uncovered should be 0

    num_restarts = 50
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        random.seed(42 + r)
        
        selected, cover_count, unc = greedy_init()
        current_uncovered = unc
        current_size = len(selected)
        
        # Calculate initial cost
        current_cost = current_uncovered * PENALTY + current_size
        
        # Local Search with Simulated Annealing / Random Walk
        temp = 5.0 # Initial temperature
        cooling = 0.999
        
        while time.time() - start_time < timeout:
            if current_uncovered == 0 and current_size < best_size:
                best_size = current_size
                best_selected = set(selected)
                # If we found a very good solution, we can reduce temperature faster
                # or just continue to try to improve further.
            
            # Choose a move
            move_type = random.random()
            
            delta_cost = 0
            move_applied = False
            b_in = -1
            b_out = -1
            gain = 0
            loss = 0
            
            if current_uncovered > 0:
                # Prioritize repair: Add a block covering an uncovered subset
                # Pick a random uncovered subset
                # We need to pick from uncovered. Maintaining a list of uncovered is expensive to update.
                # Instead, pick a random subset, if it's uncovered, use it. Else, pick random block?
                # Better: Pick a random block from all blocks? No, that's 75k.
                # Let's pick a random subset ID. If cover_count[sid] == 0, it's a target.
                # If not, we might still pick a random block to add (exploration).
                
                # Strategy: 80% chance pick a block covering a random uncovered subset.
                # 20% chance pick a random block.
                
                if random.random() < 0.8:
                    # Find an uncovered subset
                    # To avoid scanning all num_subs every time, we can keep a list of uncovered?
                    # Updating the list is O(1) amortized if we use a set and remove.
                    # But we don't have a set of uncovered in the main loop, only count.
                    # Let's just pick a random sid.
                    target_sid = random.randint(0, num_subs - 1)
                    if cover_count[target_sid] > 0:
                        # It's covered. Let's pick a random block to add anyway (exploration)
                        b_in = random.randint(0, num_blocks - 1)
                    else:
                        # It's uncovered. Pick a block covering it.
                        candidates = sub_to_blocks[target_sid]
                        b_in = random.choice(candidates)
                else:
                    b_in = random.randint(0, num_blocks - 1)
                
                if b_in not in selected:
                    # Calculate gain
                    for sid in block_to_subs[b_in]:
                        if cover_count[sid] == 0:
                            gain += 1
                    delta_cost = 1 - gain * PENALTY
                    b_out = -1 # Just adding
                else:
                    # Block already selected, skip or try remove?
                    # If we add an already selected block, it's a no-op.
                    # Let's force a different move type or skip.
                    continue

            else:
                # Current state is valid (uncovered == 0).
                # Try to reduce size or swap to allow future reductions.
                if random.random() < 0.5:
                    # Try Remove
                    if selected:
                        b_out = random.choice(list(selected))
                        # Calculate loss
                        for sid in block_to_subs[b_out]:
                            if cover_count[sid] == 1:
                                loss += 1
                        delta_cost = -1 + loss * PENALTY
                        b_in = -1
                    else:
                        continue
                else:
                    # Try Swap
                    if selected:
                        b_out = random.choice(list(selected))
                        # Pick a random b_in from blocks NOT in selected
                        # This is hard to do efficiently without a list of unselected.
                        # Let's pick a random block. If it's in selected, skip.
                        # Or, pick a block that covers a subset currently covered only by b_out?
                        # That's complex.
                        # Let's just pick a random block.
                        b_in = random.randint(0, num_blocks - 1)
                        if b_in in selected:
                            continue
                        
                        # Calculate effect of swap
                        # 1. Remove b_out
                        loss_out = 0
                        for sid in block_to_subs[b_out]:
                            if cover_count[sid] == 1:
                                loss_out += 1
                        # Temporarily update cover_count for calculation?
                        # Or just calculate gain_in assuming b_out is gone.
                        # gain_in is number of sids in b_in that have cover_count == 0 
                        # AFTER removing b_out.
                        # sids in b_in that are currently covered ONLY by b_out will become uncovered,
                        # but then covered by b_in.
                        # sids in b_in that are covered by other blocks remain covered.
                        
                        # Let's trace:
                        # Initial: cover_count[sid]
                        # After removing b_out: cover_count[sid] - (1 if sid in b_out else 0)
                        # We add b_in: cover_count[sid] + (1 if sid in b_in else 0)
                        
                        # Uncovered change:
                        # A sid becomes uncovered if it was covered (count > 0) and becomes 0.
                        # A sid becomes covered if it was uncovered (count == 0) and becomes > 0.
                        
                        # Let's just simulate the change in `uncovered_count` directly.
                        # This is O(k) per swap candidate if we are careful.
                        
                        # Actually, simpler:
                        # delta_uncovered = (loss_out - gain_in_effective)
                        # gain_in_effective:
                        # For each sid in b_in:
                        #   old_count = cover_count[sid]
                        #   new_count = old_count - (1 if sid in b_out else 0) + 1
                        #   if old_count == 0 and new_count > 0: covered (gain)
                        #   if old_count > 0 and new_count == 0: uncovered (loss)
                        
                        # Since b_in is not selected, and b_out is selected:
                        # If sid in b_in and not in b_out:
                        #   old_count >= 0. new_count = old_count + 1.
                        #   If old_count == 0, it was uncovered, now covered. Gain 1.
                        # If sid in b_in and in b_out:
                        #   old_count >= 1 (since b_out covers it).
                        #   new_count = old_count - 1 + 1 = old_count. No change.
                        # If sid not in b_in and in b_out:
                        #   old_count >= 1.
                        #   new_count = old_count - 1.
                        #   If old_count == 1, it was covered, now uncovered. Loss 1.
                        
                        # So:
                        # Gain = count of sid in (b_in \ b_out) such that cover_count[sid] == 0
                        # Loss = count of sid in (b_out \ b_in) such that cover_count[sid] == 1
                        
                        gain_swap = 0
                        loss_swap = 0
                        # We need to know which sids are in b_in \ b_out and b_out \ b_in.
                        # block_to_subs are lists. We can convert to sets for intersection/difference?
                        # That's O(56) which is fine.
                        set_in = set(block_to_subs[b_in])
                        set_out = set(block_to_subs[b_out])
                        
                        only_in = set_in - set_out
                        only_out = set_out - set_in
                        
                        for sid in only_in:
                            if cover_count[sid] == 0:
                                gain_swap += 1
                        for sid in only_out:
                            if cover_count[sid] == 1:
                                loss_swap += 1
                        
                        delta_uncovered = loss_swap - gain_swap
                        delta_cost = delta_uncovered * PENALTY
                        # size change is 0

            if not move_applied:
                # Acceptance criterion
                if delta_cost < 0:
                    accept = True
                else:
                    # Accept worsening moves with probability
                    if temp > 0.01:
                        accept = random.random() < (2.718 ** (-delta_cost / temp))
                    else:
                        accept = False
                
                if accept:
                    # Apply move
                    if b_in != -1:
                        selected.add(b_in)
                        current_size += 1
                        for sid in block_to_subs[b_in]:
                            cover_count[sid] += 1
                            if cover_count[sid] == 1:
                                current_uncovered -= 1
                    if b_out != -1:
                        selected.discard(b_out)
                        current_size -= 1
                        for sid in block_to_subs[b_out]:
                            cover_count[sid] -= 1
                            if cover_count[sid] == 0:
                                current_uncovered += 1
                    
                    current_cost += delta_cost
                    temp *= cooling
            else:
                # Should not happen with current logic, but just in case
                pass

    if best_selected is None:
        # Fallback: return greedy result from last run or a trivial invalid one
        # We should keep the last greedy result
        # Let's re-run greedy once if best is None
        selected, cover_count, unc = greedy_init()
        return [list(all_blocks[i]) for i in selected]
        
    return [list(all_blocks[i]) for i in best_selected]
