import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Pre-compute all t-subsets for the universe
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map triple to index for faster counting
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Helper to get indices of triples in a block
    def get_block_triple_indices(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, t)]

    # 1. Initial Solution via Greedy
    # We start with a good greedy solution to give SA a strong starting point.
    def greedy_construct():
        uncovered = set(range(num_triples))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start the block
            start_tri_idx = random.choice(list(uncovered))
            start_tri = all_triples[start_tri_idx]
            
            block = list(start_tri)
            
            while len(block) < k:
                current_block_set = set(block)
                candidates = [x for x in range(v) if x not in current_block_set]
                
                best_gain = -1
                best_candidates = []
                
                for x in candidates:
                    # Calculate gain: number of triples in new_block that are in uncovered
                    gain = 0
                    for pair in itertools.combinations(block, t-1):
                        tri = tuple(sorted(pair + (x,)))
                        tri_idx = triple_to_idx[tri]
                        if tri_idx in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                        
                if best_candidates:
                    block.append(random.choice(best_candidates))
                else:
                    block.append(random.choice(candidates))
                    
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for tri_idx in get_block_triple_indices(block):
                if tri_idx in uncovered:
                    uncovered.remove(tri_idx)
                    
        return blocks

    current_blocks = greedy_construct()
    best_blocks = current_blocks[:]
    best_len = len(current_blocks)
    
    # State for optimization
    # counts[i] is the number of blocks covering triple i
    counts = [0] * num_triples
    for b in current_blocks:
        for tri_idx in get_block_triple_indices(b):
            counts[tri_idx] += 1
            
    # Calculate initial "cost" (penalty for invalidity + size)
    # We want to minimize: (number of blocks) + (M * number of uncovered triples)
    # M should be large enough to ensure validity is prioritized over size,
    # but SA allows temporary invalidity.
    # Let's use a dynamic penalty or a fixed large penalty.
    # For v=24, k=7, t=3, max blocks ~ 80. Max uncovered ~ 2024.
    # If M=100, one uncovered triple costs 100 blocks. This forces validity quickly.
    PENALTY = 100
    
    def get_cost(blocks, counts):
        # Cost = len(blocks) + PENALTY * (number of uncovered triples)
        uncovered_count = 0
        for c in counts:
            if c == 0:
                uncovered_count += 1
        return len(blocks) + PENALTY * uncovered_count

    current_cost = get_cost(current_blocks, counts)
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    # Initial temperature high enough to accept bad moves
    T_start = 50.0
    T_end = 0.1
    # Cooling rate
    alpha = 0.995
    
    # Main SA Loop
    T = T_start
    no_improve_steps = 0
    
    while time.time() < deadline and T > T_end:
        # Generate a neighbor
        # Move types:
        # 1. Remove a random block (if valid or penalty accepted)
        # 2. Add a random block
        # 3. Swap a point in a block (change one element of a block)
        # 4. Replace a block with a random one
        
        move_type = random.randint(0, 3)
        old_blocks = current_blocks[:]
        old_counts = counts[:]
        old_cost = current_cost
        
        if move_type == 0:
            # Remove a random block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                b_remove = current_blocks[idx]
                b_indices = get_block_triple_indices(b_remove)
                
                # Tentatively remove
                for tri_idx in b_indices:
                    counts[tri_idx] -= 1
                    
                current_blocks.pop(idx)
                
        elif move_type == 1:
            # Add a random block
            # To make this useful, maybe pick a block that covers many uncovered triples?
            # Or just random? Random is better for exploration.
            # But adding a block increases cost by 1 (plus penalty change).
            # Usually we want to remove blocks. Adding is only useful if it allows removing 2 later?
            # In SA, we accept worse moves.
            new_block = tuple(sorted(random.sample(range(v), k)))
            if new_block not in current_blocks:
                current_blocks.append(new_block)
                for tri_idx in get_block_triple_indices(new_block):
                    counts[tri_idx] += 1
        elif move_type == 2:
            # Change one point in a random block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                b = current_blocks[idx]
                b_indices_old = get_block_triple_indices(b)
                
                # Remove old block from counts
                for tri_idx in b_indices_old:
                    counts[tri_idx] -= 1
                    
                # Pick a random position in the block and a random new point
                pos = random.randint(0, k-1)
                old_point = b[pos]
                new_point = random.choice([x for x in range(v) if x not in b])
                
                new_b_list = list(b)
                new_b_list[pos] = new_point
                new_b_list.sort()
                new_b = tuple(new_b_list)
                
                # Add new block to counts
                for tri_idx in get_block_triple_indices(new_b):
                    counts[tri_idx] += 1
                    
                current_blocks[idx] = new_b
        else:
            # Replace a random block with a random other block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                b_old = current_blocks[idx]
                b_indices_old = get_block_triple_indices(b_old)
                
                for tri_idx in b_indices_old:
                    counts[tri_idx] -= 1
                    
                new_b = tuple(sorted(random.sample(range(v), k)))
                for tri_idx in get_block_triple_indices(new_b):
                    counts[tri_idx] += 1
                    
                current_blocks[idx] = new_b

        # Calculate new cost
        new_cost = get_cost(current_blocks, counts)
        
        # Acceptance criterion
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0:
            # Better move, always accept
            current_cost = new_cost
            no_improve_steps = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_blocks = current_blocks[:]
                best_len = len(current_blocks)
        else:
            # Worse move, accept with probability exp(-delta/T)
            # Note: delta_cost is in units of "cost".
            # Since PENALTY is 100, a single uncovered triple is cost 100.
            # T=50 initially. exp(-100/50) = exp(-2) ~ 0.13.
            # This allows significant invalidity early on.
            if random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                # Don't reset no_improve_steps on worse moves? 
                # Usually we track time since last improvement.
                pass
            else:
                # Revert
                current_blocks = old_blocks
                counts = old_counts
                current_cost = old_cost
        
        # Cooling
        T *= alpha
        
        # Optional: If stuck for too long, reheat slightly?
        # Or just let it cool.
        
    # Final Cleanup: Ensure validity
    # The SA might end in an invalid state.
    # We must repair it to be a valid covering.
    # We use the greedy repair strategy on the best valid block found,
    # or repair the current state if it's close.
    
    # Actually, we should keep track of the best *valid* solution specifically.
    # The `best_blocks` above might be invalid if `best_cost` was dominated by size but had high penalty?
    # No, `get_cost` includes penalty. If `best_cost` is low, it likely has low penalty.
    # But to be safe, let's enforce validity on the result.
    
    # Rebuild counts for best_blocks
    best_counts = [0] * num_triples
    for b in best_blocks:
        for tri_idx in get_block_triple_indices(b):
            best_counts[tri_idx] += 1
            
    # Check if best_blocks is valid
    is_valid = True
    for c in best_counts:
        if c == 0:
            is_valid = False
            break
            
    if not is_valid:
        # Repair best_blocks using greedy logic
        # Find uncovered triples
        uncovered = [i for i in range(num_triples) if best_counts[i] == 0]
        
        # Greedily add blocks to cover uncovered triples
        while uncovered:
            # Pick a random uncovered triple
            target_idx = random.choice(uncovered)
            target_tri = all_triples[target_idx]
            
            # Build a block greedily
            repair_block = list(target_tri)
            while len(repair_block) < k:
                current_set = set(repair_block)
                candidates = [x for x in range(v) if x not in current_set]
                
                best_gain = -1
                best_cands = []
                for x in candidates:
                    gain = 0
                    for pair in itertools.combinations(repair_block, t-1):
                        tri = tuple(sorted(pair + (x,)))
                        tri_idx = triple_to_idx[tri]
                        if best_counts[tri_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_cands = [x]
                    elif gain == best_gain:
                        best_cands.append(x)
                        
                if best_cands:
                    repair_block.append(random.choice(best_cands))
                else:
                    repair_block.append(random.choice(candidates))
                    
            repair_block.sort()
            repair_block = tuple(repair_block)
            
            # Add to best_blocks
            if repair_block not in best_blocks:
                best_blocks.append(repair_block)
                for tri_idx in get_block_triple_indices(repair_block):
                    best_counts[tri_idx] += 1
                    if best_counts[tri_idx] == 1:
                        # It was uncovered, now covered
                        try:
                            uncovered.remove(tri_idx)
                        except ValueError:
                            pass
            else:
                # If block already exists, it doesn't help with new coverage?
                # Wait, if block exists, its triples are already counted.
                # If they were uncovered, then this block wasn't in best_blocks?
                # Contradiction. If repair_block is in best_blocks, its triples have count > 0.
                # So they wouldn't be in `uncovered`.
                # So this branch shouldn't happen if logic is tight.
                # But if it does, we just break or pick another?
                # Actually, if repair_block is already there, it covers target_tri.
                # So target_tri should have count > 0.
                # So target_tri should not be in `uncovered`.
                # So this implies a bug in my logic or state.
                # Let's just break to avoid infinite loop.
                break
                
    # Final redundancy removal
    # Rebuild counts for the final valid best_blocks
    final_counts = [0] * num_triples
    for b in best_blocks:
        for tri_idx in get_block_triple_indices(b):
            final_counts[tri_idx] += 1
            
    final_blocks = []
    for b in best_blocks:
        b_indices = get_block_triple_indices(b)
        can_remove = True
        for tri_idx in b_indices:
            if final_counts[tri_idx] == 1:
                can_remove = False
                break
        if not can_remove:
            final_blocks.append(b)
            for tri_idx in b_indices:
                final_counts[tri_idx] -= 1
                
    return [list(b) for b in final_blocks]
