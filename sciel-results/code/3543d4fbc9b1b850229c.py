import random
import time
import itertools

def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute triple to index mapping
    all_triples = [frozenset(c) for c in itertools.combinations(range(v), t)]
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    total_triples = len(all_triples)
    
    # State
    blocks = []
    counts = [0] * total_triples
    uncovered = total_triples
    
    def update_counts(block, sign):
        nonlocal uncovered
        for c in itertools.combinations(block, t):
            idx = triple_to_idx[frozenset(c)]
            old_c = counts[idx]
            new_c = old_c + sign
            counts[idx] = new_c
            if old_c == 0 and new_c > 0:
                uncovered -= 1
            elif old_c > 0 and new_c == 0:
                uncovered += 1

    def get_block_triples(block):
        return [triple_to_idx[frozenset(c)] for c in itertools.combinations(block, t)]

    # 1. Greedy Initialization
    # Simple greedy: pick block covering most uncovered triples
    # To speed up, we can use a heuristic:
    # Start with empty set.
    # While uncovered > 0:
    #   Find block maximizing coverage of uncovered triples.
    #   Add it.
    # This is O(N_blocks * C(v,k) * C(k,t)). Too slow for C(24,7) = 346k blocks.
    # Instead, use the existing _greedy_cover logic but optimized.
    
    def greedy_cover():
        nonlocal blocks, counts, uncovered
        blocks = []
        counts = [0] * total_triples
        uncovered = total_triples
        uncovered_set = set(range(total_triples))
        
        while uncovered_set:
            # Pick a random uncovered triple
            seed_triple = rng.choice(list(uncovered_set))
            seed_points = list(all_triples[seed_triple])
            
            # Build block around seed
            block = set(seed_points)
            # Greedily add points
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    gain = 0
                    new_block = block | {x}
                    for c in itertools.combinations(new_block, t):
                        if len(block) < t: continue # Not efficient
                        # Check if triple containing x is uncovered
                        # Triple must contain x and t-1 from block
                        pass 
                    # Better gain calc:
                    # Triples in (block U {x}) that are NOT in block
                    # These are triples containing x and t-1 elements from block
                    for rest in itertools.combinations(block, t-1):
                        fs = frozenset(list(rest) + [x])
                        idx = triple_to_idx[fs]
                        if idx in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.add(best_x)
                
            block = tuple(sorted(block))
            if block not in blocks:
                blocks.append(block)
                # Update counts
                for c in itertools.combinations(block, t):
                    idx = triple_to_idx[frozenset(c)]
                    if counts[idx] == 0:
                        uncovered_set.discard(idx)
                    counts[idx] += 1
                uncovered = len(uncovered_set)
        return blocks

    greedy_cover()
    
    best_blocks = list(blocks)
    best_uncovered = 0 # Greedy ensures valid
    best_count = len(blocks)
    
    # Simulated Annealing
    T = 1000.0
    cooling = 0.999
    min_T = 1.0
    
    current_blocks = list(blocks)
    current_counts = list(counts)
    current_uncovered = uncovered
    
    # Helper to calculate delta for add/remove
    def calc_delta_add(block):
        # Returns (delta_blocks, delta_uncovered)
        # delta_blocks = 1
        # delta_uncovered = - (count of triples in block with count 0)
        d_unc = 0
        for c in itertools.combinations(block, t):
            idx = triple_to_idx[frozenset(c)]
            if current_counts[idx] == 0:
                d_unc -= 1
        return 1, d_unc

    def calc_delta_remove(block):
        # Returns (delta_blocks, delta_uncovered)
        # delta_blocks = -1
        # delta_uncovered = + (count of triples in block with count 1)
        d_unc = 0
        for c in itertools.combinations(block, t):
            idx = triple_to_idx[frozenset(c)]
            if current_counts[idx] == 1:
                d_unc += 1
        return -1, d_unc

    def apply_move(move_type, block1=None, block2=None):
        nonlocal current_blocks, current_counts, current_uncovered
        # Returns delta_E
        # E = N_blocks + 100 * N_uncovered
        
        if move_type == 'add':
            b = block1
            # Check duplicate
            if b in current_blocks:
                return float('inf')
            
            d_b, d_u = calc_delta_add(b)
            delta_E = d_b + 100 * d_u
            
            # Apply
            current_blocks.append(b)
            for c in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(c)]
                current_counts[idx] += 1
            current_uncovered += d_u
            return delta_E

        elif move_type == 'remove':
            b = block1
            # Find index
            try:
                idx_b = current_blocks.index(b)
            except ValueError:
                return float('inf')
                
            d_b, d_u = calc_delta_remove(b)
            delta_E = d_b + 100 * d_u
            
            # Apply
            current_blocks.pop(idx_b)
            for c in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(c)]
                current_counts[idx] -= 1
            current_uncovered += d_u
            return delta_E

        elif move_type == 'swap':
            b_old = block1
            b_new = block2
            if b_new in current_blocks:
                return float('inf')
                
            # Remove old, Add new
            # 1. Remove
            idx_b = current_blocks.index(b_old)
            d_b1, d_u1 = calc_delta_remove(b_old)
            
            # 2. Add (logic must account for removal first)
            # We need to update counts first to get accurate add delta?
            # Yes.
            for c in itertools.combinations(b_old, t):
                idx = triple_to_idx[frozenset(c)]
                current_counts[idx] -= 1
            current_uncovered += d_u1
            
            # Now calc add delta with updated counts
            d_b2, d_u2 = calc_delta_add(b_new)
            
            # Apply add
            current_blocks[idx_b] = b_new
            for c in itertools.combinations(b_new, t):
                idx = triple_to_idx[frozenset(c)]
                current_counts[idx] += 1
            current_uncovered += d_u2
            
            delta_E = (d_b1 + d_b2) + 100 * (d_u1 + d_u2)
            return delta_E

    def revert_move(move_type, block1=None, block2=None, delta_E=None):
        # We don't have a generic revert. 
        # Better to snapshot state or use inverse moves.
        # Inverse of Add is Remove.
        # Inverse of Remove is Add.
        # Inverse of Swap is Swap(new, old).
        pass

    # Simpler SA loop with explicit state tracking
    # To avoid complex revert, we can just store the state before move if accepted, 
    # or just re-apply inverse.
    
    # Let's use a simpler approach: 
    # Generate move, calculate delta, if accept, apply. If reject, do nothing.
    # But "Apply" modifies state.
    # So we calculate delta WITHOUT modifying, then modify if accepted.
    
    def get_delta(move_type, b1, b2=None):
        if move_type == 'add':
            if b1 in current_blocks: return float('inf')
            d_u = 0
            for c in itertools.combinations(b1, t):
                idx = triple_to_idx[frozenset(c)]
                if current_counts[idx] == 0: d_u -= 1
            return 1 + 100 * d_u
        
        elif move_type == 'remove':
            if b1 not in current_blocks: return float('inf')
            d_u = 0
            for c in itertools.combinations(b1, t):
                idx = triple_to_idx[frozenset(c)]
                if current_counts[idx] == 1: d_u += 1
            return -1 + 100 * d_u
        
        elif move_type == 'swap':
            if b2 in current_blocks: return float('inf')
            if b1 not in current_blocks: return float('inf')
            
            # Simulate
            # Remove b1
            temp_counts = list(current_counts) # Too slow? 2024 ints. OK.
            d_u1 = 0
            for c in itertools.combinations(b1, t):
                idx = triple_to_idx[frozenset(c)]
                old_c = temp_counts[idx]
                temp_counts[idx] -= 1
                if old_c == 1: d_u1 += 1
                
            # Add b2
            d_u2 = 0
            for c in itertools.combinations(b2, t):
                idx = triple_to_idx[frozenset(c)]
                old_c = temp_counts[idx]
                temp_counts[idx] += 1
                if old_c == 0: d_u2 -= 1
                
            return 0 + 100 * (d_u1 + d_u2)
            
    while time.time() - start_time < time_limit:
        if T < min_T: break
        
        move_type = rng.choice(['add', 'remove', 'swap'])
        delta = float('inf')
        
        if move_type == 'add':
            b_new = tuple(sorted(rng.sample(range(v), k)))
            delta = get_delta('add', b_new)
            if delta < float('inf'):
                if rng.random() < 1.0 or delta < 0: # Always accept if valid? No, SA
                    # Accept prob
                    if delta < 0 or rng.random() < pow(2.718, -delta/T):
                        # Apply
                        current_blocks.append(b_new)
                        for c in itertools.combinations(b_new, t):
                            idx = triple_to_idx[frozenset(c)]
                            current_counts[idx] += 1
                        # Update uncovered
                        # Recalculate uncovered is expensive. 
                        # We know delta_u.
                        # d_u = (delta - 1) / 100
                        d_u = (delta - 1) // 100
                        current_uncovered += d_u
                        
                        if current_uncovered == 0 and len(current_blocks) < best_count:
                            best_count = len(current_blocks)
                            best_blocks = list(current_blocks)
                            
        elif move_type == 'remove':
            if not current_blocks: continue
            b_old = rng.choice(current_blocks)
            delta = get_delta('remove', b_old)
            if delta < float('inf'):
                if delta < 0 or rng.random() < pow(2.718, -delta/T):
                    # Apply
                    current_blocks.remove(b_old)
                    for c in itertools.combinations(b_old, t):
                        idx = triple_to_idx[frozenset(c)]
                        current_counts[idx] -= 1
                    d_u = (delta + 1) // 100
                    current_uncovered += d_u
                    
                    if current_uncovered == 0 and len(current_blocks) < best_count:
                        best_count = len(current_blocks)
                        best_blocks = list(current_blocks)
                        
        elif move_type == 'swap':
            if not current_blocks: continue
            b_old = rng.choice(current_blocks)
            b_new = tuple(sorted(rng.sample(range(v), k)))
            delta = get_delta('swap', b_old, b_new)
            if delta < float('inf'):
                if delta < 0 or rng.random() < pow(2.718, -delta/T):
                    # Apply
                    idx = current_blocks.index(b_old)
                    current_blocks[idx] = b_new
                    # Update counts
                    for c in itertools.combinations(b_old, t):
                        idx_t = triple_to_idx[frozenset(c)]
                        current_counts[idx_t] -= 1
                    for c in itertools.combinations(b_new, t):
                        idx_t = triple_to_idx[frozenset(c)]
                        current_counts[idx_t] += 1
                    d_u = delta // 100
                    current_uncovered += d_u
                    
                    if current_uncovered == 0 and len(current_blocks) < best_count:
                        best_count = len(current_blocks)
                        best_blocks = list(current_blocks)
                            
        T *= cooling
        
    # Final Polish: Remove redundant blocks from best_blocks
    # Reconstruct counts for best_blocks
    final_counts = [0] * total_triples
    for b in best_blocks:
        for c in itertools.combinations(b, t):
            final_counts[triple_to_idx[frozenset(c)]] += 1
            
    final_blocks = []
    for b in best_blocks:
        redundant = True
        for c in itertools.combinations(b, t):
            idx = triple_to_idx[frozenset(c)]
            if final_counts[idx] == 1:
                redundant = False
                break
        if not redundant:
            final_blocks.append(b)
            for c in itertools.combinations(b, t):
                final_counts[triple_to_idx[frozenset(c)]] -= 1
        # If redundant, we skip it (remove it)
        
    # Check validity of final_blocks
    # It should be valid if best_blocks was valid and we only removed redundant ones.
    
    return [list(b) for b in final_blocks]
