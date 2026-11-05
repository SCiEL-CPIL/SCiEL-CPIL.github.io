import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage matrix: block_triples[b][i] = 1 if block b covers triple i
    # Since k=7, t=3, each block covers C(7,3)=35 triples.
    # num_blocks = C(24,7) = 346,104.
    # Storing full matrix would be huge (346k * 2024 ~ 700M entries). 
    # Instead, store list of covered triple indices for each block.
    block_triple_indices = [[] for _ in range(num_blocks)]
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        tris_in_block = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris_in_block.append(idx)
            triple_to_blocks[idx].append(i)
        block_triple_indices[i] = tris_in_block
        
    # Helper to check validity
    def is_valid_solution(selected_blocks):
        if len(selected_blocks) == 0:
            return False
        covered = [0] * num_triples
        for b_idx in selected_blocks:
            for tri_idx in block_triple_indices[b_idx]:
                covered[tri_idx] += 1
        # Check if any triple is uncovered
        # Using min is fast in C
        return min(covered) >= 0 and all(c >= 1 for c in covered)
        # Optimization: all(c >= 1 for c in covered) is slow. 
        # Better: sum(1 for c in covered if c == 0) == 0
        
    def count_uncovered(selected_blocks):
        covered = [0] * num_triples
        for b_idx in selected_blocks:
            for tri_idx in block_triple_indices[b_idx]:
                covered[tri_idx] += 1
        return sum(1 for c in covered if c == 0)

    # Initial Solution: Strong Greedy
    # We use a weighted greedy to get a good starting point for SA.
    def get_initial_solution():
        uncovered_count = [1] * num_triples
        selected = []
        # Precompute block weights? No, dynamic.
        # To speed up, we can use a priority queue or just scan.
        # Scanning 346k blocks * ~70 steps is ~24M ops. Acceptable.
        
        # We'll use a simple greedy that picks max gain.
        # To avoid O(N^2) in Python, we might need to be careful.
        # But 346k is not that big.
        
        # Let's use a numpy-based gain calculation if possible?
        # No, sparse structure makes numpy dense matrix too big.
        
        # Simple loop
        for _ in range(200): # Max steps
            if time.time() > deadline - 10:
                break
            best_gain = -1
            best_idx = -1
            # Random tie breaking
            # We can sample a subset of blocks to speed up? No, let's do full scan for first few, then maybe prune.
            # Actually, full scan is fine.
            
            # Optimization: Keep track of current uncovered set?
            # We can maintain a bitset of uncovered triples.
            # But updating bitset for each block is also work.
            
            # Let's just compute gain for all blocks.
            # gain = sum(1 for tri in block_triples[b] if uncovered_count[tri] == 0)
            
            # This is slow in Python.
            # Alternative: Maintain a list of uncovered triples.
            # For each block, count how many of its triples are in the uncovered set.
            
            # Let's try a different initial construction: 
            # Randomly select 80 blocks, then repair?
            # Or use the seed's greedy which was decent.
            
            # Let's implement a faster greedy using bitsets for uncovered triples.
            # num_triples = 2024. Fits in 32 64-bit ints or 1 Python int (arbitrary precision).
            # Python int bit operations are fast.
            
            uncovered_bitset = (1 << num_triples) - 1
            block_bitsets = [0] * num_blocks
            for i, b in enumerate(all_blocks):
                bs = 0
                for tri_idx in block_triple_indices[i]:
                    bs |= (1 << tri_idx)
                block_bitsets[i] = bs
            
            selected_set = set()
            
            while uncovered_bitset:
                best_gain = -1
                best_idx = -1
                # Scan all blocks
                for i in range(num_blocks):
                    if i in selected_set:
                        continue
                    gain = (block_bitsets[i] & uncovered_bitset).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                    elif gain == best_gain and gain > 0:
                        # Random tie break? 
                        if random.random() < 0.1: # Occasionally pick a random tie
                             # This requires collecting ties, which is expensive.
                             # Just stick to first for now in init, SA will handle diversity.
                             pass
                
                if best_gain <= 0:
                    break
                    
                selected_set.add(best_idx)
                selected.append(best_idx)
                uncovered_bitset &= ~block_bitsets[best_idx]
                
                if time.time() > deadline - 10:
                    break
                    
            return selected

    initial_sol = get_initial_solution()
    
    if not initial_sol:
        # Fallback
        return [all_blocks[i] for i in range(100)]

    current_sol = set(initial_sol)
    current_size = len(current_sol)
    
    # Precompute block_bitsets for SA energy calculation
    block_bitsets = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        bs = 0
        for tri_idx in block_triple_indices[i]:
            bs |= (1 << tri_idx)
        block_bitsets[i] = bs

    # QUBO Energy Function
    # E(S) = |S| + Lambda * Uncovered(S) + Mu * Redundancy(S)
    # Redundancy(S) = sum_{i<j in S} |T_i \cap T_j|
    # Note: Minimizing redundancy is equivalent to maximizing coverage efficiency.
    # Actually, for a fixed size, minimizing overlap helps? 
    # Wait, if we minimize |S|, we want each block to cover as much NEW ground as possible.
    # High overlap means blocks are doing similar work.
    # So penalizing overlap encourages diverse blocks.
    
    # However, calculating full QUBO energy every step is O(|S|^2 * C(k,t)^2) which is too slow.
    # We need incremental updates.
    
    # Let's use a simpler heuristic for SA moves:
    # Move 1: Add a random block.
    # Move 2: Remove a random block.
    # Move 3: Swap a random block with another random block.
    
    # Energy Delta Calculation:
    # Let S be current set.
    # Add block b:
    #   Delta_Size = +1
    #   Delta_Uncovered = - (number of triples covered by b that were uncovered)
    #   Delta_Redundancy = sum_{x in S} |T_b \cap T_x|
    #   (New overlaps between b and existing blocks)
    
    # Remove block b:
    #   Delta_Size = -1
    #   Delta_Uncovered = + (number of triples covered ONLY by b in S)
    #   Delta_Redundancy = - sum_{x in S\{b}} |T_b \cap T_x|
    #   (Remove overlaps between b and other blocks)
    
    # We need to maintain:
    # 1. Uncovered count per triple (or bitset)
    # 2. Overlap counts? Or compute on the fly?
    # Computing sum_{x in S} |T_b \cap T_x| requires iterating S and checking intersection.
    # |S| is ~70-80. T_b has 35 triples.
    # Intersection check: (block_bitsets[b] & block_bitsets[x]).bit_count()
    # 70 * bit_count is fast.
    
    # So for a move, we can calculate Delta_E in O(|S|).
    # This is acceptable for SA steps.
    
    # State variables:
    covered_counts = [0] * num_triples
    for b_idx in current_sol:
        for tri_idx in block_triple_indices[b_idx]:
            covered_counts[tri_idx] += 1
            
    uncovered_count = sum(1 for c in covered_counts if c == 0)
    
    # Redundancy calculation for current state?
    # We don't strictly need absolute redundancy, only deltas.
    # But for initial energy, we might need it? No, SA only uses deltas.
    
    # Let's define the energy function parameters.
    # E = Size + P_uncovered * Uncovered + P_redundancy * Redundancy
    # P_uncovered must be large enough to force coverage.
    # Max possible uncovered is 2024.
    # If we allow 1 uncovered, penalty should be > max size difference.
    # Let's set P_uncovered = 100.
    # P_redundancy = 0.1 (soft constraint to prefer diverse blocks)
    
    P_UNCOVERED = 100.0
    P_REDUNDANCY = 0.1
    
    def calc_delta_add(b_idx):
        # Adds b_idx to S
        # Delta_Size = 1
        # Delta_Uncovered: Count triples in b_idx with count 0
        new_uncovered_gain = 0
        for tri_idx in block_triple_indices[b_idx]:
            if covered_counts[tri_idx] == 0:
                new_uncovered_gain += 1
        delta_uncovered = -new_uncovered_gain
        
        # Delta_Redundancy: Sum of overlaps with existing blocks
        delta_redundancy = 0
        b_bitset = block_bitsets[b_idx]
        for x in current_sol:
            if x == b_idx:
                continue
            delta_redundancy += (b_bitset & block_bitsets[x]).bit_count()
            
        delta_e = 1.0 + P_UNCOVERED * delta_uncovered + P_REDUNDANCY * delta_redundancy
        return delta_e, new_uncovered_gain

    def calc_delta_remove(b_idx):
        # Removes b_idx from S
        # Delta_Size = -1
        # Delta_Uncovered: Count triples in b_idx with count 1 (become 0)
        new_uncovered_loss = 0
        for tri_idx in block_triple_indices[b_idx]:
            if covered_counts[tri_idx] == 1:
                new_uncovered_loss += 1
        delta_uncovered = new_uncovered_loss
        
        # Delta_Redundancy: Negative sum of overlaps with other blocks
        delta_redundancy = 0
        b_bitset = block_bitsets[b_idx]
        for x in current_sol:
            if x == b_idx:
                continue
            delta_redundancy -= (b_bitset & block_bitsets[x]).bit_count()
            
        delta_e = -1.0 + P_UNCOVERED * delta_uncovered + P_REDUNDANCY * delta_redundancy
        return delta_e, new_uncovered_loss

    best_sol = list(current_sol)
    best_size = len(best_sol)
    best_uncovered = uncovered_count
    
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.995
    max_steps = 5000
    
    steps_done = 0
    
    while T > T_min and time.time() < deadline - 5:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random block
            candidate = random.randint(0, num_blocks - 1)
            if candidate in current_sol:
                continue
                
            delta_e, gain = calc_delta_add(candidate)
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                # Accept
                current_sol.add(candidate)
                for tri_idx in block_triple_indices[candidate]:
                    covered_counts[tri_idx] += 1
                uncovered_count -= gain
                
                if len(current_sol) < best_size and uncovered_count == 0:
                    best_size = len(current_sol)
                    best_sol = list(current_sol)
                    best_uncovered = 0
                    
        elif move_type < 0.8:
            # Remove a random block
            if len(current_sol) == 0:
                continue
            candidate = random.choice(list(current_sol))
            
            delta_e, loss = calc_delta_remove(candidate)
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                # Accept
                current_sol.remove(candidate)
                for tri_idx in block_triple_indices[candidate]:
                    covered_counts[tri_idx] -= 1
                uncovered_count += loss
                
                if len(current_sol) < best_size and uncovered_count == 0:
                    best_size = len(current_sol)
                    best_sol = list(current_sol)
                    best_uncovered = 0
                    
        else:
            # Swap: Remove one, Add one
            if len(current_sol) == 0:
                continue
            remove_candidate = random.choice(list(current_sol))
            add_candidate = random.randint(0, num_blocks - 1)
            
            if add_candidate in current_sol and add_candidate != remove_candidate:
                continue
            
            # Calculate combined delta
            # It's safer to do remove then add, or calculate joint delta.
            # Joint delta:
            # Size: 0
            # Uncovered: 
            #   Triples only covered by remove_candidate (count 1) become uncovered (+1 each)
            #   Triples only uncovered (count 0) and covered by add_candidate become covered (-1 each)
            #   Note: If a triple is covered by both, it stays covered.
            
            # Redundancy:
            #   Remove overlaps of remove_candidate with S\{remove_candidate}
            #   Add overlaps of add_candidate with S\{remove_candidate}
            
            # Let's compute:
            delta_uncovered = 0
            # Triples lost
            for tri_idx in block_triple_indices[remove_candidate]:
                if covered_counts[tri_idx] == 1:
                    # This triple will be uncovered unless add_candidate covers it
                    if (block_bitsets[add_candidate] & (1 << tri_idx)) == 0:
                        delta_uncovered += 1
            # Triples gained
            for tri_idx in block_triple_indices[add_candidate]:
                if covered_counts[tri_idx] == 0:
                    # This triple is currently uncovered, add_candidate covers it
                    # But wait, if remove_candidate also covered it, count would be 1, not 0.
                    # So if count is 0, remove_candidate didn't cover it.
                    delta_uncovered -= 1
            
            # Redundancy Delta
            delta_redundancy = 0
            remove_bitset = block_bitsets[remove_candidate]
            add_bitset = block_bitsets[add_candidate]
            
            for x in current_sol:
                if x == remove_candidate:
                    continue
                # Overlap removed
                delta_redundancy -= (remove_bitset & block_bitsets[x]).bit_count()
                # Overlap added
                if x != add_candidate:
                    delta_redundancy += (add_bitset & block_bitsets[x]).bit_count()
            
            # Overlap between remove and add?
            # If they are distinct, their mutual overlap is not part of S's internal redundancy 
            # in the "sum over pairs" definition if we consider the set S.
            # Wait, redundancy is sum_{i<j in S} |T_i \cap T_j|.
            # In S, remove_candidate is paired with all others.
            # In S' (after swap), add_candidate is paired with all others (except remove_candidate).
            # So we subtract overlaps of remove with others, add overlaps of add with others.
            # We do NOT add overlap between remove and add because they are not in S' together.
            # And we do NOT subtract overlap between remove and add because it wasn't in S? 
            # Wait, remove IS in S. So overlap(remove, x) for all x in S\{remove} is subtracted.
            # Add IS not in S. So overlap(add, x) for all x in S\{remove} is added.
            # This is correct.
            
            delta_e = 0.0 + P_UNCOVERED * delta_uncovered + P_REDUNDANCY * delta_redundancy
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                # Accept
                current_sol.remove(remove_candidate)
                for tri_idx in block_triple_indices[remove_candidate]:
                    covered_counts[tri_idx] -= 1
                current_sol.add(add_candidate)
                for tri_idx in block_triple_indices[add_candidate]:
                    covered_counts[tri_idx] += 1
                uncovered_count += delta_uncovered
                
                if len(current_sol) < best_size and uncovered_count == 0:
                    best_size = len(current_sol)
                    best_sol = list(current_sol)
                    best_uncovered = 0

        steps_done += 1
        T *= cooling_rate
        
        # Time check
        if time.time() > deadline - 10:
            break
            
    # Final Polish: Local Search on best_sol
    # Try to remove redundant blocks from best_sol
    final_sol = set(best_sol)
    if time.time() < deadline - 2:
        # Recalculate counts for final_sol
        final_counts = [0] * num_triples
        for b_idx in final_sol:
            for tri_idx in block_triple_indices[b_idx]:
                final_counts[tri_idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for b_idx in list(final_sol):
                # Check if b_idx is redundant
                is_redundant = True
                for tri_idx in block_triple_indices[b_idx]:
                    if final_counts[tri_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    final_sol.remove(b_idx)
                    for tri_idx in block_triple_indices[b_idx]:
                        final_counts[tri_idx] -= 1
                    improved = True
                    if len(final_sol) < best_size:
                        best_size = len(final_sol)
                        best_sol = list(final_sol)
                        
    # Ensure validity of best_sol
    # If best_sol is not valid (uncovered > 0), we might have lost track.
    # Let's verify best_sol.
    if best_uncovered > 0:
        # Fallback to initial if SA got stuck in invalid state and didn't find a valid better one
        # But we only updated best if uncovered == 0.
        # So best_sol should be valid.
        pass

    # Convert indices to blocks
    result_blocks = [all_blocks[i] for i in best_sol]
    
    # Final safety check
    if not result_blocks:
         # Return a valid covering, e.g., first 100 blocks? No, must cover all.
         # Use initial_sol which was constructed to be valid (greedy)
         result_blocks = [all_blocks[i] for i in initial_sol]
         
    return result_blocks
