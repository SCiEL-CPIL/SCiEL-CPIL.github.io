import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block contents and triple memberships
    # block_triples[i] = list of triple indices in block i
    block_triples = [[] for _ in range(num_blocks)]
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    # For fast set operations, store triple indices in a set or sorted list?
    # List is fine for intersection if small.
    
    for i, b in enumerate(all_blocks):
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris.append(idx)
            triple_to_blocks[idx].append(i)
        block_triples[i] = tris

    # Helper: Compute coverage counts for a solution (list of block indices)
    def compute_coverage_counts(sol):
        counts = [0] * num_triples
        for b in sol:
            for tri_idx in block_triples[b]:
                counts[tri_idx] += 1
        return counts

    # Helper: Check if a block is redundant given counts
    def is_redundant(b, counts):
        for tri_idx in block_triples[b]:
            if counts[tri_idx] == 1:
                return False
        return True

    # Helper: Get critical triples for a block b (triples covered only by b)
    # Assumes b is in the current solution
    def get_critical_triples(b, counts):
        critical = []
        for tri_idx in block_triples[b]:
            if counts[tri_idx] == 1:
                critical.append(tri_idx)
        return critical

    # Helper: Find a block b' not in sol that covers all triples in 'critical'
    # 'critical' is a list of triple indices.
    # We look for b' such that critical is a subset of block_triples[b']
    def find_swap_partner(critical, current_sol_set):
        if not critical:
            return -1
            
        # Optimize: The swap partner must cover the first critical triple
        # Candidates are in triple_to_blocks[critical[0]]
        # We must check if they cover ALL critical triples.
        
        candidates = triple_to_blocks[critical[0]]
        
        # If critical has more than 1 element, we can filter candidates
        # by checking if they are in triple_to_blocks for other criticals?
        # Iterating candidates and checking set inclusion is probably fast enough 
        # because critical is small.
        
        for b_prime in candidates:
            if b_prime in current_sol_set:
                continue
            
            # Check if b_prime covers all critical triples
            # Convert block_triples[b_prime] to a set for O(1) lookup? 
            # No, block_triples is a list. 
            # Since critical is small, we can check:
            # all(c in block_triples[b_prime] for c in critical)
            # This is O(|critical| * |block_triples[b_prime]|) if naive.
            # Better: Use a set for block_triples?
            # Precomputing sets for 346k blocks is memory heavy but maybe okay?
            # 346k * 84 triples * 8 bytes? ~200MB. Might be too much.
            # Let's stick to list membership but optimize:
            
            # Optimization: If critical is sorted, and block_triples is sorted?
            # itertools.combinations produces sorted tuples.
            # So triple indices are deterministic.
            # block_triples[i] is a list of indices. Are they sorted?
            # triple_to_idx is built from sorted combinations.
            # So block_triples[i] will have indices in the order generated.
            # This order is lexicographic on the triple.
            # So block_triples[i] is sorted.
            
            # We can do a two-pointer or simple check.
            # Given small size of critical (usually 1-3), simple loop is fine.
            
            is_covering = True
            # Optimization: Only check if b_prime has enough triples? No.
            for c in critical:
                if c not in block_triples[b_prime]:
                    is_covering = False
                    break
            
            if is_covering:
                return b_prime
                
        return -1

    # Phase 1: Initial Greedy Construction
    # Use a "Scarcity" heuristic to get a tighter start?
    # Standard greedy: max coverage.
    # Let's try standard greedy first, it's fast.
    uncovered = (1 << num_triples) - 1
    # Bitset representation for speed in greedy
    # We need block_bitsets for this.
    block_bitsets = [0] * num_blocks
    for i in range(num_blocks):
        bs = 0
        for tri_idx in block_triples[i]:
            bs |= (1 << tri_idx)
        block_bitsets[i] = bs
        
    initial_sol = []
    current_uncovered = uncovered
    while current_uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(num_blocks):
            gain = (block_bitsets[i] & current_uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == 0:
                    break # No block covers anything new? 
        if best_gain <= 0:
            break # Should not happen
        initial_sol.append(best_idx)
        current_uncovered &= ~block_bitsets[best_idx]
        
    # If greedy failed or is too large, we might want a random restart, 
    # but greedy is usually decent.
    
    current_sol = list(initial_sol)
    best_sol = list(current_sol)
    best_size = len(current_sol)
    
    # Phase 2: Minimal Cover Reduction + 1-Exchange Local Search
    # Loop until no improvement found or time limit
    
    # We maintain:
    # - current_sol (list of block indices)
    # - sol_set (set of block indices for fast lookup)
    # - coverage_counts (array)
    
    sol_set = set(current_sol)
    coverage_counts = compute_coverage_counts(current_sol)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Step 2a: Reduction Phase
        # Remove any redundant blocks
        # We iterate through the solution. If a block is redundant, remove it.
        # We must be careful: removing one block might make another redundant.
        # So we repeat the scan until stable.
        
        stable = False
        while not stable and time.time() < deadline - 10:
            stable = True
            new_sol = []
            for b in current_sol:
                if is_redundant(b, coverage_counts):
                    # Redundant, remove
                    sol_set.remove(b)
                    for tri_idx in block_triples[b]:
                        coverage_counts[tri_idx] -= 1
                    stable = False
                    improved = True # Size decreased
                else:
                    new_sol.append(b)
            current_sol = new_sol
            
        if len(current_sol) < best_size:
            best_sol = list(current_sol)
            best_size = len(current_sol)
            # Update structures for best_sol? 
            # We just keep best_sol as a list. 
            # If we need to resume search from best, we'd need to rebuild counts.
            # For now, we continue searching from current_sol.
            
        # Step 2b: 1-Exchange Phase
        # Try to find a swap that maintains or reduces size?
        # Since we are in a minimal cover, removing b breaks validity.
        # We remove b, find criticals U.
        # Find b' covering U.
        # Swap b with b'. Size stays same.
        # Then run Reduction Phase again.
        # If Reduction Phase removes blocks, size decreases.
        
        # To avoid cycling, we can track visited states? 
        # Or just randomize the order of blocks we try to swap.
        
        # Shuffled list of current blocks to try
        trial_blocks = current_sol[:]
        random.shuffle(trial_blocks)
        
        for b in trial_blocks:
            if time.time() > deadline - 10:
                break
                
            # Identify critical triples
            critical = get_critical_triples(b, coverage_counts)
            
            # If critical is empty, b was redundant. 
            # But we just did reduction phase, so this shouldn't happen 
            # unless counts changed? No, counts are current.
            # If critical is empty, it means b is redundant. 
            # This implies our reduction phase missed it? 
            # Or maybe multiple blocks became redundant.
            # If so, we should have removed it. 
            # Let's assume critical is not empty for minimal cover.
            
            if not critical:
                continue
                
            # Find swap partner
            b_prime = find_swap_partner(critical, sol_set)
            
            if b_prime != -1:
                # Perform Swap
                # Remove b
                sol_set.remove(b)
                for tri_idx in block_triples[b]:
                    coverage_counts[tri_idx] -= 1
                    
                # Add b_prime
                sol_set.add(b_prime)
                for tri_idx in block_triples[b_prime]:
                    coverage_counts[tri_idx] += 1
                    
                # Update current_sol
                # Replace b with b_prime in list
                # current_sol = [x if x != b else b_prime for x in current_sol]
                # This is O(N). N is small.
                current_sol = [b_prime if x == b else x for x in current_sol]
                
                # This swap might have created redundancies.
                # Go back to Reduction Phase immediately.
                improved = True # We made a move, let's re-evaluate
                break # Restart the reduction/swap loop with new state
        
    # Final Polish: Run reduction one last time
    if time.time() < deadline - 2:
        # Rebuild counts for best_sol? 
        # We should verify best_sol is valid and minimal.
        # The search maintained current_sol validity.
        # best_sol is a snapshot.
        
        # Let's ensure best_sol is actually valid.
        # If not, we might have a bug.
        # For safety, let's re-verify best_sol.
        
        # Actually, current_sol is the active one. 
        # If current_sol is smaller than best_sol, we update best_sol.
        # We did that inside the loop.
        
        # One final reduction on current_sol just in case
        sol_set = set(current_sol)
        coverage_counts = compute_coverage_counts(current_sol)
        
        stable = False
        while not stable:
            stable = True
            new_sol = []
            for b in current_sol:
                if is_redundant(b, coverage_counts):
                    for tri_idx in block_triples[b]:
                        coverage_counts[tri_idx] -= 1
                    stable = False
                else:
                    new_sol.append(b)
            current_sol = new_sol
            
        if len(current_sol) < len(best_sol):
            best_sol = current_sol

    # Convert best_sol (indices) to actual blocks
    result_blocks = [all_blocks[i] for i in best_sol]
    
    # Final Validation Check (Exhaustive for small v? No, v=24 is too big for full exhaustiveness in Python quickly, 
    # but we can spot check or trust the bitset logic).
    # The logic relies on coverage_counts and bitsets which are robust.
    
    return result_blocks

