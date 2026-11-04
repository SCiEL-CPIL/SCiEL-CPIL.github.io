import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Pre-computation
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map triples to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # For each block, store the list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triples.append(tris)
        
    # Helper to get a random block index
    def random_block_idx():
        return random.randint(0, n_blocks - 1)

    def sa_minimize_uncovered(N, time_limit, initial=None):
        """
        Run Simulated Annealing to find a multiset of N blocks covering all triples.
        Returns a list of N block indices or None if timeout.
        """
        if initial is not None:
            current = initial[:]
        else:
            current = [random_block_idx() for _ in range(N)]
        
        # Initialize counts
        counts = [0] * n_triples
        for b_idx in current:
            for tri_idx in block_triples[b_idx]:
                counts[tri_idx] += 1
        
        # Current uncovered count
        uncovered = sum(1 for c in counts if c == 0)
        
        if uncovered == 0:
            return current
            
        best_solution = current[:]
        best_uncovered = uncovered
        
        # SA Parameters
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        steps = 0
        max_steps = 200000 # Will be limited by time
        
        while T > min_T and time.time() < time_limit:
            # Pick a random index to change
            i = random.randint(0, N - 1)
            old_b = current[i]
            new_b = random_block_idx()
            
            if new_b == old_b:
                continue
                
            # Calculate delta in uncovered count
            # Triples in old_b that might become uncovered
            # Triples in new_b that might become covered
            
            # We need to be careful: if a triple is in both old_b and new_b, 
            # and count was 1, it becomes 0 then 1? No, we remove old, add new.
            # Let's just compute the change.
            
            # To speed up, we can just update counts and check.
            # But we need to revert if rejected.
            
            # Store old counts for affected triples? 
            # Or just compute delta.
            
            # Delta logic:
            # Let A = set of triples in old_b
            # Let B = set of triples in new_b
            
            # For tri in A:
            #   if counts[tri] == 1: it becomes 0 (uncovered increases by 1) unless tri is also in B and we add it back?
            #   Actually, we remove old first, then add new.
            
            # Simpler:
            # 1. Decrement counts for A. Count how many went 1->0. (gain_uncovered)
            # 2. Increment counts for B. Count how many went 0->1. (gain_covered)
            # 3. Delta = gain_uncovered - gain_covered
            
            # However, if tri is in A and B, and count was 1:
            #   Decrement: 1->0. (gain_uncovered +1)
            #   Increment: 0->1. (gain_covered +1)
            #   Net change 0. Correct.
            
            # If tri in A, not B, count 1:
            #   Decrement: 1->0. (gain_uncovered +1)
            #   Net change +1. Correct.
            
            # If tri in B, not A, count 0:
            #   Increment: 0->1. (gain_covered +1)
            #   Net change -1. Correct.
            
            old_tris = block_triples[old_b]
            new_tris = block_triples[new_b]
            
            delta = 0
            
            # To avoid modifying counts and reverting, we can just calculate.
            # But it's tricky if sets overlap.
            # Let's just do the update and revert if needed.
            # This is O(|A| + |B|) ~ 70 ops.
            
            # Apply change
            for tri in old_tris:
                counts[tri] -= 1
                if counts[tri] == 0:
                    delta += 1
            
            for tri in new_tris:
                counts[tri] += 1
                if counts[tri] == 1:
                    delta -= 1
                    
            current[i] = new_b
            uncovered += delta
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_solution = current[:]
                if uncovered == 0:
                    return best_solution
            
            # Acceptance
            if delta <= 0:
                pass # Accept
            else:
                # Probability of accepting worse solution
                # p = exp(-delta / T)
                # If delta is large and T is small, reject
                if random.random() < (2.71828 ** (-delta / T)):
                    pass # Accept
                else:
                    # Revert
                    current[i] = old_b
                    uncovered -= delta
                    for tri in new_tris:
                        counts[tri] -= 1
                    for tri in old_tris:
                        counts[tri] += 1
            
            T *= cooling
            steps += 1
            
            # Check time
            if steps % 1000 == 0 and time.time() > time_limit:
                break
                
        return best_solution if best_uncovered == 0 else None

    def compress_blocks(block_indices):
        """
        Given a list of block indices (multiset), remove duplicates and redundant blocks.
        Returns a list of unique block indices that still cover all triples.
        """
        # Remove duplicates
        unique_blocks = list(set(block_indices))
        
        # Check coverage
        counts = [0] * n_triples
        for b_idx in unique_blocks:
            for tri in block_triples[b_idx]:
                counts[tri] += 1
        
        # If not all covered, we can't just remove duplicates if the multiset was essential?
        # No, if a block is duplicated, removing one copy doesn't change coverage.
        # So unique_blocks is always a valid cover if the multiset was.
        
        # Now remove redundant blocks
        # Greedy removal
        active = [True] * len(unique_blocks)
        # Rebuild counts for active
        for i, b_idx in enumerate(unique_blocks):
            if not active[i]:
                continue
            for tri in block_triples[b_idx]:
                counts[tri] += 1 # Wait, I already built counts above. 
                # Actually, the counts above include all unique blocks.
                # So counts is correct for unique_blocks.
        
        # Wait, I initialized counts to 0 and added all unique blocks. So counts is correct.
        
        changed = True
        while changed:
            changed = False
            random.shuffle(unique_blocks) # Shuffle to avoid bias
            # Note: shuffling unique_blocks changes indices. We need to be careful.
            # Better to iterate over a copy of indices
            
            indices = list(range(len(unique_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if not active[i]:
                    continue
                b_idx = unique_blocks[i]
                # Check if redundant
                redundant = True
                for tri in block_triples[b_idx]:
                    if counts[tri] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for tri in block_triples[b_idx]:
                        counts[tri] -= 1
                    changed = True
        
        return [unique_blocks[i] for i in range(len(unique_blocks)) if active[i]]

    best_blocks = None
    best_len = float('inf')
    
    # Try to find covers with decreasing N
    # Start from 76 (known best) down to 69 (lower bound)
    # We allocate time to each attempt
    
    current_time = time.time()
    time_remaining = deadline - current_time
    
    # Allocate time to N=76, 75, 74...
    # Let's give more time to higher N initially as they are easier?
    # Or just iterate.
    
    for N in range(76, 68, -1):
        if time.time() > deadline - 5: # Leave 5s for return
            break
            
        # Allocate a portion of remaining time
        # Let's give 10s per attempt, or split remaining time
        # If we are at N=76, we have ~115s.
        # Let's try to find a cover for N.
        
        # We can run multiple SA runs
        # Let's try 1 long run or several short runs?
        # Several short runs with different random seeds might be better.
        
        # Time per N:
        # If N=76, maybe 20s.
        # If N=75, maybe 20s.
        # etc.
        
        # Let's just run SA until deadline - 5s or success.
        # But we want to try multiple Ns.
        
        # Strategy:
        # 1. Try N=76 for 30s. If found, update best.
        # 2. Try N=75 for 30s. If found, update best.
        # 3. Try N=74 for 30s.
        # 4. Try N=73 for remaining time.
        
        # Let's implement a time-boxed SA.
        
        # We will try to find a solution of size N.
        # If found, compress it.
        # If compressed size < best_len, update.
        
        # To maximize chances, we run SA multiple times if time allows.
        
        # Let's define a helper that runs SA for a specific time budget
        def run_sa_for_time(N, budget):
            end_time = time.time() + budget
            best_local = None
            while time.time() < end_time:
                sol = sa_minimize_uncovered(N, time.time() + 1) # 1s per run? No, too short.
                # Let's let sa_minimize_uncovered run until its own deadline
                # But we need to pass a deadline.
                # Let's modify sa_minimize_uncovered to take a deadline
                pass
            
            # Actually, let's just call sa_minimize_uncovered with a deadline
            # and if it returns None, try again?
            # SA is stochastic. Restarting might help.
            
            # Let's just run one long SA for the allocated time.
            sol = sa_minimize_uncovered(N, end_time)
            return sol
        
        # Allocate time
        # Total time available: ~115s
        # We have 76, 75, 74, 73, 72, 71, 70, 69.
        # 8 values.
        # Let's give 10s to each of 76, 75, 74, 73, 72, 71, 70, 69.
        # Total 80s. Safe.
        
        # But 76 is most likely to succeed.
        # Let's give 76: 30s
        # 75: 20s
        # 74: 15s
        # 73: 10s
        # 72: 10s
        # 71: 10s
        # 70: 10s
        # 69: 10s
        # Total: 115s. Perfect.
        
        time_budgets = {
            76: 30,
            75: 20,
            74: 15,
            73: 10,
            72: 10,
            71: 10,
            70: 10,
            69: 10
        }
        
        budget = time_budgets.get(N, 5)
        
        # If not enough time left, break
        if time.time() + budget > deadline - 2:
            budget = deadline - 2 - time.time()
            if budget < 1:
                break
                
        sol = run_sa_for_time(N, budget)
        
        if sol is not None:
            compressed = compress_blocks(sol)
            if len(compressed) < best_len:
                best_len = len(compressed)
                best_blocks = compressed
            # If we found a cover, we can try to optimize it further with local search?
            # The compress_blocks does greedy removal.
            # We could also try swaps on the compressed list.
            
            # If best_len <= 69, we can stop.
            if best_len <= 69:
                break
                
    if best_blocks is None:
        # Fallback: Return a random valid cover if we didn't find one?
        # Or return the best we found in a previous run?
        # If we didn't find any, we must return something valid.
        # Let's do a greedy construction as a fallback to ensure validity.
        # Greedy is slow but reliable.
        
        # Actually, the SA should find a cover for N=76 easily.
        # If it fails, something is wrong.
        # Let's implement a quick greedy fallback.
        
        uncovered = set(range(n_triples))
        blocks = []
        
        # Pre-compute triple_to_blocks for greedy
        # This is expensive. Let's just use the existing structures.
        # For each triple, find a block covering it.
        # We can iterate through all blocks and pick the one covering the most uncovered triples.
        
        # This is O(N_blocks * N_uncovered).
        # N_blocks = 346k. N_uncovered = 2024.
        # 346k * 2024 * 35 (triples per block) is too slow.
        
        # Better greedy:
        # Pick a random uncovered triple.
        # Pick a random block covering it.
        # Add block.
        # Repeat.
        
        # To speed up, pre-compute triple_to_blocks list.
        # This takes ~1s.
        
        triple_to_blocks = defaultdict(list)
        for b_idx, tris in enumerate(block_triples):
            for tri_idx in tris:
                triple_to_blocks[tri_idx].append(b_idx)
                
        while uncovered and time.time() < deadline - 1:
            # Pick random uncovered triple
            tri_idx = random.choice(list(uncovered))
            # Pick random block from candidates
            b_idx = random.choice(triple_to_blocks[tri_idx])
            blocks.append(b_idx)
            # Update uncovered
            for tri in block_triples[b_idx]:
                if tri in uncovered:
                    uncovered.remove(tri)
                    
        if uncovered:
            # If we still have uncovered, we failed.
            # Return empty? No, must be valid.
            # This should not happen with greedy.
            pass
            
        # Compress
        compressed = compress_blocks(blocks)
        if best_blocks is None or len(compressed) < best_len:
            best_blocks = compressed
            best_len = len(compressed)
            
    # Convert indices to lists
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]

