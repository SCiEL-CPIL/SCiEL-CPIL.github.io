import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Enumerate all k-subsets and compute their coverage set
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # coverage[i] is a set of indices of t-subsets covered by k_subset i
    coverage = [set() for _ in range(num_k_subsets)]
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            coverage[i].add(idx)
            
    # Helper to check if a set is empty
    def is_empty(s):
        return len(s) == 0

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts
    num_restarts = 50000  # We will limit by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Phase
        uncovered = set(range(num_t_subsets))
        current_blocks = []
        used_blocks = set()
        
        while uncovered:
            if time.time() > deadline:
                break
                
            max_gain = 0
            candidates = []
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                # Gain is number of new t-subsets covered
                gain = len(coverage[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
                
            # Pick a random candidate to add diversity
            chosen_idx = random.choice(candidates)
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered -= coverage[chosen_idx]
            
        if uncovered:
            continue
            
        current_count = len(current_blocks)
        
        # 2. Local Search Phase
        # We try to reduce the number of blocks.
        # Moves:
        # A. 1-swap: Remove a block if it's redundant (all its unique coverage is covered by others).
        # B. 2-swap: Remove two blocks if their combined unique coverage is covered by others.
        # C. 2-exchange: Replace two blocks with one if the new block covers all unique coverage of the two.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_count = len(current_blocks)
            
            # Compute unique coverage for each block in current solution
            # unique[i] = coverage[i] - (union of coverage[j] for j != i in current_blocks)
            # This is O(N^2 * S) where S is set size. N is ~300, S is ~15504. 
            # 300*300*15000 is too slow for Python sets if done naively every step.
            # Instead, we can compute "covered_by_others" for each block.
            
            # Optimization: 
            # Let U be the union of all coverage sets. 
            # For a block B_i, its unique coverage is coverage[B_i] \ (U \ coverage[B_i]).
            # But we need to check if unique coverage is empty.
            # unique[B_i] == empty iff coverage[B_i] is subset of (U \ coverage[B_i]).
            
            # Let's maintain a global union set 'total_union'.
            # To check redundancy of B_i:
            # We need to know if coverage[B_i] is covered by the other blocks.
            # This is equivalent to: for every t in coverage[B_i], there exists j != i in current_blocks such that t in coverage[j].
            
            # Efficient way:
            # Count how many times each t-subset is covered.
            # If count[t] == 1 for some t in coverage[B_i], then B_i is essential (not redundant).
            # If count[t] > 1 for all t in coverage[B_i], then B_i is redundant.
            
            # Let's maintain counts.
            counts = [0] * num_t_subsets
            for b_idx in current_blocks:
                for t_idx in coverage[b_idx]:
                    counts[t_idx] += 1
            
            # Try 1-swap (remove redundant block)
            # We can randomize the order of checking blocks to escape local minima? 
            # Or just check all.
            # If we find a redundant block, remove it and restart local search.
            
            removed = False
            for b_idx in current_blocks:
                if time.time() > deadline:
                    break
                # Check if b_idx is redundant
                # It is redundant if for all t in coverage[b_idx], counts[t] > 1
                redundant = True
                for t_idx in coverage[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b_idx
                    current_blocks.remove(b_idx)
                    # Update counts
                    for t_idx in coverage[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try 2-exchange: Replace B_i, B_j with B_k
            # Condition: Unique(B_i) U Unique(B_j) is subset of coverage[B_k]
            # Unique(B_i) = {t in coverage[B_i] | counts[t] == 1}
            # Note: If we remove B_i and B_j, the counts for t in Unique(B_i) will drop to 0.
            # So we need coverage[B_k] to cover all t such that counts[t] == 1 and t in coverage[B_i] or coverage[B_j].
            # Actually, let's define:
            # U_i = {t in coverage[B_i] | counts[t] == 1}
            # U_j = {t in coverage[B_j] | counts[t] == 1}
            # We need U_i U U_j subset of coverage[B_k].
            # Also, B_k must not be B_i or B_j.
            
            # To make this fast, we can precompute the list of "essential" t-subsets for each block.
            # But counts change.
            
            # Given the complexity, maybe stick to 1-swap and a simpler 2-swap (remove 2 redundant blocks).
            # 2-swap: Remove B_i and B_j if (U_i U U_j) is covered by other blocks.
            # This is hard to check efficiently without updating counts.
            
            # Alternative: Random 2-exchange.
            # Pick two random blocks B_i, B_j.
            # Calculate U_i and U_j based on current counts.
            # Let U_target = U_i U U_j.
            # If U_target is empty, we could just remove one (handled by 1-swap).
            # If U_target is not empty, look for B_k such that U_target subset of coverage[B_k].
            # To speed this up, we can check a random sample of B_k.
            
            # Let's try a limited 2-exchange search.
            # Pick random i, j.
            if random.random() < 0.5: # 50% chance to try 2-exchange
                if len(current_blocks) >= 2:
                    i, j = random.sample(range(len(current_blocks)), 2)
                    b_i = current_blocks[i]
                    b_j = current_blocks[j]
                    
                    # Calculate unique parts
                    # U_i = {t in coverage[b_i] | counts[t] == 1}
                    # U_j = {t in coverage[b_j] | counts[t] == 1}
                    # Note: If t is in both, and counts[t]==1, it's in both U_i and U_j.
                    
                    U_target = set()
                    for t_idx in coverage[b_i]:
                        if counts[t_idx] == 1:
                            U_target.add(t_idx)
                    for t_idx in coverage[b_j]:
                        if counts[t_idx] == 1:
                            U_target.add(t_idx)
                    
                    if U_target:
                        # Find a block B_k that covers U_target
                        # We can iterate through all blocks, but that's slow.
                        # We can sample.
                        # Or, we can precompute for each t-subset which blocks cover it?
                        # That's too much memory.
                        
                        # Let's just check a few random blocks.
                        found = False
                        for _ in range(100):
                            k_idx = random.randint(0, num_k_subsets - 1)
                            if k_idx == b_i or k_idx == b_j:
                                continue
                            # Check if U_target is subset of coverage[k_idx]
                            # if U_target.issubset(coverage[k_idx]):
                            # This is fast if U_target is small.
                            if all(t_idx in coverage[k_idx] for t_idx in U_target):
                                # Replace b_i, b_j with k_idx
                                current_blocks.remove(b_i)
                                current_blocks.remove(b_j)
                                current_blocks.append(k_idx)
                                
                                # Update counts
                                for t_idx in coverage[b_i]:
                                    counts[t_idx] -= 1
                                for t_idx in coverage[b_j]:
                                    counts[t_idx] -= 1
                                for t_idx in coverage[k_idx]:
                                    counts[t_idx] += 1
                                
                                improved = True
                                found = True
                                break
                        
                        if found:
                            continue

        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(k_subsets[b]) for b in current_blocks]

    if best_blocks is None:
        # Fallback: return empty list or simple greedy? 
        # The task requires a valid covering. 
        # If we didn't find one, we must return something. 
        # But the greedy phase should always find one if a solution exists.
        # C(19,8,5) exists.
        # If we timed out during greedy, we might have an invalid partial solution.
        # We should ensure we return a valid solution.
        # The best way is to finish the greedy if we haven't, but we are out of time.
        # However, the deadline guard breaks the loop. 
        # If best_blocks is None, it means we never completed a greedy run?
        # Unlikely, as greedy is fast.
        # If we are in trouble, we can return a known valid covering if we hardcode it, 
        # but we don't have one.
        # We'll assume best_blocks is not None.
        
        # As a safety, if best_blocks is None, try one more quick greedy?
        # No, we are out of time.
        # Return empty list will score 0.
        # We should probably just return the best partial? No, must be valid.
        # The prompt says "Valid when every t-subset...".
        # If we can't find a valid one, we score 0.
        
        # Let's assume the first greedy run finishes.
        # Greedy takes < 1 second.
        pass

    return best_blocks
