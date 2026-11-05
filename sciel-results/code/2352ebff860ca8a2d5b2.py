import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    points = range(v)
    # Generate all possible k-blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices for fast lookup
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each block, the list of t-subset indices it covers
    # This is critical for speed
    block_to_tsubs = []
    for block in all_blocks:
        subs = list(itertools.combinations(block, t))
        idxs = [t_subset_to_idx[s] for s in subs]
        block_to_tsubs.append(idxs)
        
    # Convert to numpy arrays for faster indexing if possible, 
    # but lists of lists are okay if we are careful with loops.
    # Let's use a list of arrays for better memory locality and speed in summing.
    block_to_tsubs_arr = [np.array(idxs, dtype=np.int32) for idxs in block_to_tsubs]

    # State: 
    # x: boolean array of length num_blocks, True if block is in solution
    # cover_counts: array of length num_t_subsets, count of how many selected blocks cover this t-subset
    
    # Initialize: Start with a greedy solution to have a decent starting point, 
    # then use SA to refine. Starting from scratch is too slow to converge.
    
    # --- Phase 1: Fast Greedy Initialization ---
    # We need to quickly get a valid cover to start SA from.
    # Use a simple greedy: pick block covering most uncovered t-subsets.
    
    # For speed in greedy, we can maintain a count of uncovered t-subsets per block.
    # Initially all t-subsets are uncovered.
    # gain[i] = number of t-subsets in block i (since all are uncovered initially)
    # But as we add blocks, gains change.
    
    # To make greedy fast enough, we will just run a few iterations of greedy 
    # with random restarts or just one strong greedy pass.
    
    # Let's do a standard greedy set cover.
    # We need an efficient way to update gains.
    # Given the small size (C(19,7)=50388 blocks, C(19,4)=3876 t-subsets),
    # we can afford a slightly slower greedy if it's robust.
    
    # Let's use a priority queue or just max search.
    # Since num_blocks is ~50k, max search is O(N). Doing this ~150 times is ~7.5M ops. Acceptable.
    
    uncovered = np.ones(num_t_subsets, dtype=np.int32)
    selected = [False] * num_blocks
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    
    # Precompute reverse index? 
    # For greedy, we need to update gains of all blocks sharing a t-subset with the chosen block.
    # This requires knowing which blocks cover a given t-subset.
    # Let's build a list: tsub_to_blocks[j] = list of block indices covering t-subset j.
    tsub_to_blocks = [[] for _ in range(num_t_subsets)]
    for i, idxs in enumerate(block_to_tsubs_arr):
        for j in idxs:
            tsub_to_blocks[j].append(i)
    # Convert to arrays for speed
    tsub_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Initial gains
    gains = np.array([len(idxs) for idxs in block_to_tsubs_arr], dtype=np.int32)
    
    greedy_selected = []
    # Time budget for greedy: 10 seconds
    greedy_deadline = start_time + 10.0
    
    while time.time() < greedy_deadline:
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        b_idx = int(np.argmax(gains))
        if selected[b_idx]:
            # This shouldn't happen if we set gains to 0 for selected, 
            # but let's be safe.
            gains[b_idx] = 0
            continue
            
        selected[b_idx] = True
        greedy_selected.append(b_idx)
        
        # Update cover counts and gains
        covered_tsubs = block_to_tsubs_arr[b_idx]
        
        # For each t-subset covered by this block:
        for j in covered_tsubs:
            if uncovered[j] == 1:
                uncovered[j] = 0
                # Decrease gain of all blocks that cover this t-subset
                blocks_covering_j = tsub_to_blocks_arr[j]
                # Note: b_idx is in blocks_covering_j, so it will be decremented too.
                # Since b_idx is now selected, we don't care about its gain anymore, 
                # but it must not be picked again. We handle this by checking 'selected' 
                # or setting gain to 0 if selected.
                gains[blocks_covering_j] -= 1
                
                # Ensure selected blocks have 0 gain to prevent re-selection
                # (Though argmax might pick it if it's the only one with high gain? 
                # No, we set selected[b_idx]=True. We should check in the loop.)
                # Actually, simpler: just set gains[b_idx]=0 after selection.
                # But we already did that? No, we did it before update.
                # Let's explicitly set it.
            else:
                # Already covered, no change to uncovered status, but cover count increases
                pass
        
        cover_counts[covered_tsubs] += 1
        gains[b_idx] = 0 # Mark as selected
        
        if np.all(uncovered == 0):
            break
            
    # If greedy didn't finish in time, we still have a partial cover.
    # For SA, we need a valid cover to start from, or we handle invalid states with penalty.
    # Let's assume greedy gave a valid cover. If not, we'll try to fix it in SA 
    # by allowing moves that reduce uncovered count.
    
    # Check if valid
    if not np.all(cover_counts >= 1):
        # Fallback: Add random blocks until valid (should be fast)
        random.shuffle(all_blocks)
        for b_idx in all_blocks:
            if not selected[b_idx]:
                selected[b_idx] = True
                greedy_selected.append(b_idx)
                covered_tsubs = block_to_tsubs_arr[b_idx]
                cover_counts[covered_tsubs] += 1
                if np.all(cover_counts >= 1):
                    break

    # --- Phase 2: Simulated Annealing ---
    # Objective: Minimize number of blocks (len(selected))
    # Constraint: All t-subsets covered (cover_counts >= 1)
    
    # We represent the state by the set of selected blocks.
    # Moves:
    # 1. Remove a block: If it doesn't make any t-subset uncovered (cover_counts[j] > 1 for all j in block), accept.
    #    If it makes some uncovered, accept with probability exp(-DeltaE/T) where DeltaE is penalty.
    # 2. Add a block: If it covers new t-subsets (reduces uncovered count if we had any, or just adds redundancy), 
    #    it increases cost by 1. Accept with prob exp(-1/T).
    # 3. Swap a block: Remove one, add another.
    
    # To make SA effective, we should focus on moves that might reduce the count.
    # Removing a redundant block is a "free" improvement.
    # Removing a non-redundant block is "bad" unless it allows removing another block later (complex).
    
    # Strategy:
    # Start with the greedy solution.
    # Temperature T starts high, cools down.
    # At each step:
    #   Pick a random block in solution to try removing.
    #   If removal is safe (no uncovered t-subsets created), do it. (This is a hill climb step).
    #   If removal is unsafe, calculate "damage" = number of t-subsets that become uncovered.
    #   Accept with probability exp(-damage / T).
    #   Also, occasionally try adding a random block not in solution? 
    #   Adding blocks increases cost, so we only do this if it helps remove more blocks later.
    #   However, pure removal-based SA (stochastic local search for minimum vertex cover-like problems)
    #   often gets stuck.
    
    # Better Move Set:
    # 1. Try to remove a random block from the current solution.
    #    - If valid, accept.
    #    - If invalid, accept with prob exp(-num_uncovered_created / T).
    # 2. Try to swap a block in the solution with a block not in the solution.
    #    - Remove B1, Add B2.
    #    - Calculate change in uncovered count and change in size (0).
    #    - If it reduces uncovered count or keeps it 0, accept.
    #    - If it increases uncovered count, accept with prob.
    
    # Let's implement a hybrid: 
    # Most steps: Try removing a random block.
    # Some steps: Try swapping.
    
    current_solution = [b for b in range(num_blocks) if selected[b]]
    current_size = len(current_solution)
    best_solution = list(current_solution)
    best_size = current_size
    
    # Update cover_counts to match current_solution
    # (It should already be correct from greedy, but let's be sure)
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b in current_solution:
        cover_counts[block_to_tsubs_arr[b]] += 1
        
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    sa_deadline = deadline - 5.0
    
    # Precompute a list of all blocks for random selection
    all_block_indices = np.arange(num_blocks)
    
    # For faster "is block i in solution" check, use a boolean array
    in_solution = np.zeros(num_blocks, dtype=bool)
    for b in current_solution:
        in_solution[b] = True
        
    # For random selection of "not in solution", we can just pick random indices and check.
    # Or maintain a list of candidates. Given 50k blocks, checking is fast.
    
    while time.time() < sa_deadline:
        if T < T_min:
            T = T_min
            
        # Choose move type
        # 80% removal attempt, 20% swap attempt
        if random.random() < 0.8:
            # Try removing a random block from current_solution
            if current_size == 0:
                break
            idx_in_sol = random.randint(0, current_size - 1)
            b_remove = current_solution[idx_in_sol]
            
            # Calculate impact
            tsubs_b = block_to_tsubs_arr[b_remove]
            # Number of t-subsets that would become uncovered
            # These are t-subsets j where cover_counts[j] == 1
            # and b_remove is the only one covering it.
            # Since b_remove is in solution, cover_counts[j] >= 1.
            # If cover_counts[j] == 1, removing b_remove makes it 0.
            num_uncovered_would_be = np.sum(cover_counts[tsubs_b] == 1)
            
            if num_uncovered_would_be == 0:
                # Safe removal
                # Perform removal
                cover_counts[tsubs_b] -= 1
                current_solution.pop(idx_in_sol)
                current_size -= 1
                in_solution[b_remove] = False
                
                # Update best
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_solution)
            else:
                # Unsafe removal
                # Accept with probability exp(-num_uncovered_would_be / T)
                prob = np.exp(-num_uncovered_would_be / T)
                if random.random() < prob:
                    # Perform removal (state becomes invalid)
                    cover_counts[tsubs_b] -= 1
                    current_solution.pop(idx_in_sol)
                    current_size -= 1
                    in_solution[b_remove] = False
                    # Note: We don't update best here because it's invalid.
                    # We rely on subsequent moves to fix it or cool down.
        else:
            # Try swap
            if current_size == 0:
                # Add a random block
                b_add = int(random.choice(all_block_indices))
                if not in_solution[b_add]:
                    current_solution.append(b_add)
                    current_size += 1
                    in_solution[b_add] = True
                    cover_counts[block_to_tsubs_arr[b_add]] += 1
            else:
                # Pick block to remove
                idx_in_sol = random.randint(0, current_size - 1)
                b_remove = current_solution[idx_in_sol]
                # Pick block to add (random from all)
                b_add = int(random.choice(all_block_indices))
                if b_add == b_remove or in_solution[b_add]:
                    continue
                    
                tsubs_remove = block_to_tsubs_arr[b_remove]
                tsubs_add = block_to_tsubs_arr[b_add]
                
                # Calculate change in uncovered count
                # This is complex. 
                # Let's approximate: 
                # Delta = (new_uncovered) - (old_uncovered)
                # Old uncovered: cover_counts[j] == 1 for j in tsubs_remove (potentially)
                # New state: 
                #   cover_counts_new[j] = cover_counts_old[j] - (1 if j in tsubs_remove else 0) + (1 if j in tsubs_add else 0)
                
                # We want to minimize uncovered count.
                # Let's just calculate the new uncovered count for the affected t-subsets.
                # Affected t-subsets are union of tsubs_remove and tsubs_add.
                
                # For efficiency, we can just do the swap and check, then revert if bad?
                # But we need to decide BEFORE swapping based on energy change.
                
                # Let's compute energy change:
                # E = num_uncovered + lambda * size
                # Here size is constant (swap), so E change = change in num_uncovered.
                
                # num_uncovered_old = sum(1 for j in tsubs_remove if cover_counts[j] == 1) 
                #                    + sum(1 for j in tsubs_add if j not in tsubs_remove and cover_counts[j] == 0)
                # Wait, if j is in tsubs_add and not in tsubs_remove, cover_counts[j] >= 0.
                # If cover_counts[j] == 0, it's currently uncovered.
                
                # Let's use a set for tsubs_add for fast lookup? 
                # tsubs_add is an array. Converting to set is O(K).
                # K is C(7,4)=35. Small.
                set_add = set(tsubs_add.tolist())
                set_remove = set(tsubs_remove.tolist())
                
                # Affected indices
                # We need to check cover_counts for these indices.
                
                # Change in uncovered count:
                # For j in tsubs_remove:
                #   if cover_counts[j] == 1: it becomes 0 (if not in add) -> +1 uncovered
                #   if cover_counts[j] == 2: it becomes 1 (if not in add) -> no change
                #   if j in add: 
                #     new_count = old_count - 1 + 1 = old_count. No change in coverage status.
                
                # For j in tsubs_add (not in remove):
                #   if cover_counts[j] == 0: it becomes 1 -> -1 uncovered
                #   if cover_counts[j] > 0: no change
                
                delta_uncovered = 0
                
                # Process removals
                for j in tsubs_remove:
                    if j in set_add:
                        continue # Covered by add, no net change in count
                    if cover_counts[j] == 1:
                        delta_uncovered += 1
                    # else no change or decrease in uncovered? No, if count > 1, it stays covered.
                    
                # Process additions
                for j in tsubs_add:
                    if j in set_remove:
                        continue
                    if cover_counts[j] == 0:
                        delta_uncovered -= 1
                        
                if delta_uncovered <= 0:
                    # Improvement or neutral
                    # Perform swap
                    cover_counts[tsubs_remove] -= 1
                    cover_counts[tsubs_add] += 1
                    current_solution[idx_in_sol] = b_add
                    in_solution[b_remove] = False
                    in_solution[b_add] = True
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_solution = list(current_solution)
                else:
                    # Worse
                    prob = np.exp(-delta_uncovered / T)
                    if random.random() < prob:
                        cover_counts[tsubs_remove] -= 1
                        cover_counts[tsubs_add] += 1
                        current_solution[idx_in_sol] = b_add
                        in_solution[b_remove] = False
                        in_solution[b_add] = True
                        
        T *= cooling
        
    # --- Phase 3: Local Search Refinement (Deletion) ---
    # Ensure we end with a valid solution.
    # If current_solution is invalid, we need to fix it.
    # Best strategy: Start from best_solution (which is always valid? 
    # No, best_solution is updated only when current_size < best_size.
    # In the "unsafe removal" case, we did NOT update best_solution.
    # In the "swap" case, we did NOT update best_solution if delta_uncovered > 0.
    # So best_solution is always a valid cover?
    # Let's trace:
    # Initial: Valid.
    # Safe removal: Valid. Update best.
    # Unsafe removal: Invalid. Do NOT update best.
    # Swap (delta <= 0): Valid (or same validity). Update best if size smaller.
    # Swap (delta > 0): Invalid. Do NOT update best.
    
    # So best_solution should be valid.
    # However, if we did many unsafe moves, current_solution is invalid.
    # We should use best_solution as the final answer, but we can try to improve it 
    # by restarting SA from best_solution or doing a local search on it.
    
    # Let's do a final local search on best_solution to remove any redundancies.
    final_solution = list(best_solution)
    final_size = len(final_solution)
    
    # Rebuild cover counts for final_solution
    cover_counts_final = np.zeros(num_t_subsets, dtype=np.int32)
    for b in final_solution:
        cover_counts_final[block_to_tsubs_arr[b]] += 1
        
    # Greedy deletion
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(final_solution):
            tsubs = block_to_tsubs_arr[b_idx]
            if np.all(cover_counts_final[tsubs] >= 2):
                to_remove.append(i)
                
        if to_remove:
            for i in to_remove:
                b_idx = final_solution[i]
                cover_counts_final[block_to_tsubs_arr[b_idx]] -= 1
            remove_set = set(to_remove)
            final_solution = [b for i, b in enumerate(final_solution) if i not in remove_set]
            changed = True
            
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in final_solution]
    return blocks

# Test/Verification helper (not used in construct)
def verify(v, k, t, blocks):
    points = range(v)
    all_t_subsets = set(itertools.combinations(points, t))
    covered = set()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            covered.add(sub)
    return all_t_subsets.issubset(covered)

if __name__ == "__main__":
    # Quick test
    v, k, t = 19, 7, 4
    blocks = construct(v, k, t)
    print(f"Number of blocks: {len(blocks)}")
    if verify(v, k, t, blocks):
        print("Valid cover")
    else:
        print("INVALID COVER")
