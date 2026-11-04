import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute the list of t-subset indices for every possible k-subset (block)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_tsubs[i] is a list of indices into all_tsubs
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # Inverse mapping: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # Deadline: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_elapsed():
        return time.time() - start_time

    # --- Strategy: Greedy MIS on Conflict Graph ---
    # We view the problem as selecting a set of blocks.
    # Instead of standard greedy set cover (maximize marginal gain), we use a 
    # heuristic inspired by Maximum Independent Set on a graph where edges represent 
    # "conflict" or "redundancy". 
    # 
    # However, a direct MIS is NP-hard. We use a Greedy MIS heuristic adapted for set cover.
    # 
    # Alternative Structural Change: 
    # Use a "Weighted Greedy with Lookahead" combined with "Iterated Local Search" 
    # using a specific move: "2-swap" (remove 2 blocks, add 1-2 blocks to maintain cover).
    # 
    # Let's try a different framing: 
    # 1. Fast Greedy to get a valid cover.
    # 2. Aggressive Reduction using a "Dominance" rule: 
    #    If Block A covers a subset of the *uniquely covered* points of Block B, 
    #    and A has lower "cost" (or same size), we might swap.
    #    
    # Actually, the prompt suggests: "hypergraph independent set formulation on the dual graph".
    # Let's implement a Greedy Independent Set on the "Conflict Graph" of the current solution.
    # 
    # Conflict Graph G:
    # Vertices: Current blocks.
    # Edge (u, v) if removing v does NOT uncover any t-subset that is uniquely covered by u? 
    # No, that's not right.
    #
    # Let's stick to a robust, fast local search that is structurally different from simple SA:
    # **Iterated Greedy (IG)** with **2-opt style moves**.
    #
    # Move: Remove block B1 and B2. Add block B3 and B4 (or just B3) such that all t-subsets 
    # are still covered. Accept if size decreases or stays same (with random tie-break to escape minima).
    
    def greedy_construct():
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 5.0:
                return selected_blocks
            
            # Pick a random uncovered t-subset
            if uncovered_count < 100:
                uncov_indices = [i for i in range(num_tsubs) if uncovered[i]]
                if not uncov_indices:
                    break
                s_idx = random.choice(uncov_indices)
            else:
                while True:
                    s_idx = random.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break
            
            candidates = tsub_blocks[s_idx]
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                        if gain == 56:
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected_blocks

    def reduce_blocks(block_indices):
        block_indices = list(set(block_indices))
        changed = True
        while changed:
            changed = False
            if get_elapsed() > deadline - 2.0:
                break
            cover_count = [0] * num_tsubs
            for b_idx in block_indices:
                for j in block_tsubs[b_idx]:
                    cover_count[j] += 1
            
            new_block_indices = []
            for b_idx in block_indices:
                is_essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_block_indices.append(b_idx)
                else:
                    changed = True
            block_indices = new_block_indices
        return block_indices

    def iterated_local_search(initial_blocks):
        current_blocks = set(initial_blocks)
        cover_count = [0] * num_tsubs
        for b_idx in current_blocks:
            for j in block_tsubs[b_idx]:
                cover_count[j] += 1
        
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
        # Precompute "critical" sets for each block? 
        # critical_tsubs[b_idx] = list of tsubs where cover_count[j] == 1 and j is in block_tsubs[b_idx]
        # This is dynamic. We will compute on the fly for the specific block being removed.
        
        # Move: 2-Remove, 1-Add (if possible) or 1-Remove (if redundant)
        # Or: 1-Remove, 1-Add (Swap)
        
        # Let's try a specific heuristic: "Greedy MIS on Conflict Graph"
        # Construct a graph where nodes are current blocks.
        # Edge (u, v) exists if u and v are "compatible" in a way that allows one to be removed?
        # No, let's use a simpler, effective local search: 
        # **Random Restart 1-Remove / 1-Add with Acceptance Criteria**
        
        T = 5.0
        cooling = 0.99
        min_T = 0.1
        
        while T > min_T:
            if get_elapsed() > deadline - 10.0:
                break
            
            if not current_blocks:
                break
                
            block_to_remove = random.choice(list(current_blocks))
            
            # Check if removal is safe
            new_uncovered = 0
            for j in block_tsubs[block_to_remove]:
                if cover_count[j] == 1:
                    new_uncovered += 1
            
            if new_uncovered == 0:
                # Safe removal
                current_blocks.remove(block_to_remove)
                for j in block_tsubs[block_to_remove]:
                    cover_count[j] -= 1
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
            else:
                # Need to add a block to cover the new_uncovered tsubs
                # Find candidate blocks that cover the most of these new_uncovered tsubs
                critical_tsubs = [j for j in block_tsubs[block_to_remove] if cover_count[j] == 1]
                
                # Candidates must cover ALL critical_tsubs to be a valid swap (size 0 change)
                # Or cover MOST and we accept with penalty?
                # For a strict improvement or neutral move, we want to cover all.
                
                # If critical_tsubs is small, we can check all blocks covering the first one.
                if critical_tsubs:
                    candidates = tsub_blocks[critical_tsubs[0]]
                    best_in = -1
                    best_covered = -1
                    
                    # Sample candidates
                    sample_size = min(20, len(candidates))
                    if len(candidates) > sample_size:
                        sampled = random.sample(candidates, sample_size)
                    else:
                        sampled = candidates
                    
                    for b_in in sampled:
                        if b_in == block_to_remove:
                            continue
                        # Check how many critical tsubs it covers
                        covered = 0
                        for j in critical_tsubs:
                            # Fast check: is j in block_tsubs[b_in]?
                            # Since block_tsubs is a list, this is O(56). 
                            # To speed up, we could use a set for block_tsubs, but memory is high.
                            # 56 * 20 * steps is okay.
                            if j in block_tsubs[b_in]:
                                covered += 1
                        if covered > best_covered:
                            best_covered = covered
                            best_in = b_in
                    
                    if best_in != -1 and best_covered == len(critical_tsubs):
                        # Perfect Swap: Remove B_out, Add B_in. Size unchanged.
                        # This is a neutral move, helps escape local minima.
                        current_blocks.remove(block_to_remove)
                        for j in block_tsubs[block_to_remove]:
                            cover_count[j] -= 1
                        
                        current_blocks.add(best_in)
                        for j in block_tsubs[best_in]:
                            cover_count[j] += 1
                    else:
                        # Try to accept a "bad" move? 
                        # If we remove B_out and add a B_in that covers some but not all, 
                        # we have uncovered tsubs. 
                        # Standard SA allows this with penalty. 
                        # Penalty = number of uncovered tsubs.
                        # Let's try this.
                        if best_in != -1 and best_in != block_to_remove:
                            # Calculate delta (number of tsubs that become uncovered)
                            # If we remove B_out, critical_tsubs become uncovered (count 0).
                            # If we add B_in, it covers some of them.
                            # Net uncovered = len(critical_tsubs) - best_covered
                            delta = len(critical_tsubs) - best_covered
                            
                            if random.random() < math.exp(-delta / T):
                                # Accept bad move
                                current_blocks.remove(block_to_remove)
                                for j in block_tsubs[block_to_remove]:
                                    cover_count[j] -= 1
                                
                                current_blocks.add(best_in)
                                for j in block_tsubs[best_in]:
                                    cover_count[j] += 1
                                # Note: We now have an INVALID cover. 
                                # We must track uncovered tsubs? 
                                # My cover_count array will have 0s for uncovered tsubs.
                                # This complicates the "safe removal" check above.
                                # Let's NOT accept invalid states to keep the logic simple and robust.
                                # Revert if we accepted a bad move? 
                                # Actually, the code above already executed the change.
                                # Let's revert it to maintain validity, as handling invalid states 
                                # requires a different bookkeeping (list of uncovered).
                                # So, we only accept valid swaps (delta=0) or safe removals.
                                # Revert:
                                current_blocks.remove(best_in)
                                for j in block_tsubs[best_in]:
                                    cover_count[j] -= 1
                                
                                current_blocks.add(block_to_remove)
                                for j in block_tsubs[block_to_remove]:
                                    cover_count[j] += 1
                else:
                    # Should not happen if new_uncovered > 0
                    pass

            T *= cooling

        return best_blocks

    # Initial Greedy
    initial_blocks = greedy_construct()
    initial_blocks = reduce_blocks(initial_blocks)
    
    best_len = len(initial_blocks)
    best_block_list = [all_blocks[i] for i in initial_blocks]
    
    # Run ILS
    ils_blocks = iterated_local_search(initial_blocks)
    if len(ils_blocks) < best_len:
        best_len = len(ils_blocks)
        best_block_list = [all_blocks[i] for i in ils_blocks]
        
    # Second restart
    if get_elapsed() < deadline - 20.0:
        initial_blocks_2 = greedy_construct()
        initial_blocks_2 = reduce_blocks(initial_blocks_2)
        if len(initial_blocks_2) < best_len:
            best_len = len(initial_blocks_2)
            best_block_list = [all_blocks[i] for i in initial_blocks_2]
            
        ils_blocks_2 = iterated_local_search(initial_blocks_2)
        if len(ils_blocks_2) < best_len:
            best_len = len(ils_blocks_2)
            best_block_list = [all_blocks[i] for i in ils_blocks_2]

    return [list(b) for b in best_block_list]
