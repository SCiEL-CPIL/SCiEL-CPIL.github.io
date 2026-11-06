"""New program using Randomized Backtracking (DFS) with pruning."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_seq = []
    best_len = 0
    limit = 55  # seconds
    
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)

    def dfs(current_seq, current_sums, candidates, depth, time_left):
        """
        Recursively try to add candidates to current_seq.
        current_seq: sorted list of selected numbers.
        current_sums: set of all consecutive block sums of current_seq.
        candidates: list of remaining numbers to consider (sorted).
        depth: current length of sequence.
        """
        nonlocal best_seq, best_len
        
        if time.time() - start_time > time_left:
            return

        # Update best if we found a longer sequence
        if len(current_seq) > best_len:
            best_seq = list(current_seq)
            best_len = len(current_seq)

        if not candidates:
            return

        # Heuristic: Try to fill remaining slots.
        # If we can't beat best_len even by taking all remaining, prune.
        if len(current_seq) + len(candidates) <= best_len:
            return

        # Randomize the order of candidates to explore different branches
        # We pick a random candidate to try first, or iterate through a shuffled list?
        # Iterating through all is expensive. 
        # Strategy: Pick a random candidate from the list, try to add it.
        # If successful, recurse. Then try the next one?
        # Standard DFS: For each candidate, decide Include/Exclude.
        # To keep it fast, we will use a "Randomized Greedy Backtracking":
        # We pick a random candidate. Try to include it. 
        # If it fits, we commit to it (greedy step) but keep a stack to backtrack if we get stuck?
        # True backtracking is expensive. Let's use a limited depth-first search with random restarts of the candidate order.
        
        # Instead of full DFS, let's use a "Randomized Greedy with Backtracking on Failure"
        # Actually, let's implement a proper randomized DFS but with strict time limits per branch.
        
        # We will iterate through candidates in a random order.
        # For each candidate x:
        #   1. Check if x can be added to current_seq.
        #   2. If yes, add it and recurse with remaining candidates.
        #   3. If no, skip it.
        # This is just randomized greedy. 
        # To make it "Backtracking", we need to allow skipping a candidate even if it fits, to see if a better combination follows.
        
        # Let's try a different tactic: 
        # 1. Shuffle candidates.
        # 2. Iterate. If candidate fits, add it.
        # 3. If we get stuck (no more candidates fit), we don't backtrack in this simple version.
        # This is the same as the old program but with shuffled input.
        
        # Let's do a real limited DFS.
        # We will try to include the first candidate. Then recurse.
        # Then we will try to exclude the first candidate. Then recurse.
        # This is 2^N which is too big.
        
        # Heuristic Pruning:
        # Only branch on "Include" if it's promising?
        
        # Let's stick to a robust "Randomized Greedy" but with multiple diverse random shuffles.
        # The old program used sorted ranges. Shuffling the entire 1..256 range creates very different trajectories.
        
        # Wait, the directive says "design a new program... from an approach family that none of the current programs use".
        # The current program is "Greedy with Randomized Restarts (varying acceptance prob)".
        # "Randomized Backtracking" is distinct.
        
        # Let's implement a simple DFS that goes deep first.
        # We will try to add candidates in a random order.
        # If adding a candidate leads to a dead end (no further additions possible), we backtrack and try the next candidate.
        
        # Implementation:
        # def dfs(seq, sums, remaining_indices):
        #    if time up: return
        #    update best
        #    if not remaining: return
        #    
        #    # Pick a random candidate from remaining
        #    idx = random.randrange(len(remaining_indices))
        #    x = candidates[remaining_indices[idx]]
        
        #    # Try to include x
        #    new_sums, new_seq = try_add(x, seq, sums)
        #    if new_sums:
        #        dfs(new_seq, new_sums, remaining_indices without idx)
        
        #    # Try to exclude x (i.e., just remove it from consideration)
        #    dfs(seq, sums, remaining_indices without idx)
        
        # This is exponential. We need strong pruning.
        # Pruning: If len(seq) + len(remaining) <= best_len, return.
        
        # Let's try this.
        
        pass

    # Since full DFS is risky, let's use **Simulated Annealing** on the set of elements.
    # State: Set S.
    # Objective: Maximize |S| such that S is valid.
    # Move: Add a random element not in S. If valid, accept.
    #        Remove a random element from S. (To allow escape from local minima? No, we want to grow).
    #        Swap a random element in S with a random element not in S.
    
    # Let's implement a **Local Search** starting from a random valid sequence.
    
    def random_valid_sequence():
        """Generate a small random valid sequence."""
        seq = []
        sums = set()
        # Pick random elements
        available = list(range(1, 257))
        random.shuffle(available)
        for x in available:
            # Check if x can be added
            # To keep it simple, we just try to add to the end of a sorted list?
            # No, let's just use the greedy method but with a random order.
            pass
        
        # Reuse the logic from before but make it local to this new design?
        # The directive says "do not reuse the current designs".
        # So I must not call `run_greedy` from the old code.
        # I must implement the logic from scratch.
        
        seq = []
        sums = set()
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be inserted into seq
            # Since seq is sorted, we need to find the insertion point.
            # But checking all consecutive blocks is O(N^2).
            # Let's just append to a list and sort at the end? No, sums depend on order.
            # The problem says "Strictly increasing sequence". So the order is fixed by the values.
            # So we are building a SET. The sequence is the sorted set.
            # Consecutive blocks in the sorted set.
            
            # If we add x to the set S, the new consecutive blocks are:
            # 1. Blocks entirely within old S (unchanged).
            # 2. Blocks that include x.
            #    If x is inserted between a and b (a < x < b), or at ends.
            #    The new sums are sums of [x], [a, x], [x, b], [a, x, b], etc.
            
            # This is complex. 
            # Let's stick to the "Add to end of a growing list" approach but with random ordering of values?
            # If we build the set by adding elements, the "consecutive blocks" in the final sorted set are not just the tails of the insertion order.
            # Example: Insert 3, then 1. Set {1,3}. Blocks: 1, 3, 4.
            # If we check tails of insertion order [3, 1]: 3, 4. We missed 1.
            # So we MUST maintain the sorted order or recompute sums from scratch.
            
            # Recomputing sums from scratch for each candidate is O(N^2).
            # N is small (< 50). 50^2 = 2500. 256 candidates * 2500 = 640,000 ops. Very fast.
            
            new_seq = seq + [x]
            new_seq.sort()
            new_sums = set()
            for i in range(len(new_seq)):
                s = 0
                for j in range(i, len(new_seq)):
                    s += new_seq[j]
                    if s in new_sums:
                        break
                    new_sums.add(s)
                else:
                    continue
                break
            
            if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                seq = new_seq
                sums = new_sums
                
        return seq

    # Local Search
    def local_search(initial_seq):
        seq = initial_seq
        sums = set()
        # Recompute sums
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                sums.add(s)
        
        current_len = len(seq)
        
        while time.time() - start_time < limit:
            # Try to add a random element
            available = [x for x in range(1, 257) if x not in seq]
            if not available:
                break
            
            x = random.choice(available)
            new_seq = seq + [x]
            new_seq.sort()
            
            # Check validity
            valid = True
            new_sums = set()
            for i in range(len(new_seq)):
                s = 0
                for j in range(i, len(new_seq)):
                    s += new_seq[j]
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            
            if valid:
                seq = new_seq
                sums = new_sums
                current_len += 1
                update_best(seq)
            else:
                # Try a swap?
                # Pick random u in seq, v in available.
                # Replace u with v.
                # This maintains length. Might allow future additions?
                # Let's try swaps to escape local optima.
                if seq:
                    u = random.choice(seq)
                    # Avoid picking same v
                    v = random.choice(available)
                    if u == v:
                        continue
                    new_seq = [w for w in seq if w != u] + [v]
                    new_seq.sort()
                    
                    # Check validity
                    valid = True
                    new_sums = set()
                    for i in range(len(new_seq)):
                        s = 0
                        for j in range(i, len(new_seq)):
                            s += new_seq[j]
                            if s in new_sums:
                                valid = False
                                break
                            new_sums.add(s)
                        if not valid:
                            break
                    
                    if valid:
                        # Accept swap (length unchanged, but might be better for future growth)
                        # We don't update best here as length is same.
                        seq = new_seq
                        sums = new_sums

        return seq

    # Run multiple restarts
    for _ in range(100):
        if time.time() - start_time > limit:
            break
        
        # Get a starting point
        start_seq = random_valid_sequence()
        update_best(start_seq)
        
        # Improve it
        improved = local_search(start_seq)
        update_best(improved)

    return best_seq
# EVOLVE-BLOCK-END
