import time
import random

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 55  # 55s safety margin out of 90s
    
    def get_consecutive_sums(seq):
        """
        Returns a set of all non-empty consecutive block sums for the given sequence.
        """
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        """
        Checks if all non-empty consecutive block sums are distinct.
        """
        if not seq:
            return True
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_backward():
        """
        Start with all numbers 1..N.
        While invalid, remove an element.
        Heuristic: Remove the element that, if removed, results in the fewest 'conflicts'
        or simply remove the element that is 'most redundant'.
        
        A simpler heuristic: 
        1. Find all duplicate sums.
        2. For each duplicate sum, identify the pairs of blocks that create it.
        3. Any element in the symmetric difference of these blocks is a candidate for removal.
        4. Pick the candidate that minimizes the new number of conflicts.
        
        However, computing this is expensive. 
        Alternative heuristic:
        Start with [1..N].
        While invalid:
            Find a pair of blocks with the same sum.
            The union of indices in these two blocks must be broken.
            Removing any element in the symmetric difference of the two blocks will break this specific conflict.
            Among all such candidate elements across all current conflicts, pick one to remove.
            To be efficient, we can just pick the first conflict found and remove an element from its symmetric difference.
            To improve, we can prioritize removing elements that appear in many conflicts.
        """
        current_seq = list(range(1, N + 1))
        
        # Precompute prefix sums to quickly get block sums
        # But since we remove elements, indices shift. 
        # It's easier to just recompute sums from scratch in is_valid or a helper.
        
        # To make this fast, we need a way to find conflicts quickly.
        # Let's try a different backward heuristic:
        # Sort elements by some 'cost' and remove the 'costliest' one if invalid.
        # Actually, let's try a randomized backward removal.
        
        # Better Backward Heuristic:
        # 1. Check if valid. If yes, return.
        # 2. Find all duplicate sums.
        # 3. For each duplicate sum, we have at least two blocks (i1, j1) and (i2, j2) with same sum.
        #    Removing any element in the set of indices involved in these blocks (specifically, the symmetric difference)
        #    will resolve this specific conflict.
        # 4. Count how many conflicts each element is involved in.
        # 5. Remove the element involved in the most conflicts.
        
        while True:
            if time.time() > deadline:
                return current_seq
            
            # Check validity and collect conflicts
            n = len(current_seq)
            sums = {}
            conflicts = []
            
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += current_seq[j]
                    if s in sums:
                        # Conflict found between (i, j) and (sums[s][0], sums[s][1])
                        conflicts.append((i, j, sums[s][0], sums[s][1]))
                    else:
                        sums[s] = (i, j)
                        
            if not conflicts:
                return current_seq
            
            # Count involvement of each index in conflicts
            # An index k is involved if it is in the symmetric difference of the two blocks.
            # Block 1: [i1, j1], Block 2: [i2, j2]
            # Symmetric difference indices are those in one but not the other.
            # Removing any such index breaks this conflict.
            
            involvement = [0] * n
            
            for c in conflicts:
                i1, j1, i2, j2 = c
                # Indices in block 1: i1..j1
                # Indices in block 2: i2..j2
                # We want to mark indices that are in block1 XOR block2.
                # This is equivalent to:
                # If i1 <= i2:
                #   If j1 < i2: disjoint. All indices in both blocks are candidates? 
                #   Wait. If blocks are disjoint, removing an element from block 1 changes its sum. 
                #   Removing an element from block 2 changes its sum.
                #   So ANY element in block 1 or block 2 is a candidate to break the equality.
                #   Actually, if we remove an element from block 1, block 1's sum changes, so it no longer equals block 2's sum.
                #   So all elements in the union of the two blocks are candidates.
                
                # Let's just mark all elements in the union of the two blocks as "involved".
                # This is a safe over-approximation.
                
                min_idx = min(i1, i2)
                max_idx = max(j1, j2)
                for k in range(min_idx, max_idx + 1):
                    involvement[k] += 1
                    
            # Find the index with maximum involvement
            max_inv = max(involvement)
            # Pick the first one with max involvement (or random among them for diversity)
            candidates = [k for k in range(n) if involvement[k] == max_inv]
            k_to_remove = random.choice(candidates)
            
            # Remove element at k_to_remove
            current_seq.pop(k_to_remove)
            
        return current_seq

    # Strategy:
    # 1. Run a forward greedy to get a baseline.
    # 2. Run multiple iterations of backward greedy with randomization.
    # 3. Keep the best result.
    
    def greedy_forward():
        values = []
        sums = set()
        for x in range(1, N + 1):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best_seq = greedy_forward()
    best_len = len(best_seq)
    
    # Try backward greedy multiple times
    # Note: Backward greedy on [1..256] starts with 256 elements.
    # Each step removes 1 element. 
    # The check is O(N^2). 
    # If we remove ~200 elements, that's 200 * 256^2 ~ 13M operations.
    # Python might be slow for this. 
    # Let's optimize the conflict finding.
    
    # Optimized Backward Greedy
    def greedy_backward_opt():
        current_seq = list(range(1, N + 1))
        
        while True:
            if time.time() > deadline:
                return current_seq
            
            n = len(current_seq)
            sums = {}
            conflicts = []
            
            # Compute sums and find conflicts
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += current_seq[j]
                    if s in sums:
                        conflicts.append((i, j, sums[s][0], sums[s][1]))
                    else:
                        sums[s] = (i, j)
                        
            if not conflicts:
                return current_seq
            
            # To speed up, we can just pick the first conflict and remove a random element from its union.
            # This is faster than counting involvement.
            c = random.choice(conflicts)
            i1, j1, i2, j2 = c
            min_idx = min(i1, i2)
            max_idx = max(j1, j2)
            
            # Pick a random index in the union
            k_to_remove = random.randint(min_idx, max_idx)
            current_seq.pop(k_to_remove)
            
        return current_seq

    # Run several iterations
    for _ in range(10):
        if time.time() > deadline:
            break
        seq = greedy_backward_opt()
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq

    return best_seq
