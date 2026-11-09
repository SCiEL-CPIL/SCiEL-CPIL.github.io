import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # 1. Precompute LCMs
    N = 128
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]

    # 2. Find all bad triples
    bad_triples = []
    triple_indices = [] # indices into bad_triples for each element
    elem_to_triples = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lab == lcm_table[a][c] == lcm_table[b][c]:
                    idx = len(bad_triples)
                    bad_triples.append((a, b, c))
                    elem_to_triples[a].append(idx)
                    elem_to_triples[b].append(idx)
                    elem_to_triples[c].append(idx)

    if not bad_triples:
        return list(range(1, N + 1))

    num_triples = len(bad_triples)
    
    def greedy_hitting_set(order=None):
        """
        Returns a hitting set R (list of elements) and the set of elements in R.
        """
        hits = [0] * num_triples
        in_R = [False] * (N + 1)
        R = []
        
        # If order is None, use greedy selection based on current uncovered count
        # We need to efficiently find the element covering the most uncovered triples.
        
        # For N=128, we can just iterate all elements each time.
        # Number of iterations is at most N.
        # Each iteration checks all elements and their triples.
        # Total cost: N * (N * avg_triples_per_elem).
        
        while True:
            # Check if all hit
            if all(h > 0 for h in hits):
                break
            
            # Find best element
            best_elem = -1
            best_count = -1
            
            for x in range(1, N + 1):
                if in_R[x]:
                    continue
                count = 0
                for t_idx in elem_to_triples[x]:
                    if hits[t_idx] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_elem = x
            
            if best_elem == -1:
                break # Should not happen if there are uncovered triples
            
            # Add best_elem to R
            in_R[best_elem] = True
            R.append(best_elem)
            for t_idx in elem_to_triples[best_elem]:
                hits[t_idx] += 1

        # Pruning phase
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of R to allow removal
            for x in R[:]:
                # Check if x is critical
                critical = False
                for t_idx in elem_to_triples[x]:
                    if hits[t_idx] == 1:
                        critical = True
                        break
                
                if not critical:
                    # Remove x from R
                    R.remove(x)
                    in_R[x] = False
                    for t_idx in elem_to_triples[x]:
                        hits[t_idx] -= 1
                    changed = True
                    # Restart pruning loop as removing one might make another removable
                    break
        
        return R, in_R

    best_R = None
    best_size = N + 1
    
    # Run multiple randomized restarts
    # The greedy choice can be randomized by perturbing the counts or shuffling ties.
    # Or simply shuffle the element order if we use a simpler greedy.
    # But the "most uncovered" greedy is strong.
    # We can randomize the tie-breaking.
    
    while time.time() < deadline:
        # Randomize tie-breaking by adding small random noise to counts
        # Or just run standard greedy, which is deterministic.
        # To explore different local minima, we can randomly select among top-k candidates.
        
        # Let's implement a randomized version of the selection step
        # We'll modify the inner loop slightly to allow randomness.
        
        # For simplicity and speed, let's just run the deterministic greedy first.
        # Then, if time permits, we can try to improve via swaps.
        
        R, in_R = greedy_hitting_set()
        
        if len(R) < best_size:
            best_size = len(R)
            best_R = R[:]
            
        # If we found a small set, we can try to improve it locally
        # by trying to swap elements out of R.
        # But the pruning step already does single-element removals.
        # Swaps are harder.
        
        # Let's try a simple local search on the removal set:
        # Try to remove x from R and add y to R if it reduces size? No, size is fixed in swap.
        # We want to reduce size.
        
        # If time is left, we can try to restart with different random seeds for tie-breaking.
        # We'll implement a randomized greedy by shuffling the elements when counts are equal.
        
        # Actually, the deterministic greedy is likely optimal or near-optimal for this structure.
        # Let's add a check: if best_size is very small, we can stop early?
        # No, keep searching.

        # To introduce randomness, we can shuffle the order of elements considered in the "best_elem" selection
        # if there are ties.
        
        # For now, let's just run the deterministic one. If it's fast, we can add randomization.
        
        # Let's add a randomized variant:
        # Instead of picking the absolute max, pick from top 5 with some probability.
        
        # Re-implement greedy with randomization option
        def greedy_randomized():
            hits = [0] * num_triples
            in_R = [False] * (N + 1)
            R = []
            
            while True:
                if all(h > 0 for h in hits):
                    break
                
                counts = {}
                for x in range(1, N + 1):
                    if in_R[x]:
                        continue
                    count = 0
                    for t_idx in elem_to_triples[x]:
                        if hits[t_idx] == 0:
                            count += 1
                    if count > 0:
                        counts[x] = count
                
                if not counts:
                    break
                
                max_count = max(counts.values())
                candidates = [x for x, c in counts.items() if c == max_count]
                
                # Randomly pick from top candidates
                x = random.choice(candidates)
                
                in_R[x] = True
                R.append(x)
                for t_idx in elem_to_triples[x]:
                    hits[t_idx] += 1
            
            # Pruning
            changed = True
            while changed:
                changed = False
                for x in R[:]:
                    critical = False
                    for t_idx in elem_to_triples[x]:
                        if hits[t_idx] == 1:
                            critical = True
                            break
                    if not critical:
                        R.remove(x)
                        in_R[x] = False
                        for t_idx in elem_to_triples[x]:
                            hits[t_idx] -= 1
                        changed = True
                        break
            return R, in_R

        R, in_R = greedy_randomized()
        if len(R) < best_size:
            best_size = len(R)
            best_R = R[:]

    # Construct final set
    if best_R is None:
        best_R = []
    
    removed_set = set(best_R)
    result = [x for x in range(1, N + 1) if x not in removed_set]
    
    return result
