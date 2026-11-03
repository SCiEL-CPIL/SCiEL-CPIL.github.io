import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # The task specifies q=3, n=8, radius=1.
    N = q ** n
    powers = [q ** i for i in range(n)]
    
    # Helper to decode an integer to a list of digits
    def decode(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    # Precompute neighbors for each point as a list of indices
    # For q=3, n=8: each point has 1 + n*(q-1) = 17 neighbors
    neighbor_lists = []
    for c in range(N):
        nbrs = [c]
        for i in range(n):
            base = powers[i]
            val = (c // base) % q
            for new_val in range(q):
                if new_val == val:
                    continue
                neighbor = c + (new_val - val) * base
                nbrs.append(neighbor)
        neighbor_lists.append(nbrs)
    
    # Also precompute bitmasks for greedy initialization
    coverage = [0] * N
    for c in range(N):
        bits = 0
        for nb in neighbor_lists[c]:
            bits |= (1 << nb)
        coverage[c] = bits
    
    full_mask = (1 << N) - 1
    
    best_code = None
    best_size = N + 1
    
    # === Greedy initialization ===
    def greedy_init(randomize=True):
        uncovered = full_mask
        selected = []
        while uncovered:
            best_c = -1
            best_gain = 0
            if randomize:
                # Sample a random subset for speed and diversity
                candidates = random.sample(range(N), min(500, N))
            else:
                candidates = range(N)
            for c in candidates:
                gain = (uncovered & coverage[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                elif gain == best_gain and best_gain > 0 and random.random() < 0.3:
                    best_c = c
            if best_gain == 0:
                break
            selected.append(best_c)
            uncovered &= ~coverage[best_c]
        return selected
    
    # === Coverage count based local search (simulated annealing style) ===
    def local_search(selected, max_time):
        """
        Use coverage counts for O(1) redundancy checks.
        Moves: remove redundant codeword, or swap (remove one, add another that helps).
        """
        if not selected:
            return selected
        
        # Build coverage counts
        cover_count = np.zeros(N, dtype=np.int32)
        for c in selected:
            for nb in neighbor_lists[c]:
                cover_count[nb] += 1
        
        uncovered = np.count_nonzero(cover_count == 0)
        
        if uncovered > 0:
            # Fill gaps with greedy
            selected = list(selected)
            while uncovered > 0:
                # Find uncovered points
                unc_idx = np.where(cover_count == 0)[0]
                # Greedily pick best codeword for uncovered
                best_c = -1
                best_gain = 0
                candidates = random.sample(range(N), min(300, N))
                for c in candidates:
                    gain = 0
                    for nb in neighbor_lists[c]:
                        if cover_count[nb] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                if best_gain == 0:
                    break
                selected.append(best_c)
                for nb in neighbor_lists[best_c]:
                    cover_count[nb] += 1
                uncovered = np.count_nonzero(cover_count == 0)
        
        selected = list(selected)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(selected):
                    continue
                c = selected[i]
                # Check if c is redundant (all its neighbors are covered by others)
                redundant = True
                for nb in neighbor_lists[c]:
                    if cover_count[nb] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    for nb in neighbor_lists[c]:
                        cover_count[nb] -= 1
                    selected.pop(i)
                    improved = True
                    break
            
            if not improved and time.time() < max_time:
                # Try swaps: remove a codeword, then add a better one
                # Focus on codewords that are "least useful" (most redundant neighbors)
                if len(selected) > 0:
                    # Score each codeword by how many of its neighbors it uniquely covers
                    scores = []
                    for idx, c in enumerate(selected):
                        unique = 0
                        for nb in neighbor_lists[c]:
                            if cover_count[nb] == 1:
                                unique += 1
                        scores.append((unique, idx))
                    scores.sort()
                    # Try removing the least useful codewords
                    for unique_count, idx in scores[:min(10, len(scores))]:
                        if unique_count > 0:
                            break
                        c = selected[idx]
                        # Remove c
                        for nb in neighbor_lists[c]:
                            cover_count[nb] -= 1
                        # Check if any point becomes uncovered
                        newly_uncovered = 0
                        for nb in neighbor_lists[c]:
                            if cover_count[nb] == 0:
                                newly_uncovered += 1
                        if newly_uncovered == 0:
                            # Safe removal
                            selected.pop(idx)
                            improved = True
                            break
                        else:
                            # Need to add a codeword to cover the gaps
                            # Find best codeword to cover newly uncovered points
                            best_add = -1
                            best_gain = 0
                            unc_pts = [nb for nb in neighbor_lists[c] if cover_count[nb] == 0]
                            candidates = random.sample(range(N), min(200, N))
                            for add_c in candidates:
                                gain = 0
                                for nb in neighbor_lists[add_c]:
                                    if cover_count[nb] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_add = add_c
                            if best_add >= 0 and best_gain >= newly_uncovered:
                                # Do the swap
                                selected.pop(idx)
                                selected.append(best_add)
                                for nb in neighbor_lists[best_add]:
                                    cover_count[nb] += 1
                                improved = True
                                break
                            else:
                                # Revert
                                for nb in neighbor_lists[c]:
                                    cover_count[nb] += 1
        
        return selected
    
    # === Main search loop ===
    num_restarts = 0
    while time.time() < deadline - 15:
        num_restarts += 1
        
        # Randomized greedy initialization
        selected = greedy_init(randomize=True)
        
        if len(selected) >= best_size:
            continue
        
        # Local search with time budget
        time_budget = min(20, deadline - 15 - time.time())
        if time_budget < 2:
            break
        selected = local_search(selected, start_time + time.time() - start_time + time_budget)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
    
    # Final polish on best code
    if best_code is not None and time.time() < deadline - 3:
        selected = best_code[:]
        final_budget = min(10, deadline - 3 - time.time())
        if final_budget > 1:
            selected = local_search(selected, start_time + time.time() - start_time + final_budget)
            if len(selected) < best_size:
                best_size = len(selected)
                best_code = selected[:]
    
    # Fallback
    if best_code is None:
        best_code = list(range(N))
    
    result = []
    for c in best_code:
        result.append(decode(c))
        
    return result
