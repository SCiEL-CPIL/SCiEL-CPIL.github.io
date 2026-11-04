"""Lagrangian-relaxation guided greedy + local search for K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def _build_neighbors(q, n, radius):
    """Build neighbor lists (distance <= radius) for all q^n words as integers."""
    total = q ** n
    # Precompute base-q digits for all words
    digits_all = np.zeros((total, n), dtype=np.int8)
    for x in range(total):
        tmp = x
        for pos in range(n - 1, -1, -1):
            digits_all[x, pos] = tmp % q
            tmp //= q
    
    # For radius 1, neighbors are the word itself plus all single-position changes
    # Each word has 1 + n*(q-1) neighbors = 1 + 7*2 = 15 for q=3, n=7
    neighbors = []
    for x in range(total):
        covered = set()
        covered.add(x)
        d = digits_all[x]
        for pos in range(n):
            for val in range(q):
                if val == d[pos]:
                    continue
                # Compute new integer
                # x = sum(d[i] * q^(n-1-i))
                # Change position pos to val: new_x = x + (val - d[pos]) * q^(n-1-pos)
                power = q ** (n - 1 - pos)
                new_x = x + (val - d[pos]) * power
                covered.add(new_x)
        neighbors.append(covered)
    return neighbors

def _digits(x, n, q):
    """Convert integer to list of n digits in base q (most significant first)."""
    digits = []
    tmp = x
    for _ in range(n):
        digits.append(tmp % q)
        tmp //= q
    return digits[::-1]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    total = q ** n  # 2187
    
    # Build neighbor sets
    neighbors = _build_neighbors(q, n, radius)
    
    # Convert to numpy arrays for faster operations
    # neighbor_arrays[i] = numpy array of neighbors of word i
    neighbor_arrays = [np.array(sorted(nbrs), dtype=np.int32) for nbrs in neighbors]
    
    # Lagrangian relaxation approach:
    # Primal: min sum_c x_c s.t. for each point p, sum_{c: p in N(c)} x_c >= 1
    # Dual: max sum_p y_p s.t. for each c, sum_{p in N(c)} y_p <= 1, y_p >= 0
    # 
    # We'll use a greedy approach guided by dual variables:
    # 1. Initialize dual variables y_p
    # 2. Compute reduced cost for each candidate c: rc(c) = 1 - sum_{p in N(c)} y_p
    # 3. Pick the c with most negative reduced cost (or highest coverage gain)
    # 4. Update dual variables
    # 5. Local search refinement
    
    # Initialize with greedy set cover using coverage counts
    uncovered_mask = np.ones(total, dtype=bool)
    selected = []
    
    # Precompute coverage counts for all candidates
    # cover_count[c] = number of currently uncovered points in neighbors[c]
    cover_counts = np.zeros(total, dtype=np.int32)
    for c in range(total):
        cover_counts[c] = np.sum(uncovered_mask[neighbor_arrays[c]])
    
    # Greedy phase with time limit
    greedy_deadline = start_time + 40.0
    
    while np.any(uncovered_mask) and time.time() < greedy_deadline:
        # Find candidate with max coverage of uncovered points
        best_c = int(np.argmax(cover_counts))
        if cover_counts[best_c] == 0:
            break
        
        selected.append(best_c)
        
        # Update uncovered mask
        uncovered_mask[neighbor_arrays[best_c]] = False
        
        # Update cover counts for all other candidates
        # For each candidate c, recompute cover_counts[c]
        # This is expensive; instead, we can do incremental updates
        # But for simplicity, let's just recompute periodically or use a heuristic
        
        # Actually, let's use a more efficient approach:
        # Only recompute cover counts for candidates that might be affected
        # Since we're removing points, cover_counts only decrease
        # We can skip updating and just recompute the argmax from scratch each time
        # But that's O(total * avg_neighbors) per iteration
        
        # Better: maintain cover_counts and update them
        for c in range(total):
            if c == best_c:
                continue
            # cover_counts[c] should decrease by the number of newly covered points in N(c)
            # newly covered = neighbor_arrays[best_c] that were uncovered before
            # This is hard to track incrementally without extra bookkeeping
            # Let's just recompute all cover counts every few iterations
            pass
        
        # Recompute cover counts (expensive but correct)
        # Optimization: only recompute if we're making progress
        for c in range(total):
            if c in selected:
                cover_counts[c] = 0
            else:
                cover_counts[c] = np.sum(uncovered_mask[neighbor_arrays[c]])
    
    # If still not covered, add more
    while np.any(uncovered_mask) and time.time() < deadline - 10.0:
        uncovered_idx = np.where(uncovered_mask)[0]
        if len(uncovered_idx) == 0:
            break
        
        # For each uncovered point, find which candidates cover it
        # Pick the candidate that covers the most uncovered points
        best_c = -1
        best_gain = 0
        for c in range(total):
            if c in selected:
                continue
            gain = np.sum(uncovered_mask[neighbor_arrays[c]])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c == -1 or best_gain == 0:
            break
        
        selected.append(best_c)
        uncovered_mask[neighbor_arrays[best_c]] = False
    
    # Ensure full coverage
    final_covered = np.zeros(total, dtype=bool)
    for c in selected:
        final_covered[neighbor_arrays[c]] = True
    
    if not np.all(final_covered):
        # Force-add codewords for uncovered points
        for p in range(total):
            if not final_covered[p]:
                # Find a candidate that covers p
                for c in range(total):
                    if c in selected:
                        continue
                    if neighbor_arrays[c][0] == p or p in neighbor_arrays[c]:
                        selected.append(c)
                        final_covered[neighbor_arrays[c]] = True
                        break
    
    # Local search: try to remove codewords
    if time.time() < deadline - 10.0:
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            for i in range(len(selected)):
                if time.time() > deadline - 10.0:
                    break
                # Try removing selected[i]
                candidate = selected[:i] + selected[i+1:]
                # Check if still covered
                covered_check = np.zeros(total, dtype=bool)
                for c in candidate:
                    covered_check[neighbor_arrays[c]] = True
                if np.all(covered_check):
                    selected = candidate
                    changed = True
                    break  # Restart removal loop
    
    # 1-exchange moves: remove one, add another
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 3.0:
                    break
                
                # Remove selected[i], find best replacement
                sel_without = selected[:i] + selected[i+1:]
                covered_without = np.zeros(total, dtype=bool)
                for c in sel_without:
                    covered_without[neighbor_arrays[c]] = True
                
                # Find best replacement
                best_c = -1
                best_gain = 0
                for c in range(total):
                    if c in selected:
                        continue
                    gain = np.sum(~covered_without[neighbor_arrays[c]])
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                
                if best_c != -1 and best_gain > 0:
                    new_sel = sel_without + [best_c]
                    covered_new = np.zeros(total, dtype=bool)
                    for c in new_sel:
                        covered_new[neighbor_arrays[c]] = True
                    if np.all(covered_new):
                        selected = new_sel
                        improved = True
                        break  # Restart exchange loop
    
    # Convert to result format
    result = []
    seen = set()
    for c in selected:
        word = _digits(c, n, q)
        t = tuple(word)
        if t not in seen:
            seen.add(t)
            result.append(word)
    
    return result
# EVOLVE-BLOCK-END
