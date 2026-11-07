import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 80  # Leave 10s margin for safety
    
    # 1. Build the full comparability matrix (4096x4096)
    # comp[i][j] = 1 if i and j are comparable (strict subset)
    # We can compute this efficiently using bit operations.
    # For each pair (i, j), i < j, check if (i & ~j) == 0 or (j & ~i) == 0
    # Since we need a symmetric matrix for clique search, we fill both [i][j] and [j][i]
    
    # Optimization: Precompute comparability for all pairs using vectorized ops if possible?
    # 4096^2 / 2 = ~8M pairs. In pure Python this might take a few seconds.
    # Let's try to do it efficiently.
    
    # Actually, we can just build the adjacency list or matrix on the fly during search?
    # No, branch-and-bound needs fast access. A 4096x4096 boolean matrix is 16MB.
    # Let's build it once.
    
    # To speed up construction, we can iterate i from 0 to 4095, and for each i,
    # find all j such that they are comparable.
    # But simple nested loop is 8M iterations. Each iteration is a few bit ops.
    # 8M * ~10 ops = 80M ops. In Python, ~10-20s. Might be too slow if we have other work.
    # Let's use numpy to vectorize the comparison.
    
    # Create array of all masks
    masks = np.arange(total_masks, dtype=np.int64)
    
    # We want a matrix C where C[i, j] = 1 if masks[i] and masks[j] are comparable.
    # Condition: (masks[i] & ~masks[j]) == 0 OR (masks[j] & ~masks[i]) == 0
    # This is hard to vectorize directly for all pairs without exploding memory (4096x4096 int64 is 128MB, which is fine).
    
    # Let's try a different approach:
    # For each mask i, the sets comparable to it are:
    # 1. All subsets of i (strict)
    # 2. All supersets of i (strict)
    
    # We can precompute for each mask i, the list of masks comparable to it?
    # Or just build the boolean matrix.
    
    # Let's try the numpy vectorization for the matrix construction.
    # i_idx = masks[:, None]
    # j_idx = masks[None, :]
    # comp = ((i_idx & ~j_idx) == 0) | ((j_idx & ~i_idx) == 0)
    # This creates two 4096x4096 boolean arrays. 4096*4096*2 bytes = 32MB each. Total 64MB.
    # This should be fast in numpy.
    
    i_idx = masks[:, None]
    j_idx = masks[None, :]
    
    # Note: ~j_idx on int64 works correctly for the lower 12 bits?
    # Yes, as long as we are only interested in the lower 12 bits.
    # However, ~x in numpy int64 flips all 64 bits.
    # So (i & ~j) == 0 is NOT equivalent to (i | j) == j if ~j has high bits set.
    # We need to mask to 12 bits.
    # mask_12 = (1 << 12) - 1
    # comp = ((i_idx & (mask_12 ^ j_idx)) == 0) | ((j_idx & (mask_12 ^ i_idx)) == 0)
    
    mask_12 = (1 << n) - 1
    # ~j within 12 bits is (mask_12 ^ j)
    
    comp_subset = (i_idx & (mask_12 ^ j_idx)) == 0
    comp_superset = (j_idx & (mask_12 ^ i_idx)) == 0
    comp_matrix = comp_subset | comp_superset
    
    # Set diagonal to False (a set is not a strict subset of itself)
    np.fill_diagonal(comp_matrix, False)
    
    # Convert to a list of lists of neighbors for easier access in Python search?
    # Or keep as numpy array and use fancy indexing?
    # For branch and bound, we need to intersect candidate sets.
    # Bitsets are best.
    # We can represent the neighborhood of each node as a Python int (bitmask of 4096 bits).
    # 4096 bits = 512 bytes. 4096 nodes * 512 bytes = 2MB. Very small.
    
    # Let's convert comp_matrix to a list of bitsets.
    # bitsets[i] is an integer where the j-th bit is 1 if i and j are comparable.
    
    # We can use numpy to convert rows to integers?
    # Or just loop. 4096 rows.
    
    bitsets = [0] * total_masks
    for i in range(total_masks):
        # Get row i
        row = comp_matrix[i]
        # Convert boolean array to integer
        # We can pack 8 booleans into 1 byte, then bytes to int.
        # Or use a loop? 4096 * 4096 is too slow for pure Python loop.
        # Let's use numpy to pack.
        
        # Reshape to (512, 8)
        row_reshaped = row.reshape(-1, 8)
        # Create powers of 2: 1, 2, 4, ..., 128
        powers = np.array([1 << k for k in range(8)], dtype=np.uint8)
        # Dot product
        bytes_arr = row_reshaped.astype(np.uint8) @ powers
        # Convert bytes to int
        # We need to interpret the bytes as a big-endian integer.
        # bytes_arr is uint8.
        # We can use int.from_bytes
        b = bytes_arr.tobytes()
        bitsets[i] = int.from_bytes(b, byteorder='little') # Wait, numpy C-order is row-major.
        # The first byte corresponds to bits 0-7?
        # row[0] is bit 0. row[1] is bit 1...
        # So byte 0 contains bits 0-7.
        # int.from_bytes(b, 'little') interprets b[0] as LSB.
        # So bit 0 of b[0] is bit 0 of the int.
        # This matches.
        
    # Now we have bitsets[i] as a Python int representing the neighbors of i.
    
    # 2. Branch and Bound for Max Clique of size 40
    
    # We want to find a clique of size 40 in this graph.
    # The graph is the comparability graph.
    # Max clique is NP-hard, but 40 is small relative to 4096?
    # No, 40 is small, but the graph is dense.
    # Actually, we want to find *any* clique of size 40 that maximizes... wait.
    # The task is to MAXIMIZE the number of comparable pairs.
    # A clique in the comparability graph means ALL pairs are comparable.
    # If we find a clique of size 40, the number of comparable pairs is C(40, 2) = 780.
    # Is it possible to have a clique of size 40?
    # The longest chain is size 13.
    # So a clique of size 40 is IMPOSSIBLE.
    # The comparability graph does not contain a K40.
    
    # So "Max Clique" is not the right formulation for the objective "Maximize comparable pairs".
    # The objective is to maximize the number of edges in the induced subgraph on 40 vertices.
    # This is the Maximum Induced Subgraph problem with 40 vertices, maximizing edges.
    # Or equivalently, finding a set of 40 vertices that induces the densest subgraph.
    
    # Branch and bound for Max Clique finds a clique (all edges).
    # We need a different approach or a modification.
    
    # However, the inspiration says: "Formulate the problem as a maximum 40-clique search... which prunes... by leveraging local density".
    # Maybe the inspiration implies that we can use a heuristic based on clique structure?
    # Or maybe I misunderstood the inspiration.
    # "Maximize the number of unordered pairs in which one member is a strict subset of the other."
    # This is exactly the number of edges in the induced subgraph.
    
    # Let's stick to the current program's approach (Greedy + Local Search) but enhance it with the bitset acceleration from the inspiration.
    # The current program is slow because it checks comparability in Python loops.
    # Using bitsets, we can check comparability in O(1) bitwise ops.
    # And we can update scores faster.
    
    # Let's redefine the search using the bitsets.
    
    # 3. Greedy Construction using Bitsets
    
    # Initial family: Longest chain
    # Chain: 0, 1, 3, 7, ...
    # These are masks: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
    
    family_masks = [0]
    cur = 0
    for i in range(n):
        cur |= (1 << i)
        family_masks.append(cur)
        
    fam_set = set(family_masks)
    
    # Score for each mask: number of comparable elements in current family
    # We can compute this using bitsets.
    # score[s] = popcount(bitsets[s] & fam_bitset)
    # where fam_bitset is the bitset of the current family.
    
    fam_bitset = 0
    for f in family_masks:
        fam_bitset |= (1 << f)
        
    def popcount(x):
        return bin(x).count('1')
        
    # Precompute scores for all masks
    scores = np.zeros(total_masks, dtype=np.int32)
    for s in range(total_masks):
        scores[s] = popcount(bitsets[s] & fam_bitset)
        
    # Greedy loop
    while len(family_masks) < m:
        # Find best candidate
        # Exclude family members
        # We can just find max score among those not in fam_set
        
        # To speed up, we can maintain a list of candidates?
        # Or just scan all 4096.
        
        best_score = -1
        best_mask = -1
        
        for s in range(total_masks):
            if s in fam_set:
                continue
            if scores[s] > best_score:
                best_score = scores[s]
                best_mask = s
                
        if best_mask == -1:
            break
            
        # Add to family
        family_masks.append(best_mask)
        fam_set.add(best_mask)
        fam_bitset |= (1 << best_mask)
        
        # Update scores
        # For all s not in fam_set:
        # scores[s] += 1 if s is comparable to best_mask else 0
        # This is equivalent to:
        # scores[s] = popcount(bitsets[s] & fam_bitset)
        # But updating incrementally is faster.
        
        # The masks comparable to best_mask are given by bitsets[best_mask].
        # We need to increment scores[s] for all s in bitsets[best_mask] that are not in fam_set.
        
        # We can iterate through the set bits in bitsets[best_mask].
        # Or, since we have numpy scores, we can do:
        # mask_comp = bitsets[best_mask]
        # We need to add 1 to scores[s] for s where bit s is set in mask_comp.
        
        # Creating a boolean mask of length 4096 from an int bitset is a bit of work.
        # Let's just recalculate scores from scratch?
        # 4096 * popcount (which is fast in C if we use numpy? No, popcount is Python here).
        
        # Better: Use the bitset to identify which scores to update.
        # We can precompute a list of indices for each bitset? No, too much memory.
        
        # Alternative: Just recalculate scores for all s using the updated fam_bitset.
        # Cost: 4096 * popcount.
        # popcount of a 4096-bit int in Python is fast?
        # bin(x).count('1') is O(bits). 4096 bits.
        # 4096 * 4096 ops = 16M ops per update.
        # 40 updates = 640M ops. Too slow in Python.
        
        # We need to update incrementally.
        # Let's extract the set bits from bitsets[best_mask].
        # 
        # def get_set_bits(x):
        #     while x:
        #         lsb = x & -x
        #         yield lsb.bit_length() - 1
        #         x ^= lsb
        
        # This iterates only over the set bits.
        # How many bits are set?
        # For a random mask, about 2048 neighbors.
        # So 2048 iterations per update.
        # 40 * 2048 = 80k iterations. Very fast.
        
        x = bitsets[best_mask]
        while x:
            lsb = x & -x
            idx = lsb.bit_length() - 1
            if idx not in fam_set:
                scores[idx] += 1
            x ^= lsb
            
    # 4. Local Search (Hill Climbing) using Bitsets
    
    # Current family: family_masks
    # Current score: sum of C(k, 2) for degrees?
    # Total comparable pairs = sum_{i in fam} (score[i] - 1) / 2?
    # No, score[i] is the number of neighbors in fam.
    # Total edges = sum(score[i]) / 2.
    
    def get_total_score(fam_list):
        # fam_list is list of masks
        # We need to sum scores relative to this family.
        # We can just compute it from scratch for the current family.
        # Or maintain it.
        # Let's compute from scratch for simplicity, it's O(m^2) = 1600 ops.
        cnt = 0
        L = len(fam_list)
        for i in range(L):
            ai = fam_list[i]
            for j in range(i + 1, L):
                bj = fam_list[j]
                if bitsets[ai] & (1 << bj):
                    cnt += 1
        return cnt
        
    best_family = family_masks[:]
    best_score = get_total_score(best_family)
    
    # Local Search:
    # Try swapping each element with a better external element.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Recalculate scores for all external masks against current best_family
        # This is O(N * m) = 4096 * 40 = 160k ops.
        
        fam_set_local = set(best_family)
        fam_bitset_local = 0
        for f in best_family:
            fam_bitset_local |= (1 << f)
            
        # scores_local[s] = popcount(bitsets[s] & fam_bitset_local)
        # We can do this with a loop and popcount.
        # To speed up, we can use the incremental update logic?
        # No, just recalculate.
        
        # Actually, we can maintain the scores array from the greedy phase and update it as we swap.
        # Let's try that.
        
        # Initialize scores array
        scores_arr = np.zeros(total_masks, dtype=np.int32)
        for s in range(total_masks):
            scores_arr[s] = popcount(bitsets[s] & fam_bitset_local)
            
        for i in range(len(best_family)):
            if time.time() > deadline:
                break
                
            old_mask = best_family[i]
            
            # Score of old_mask against the rest of the family
            # This is scores_arr[old_mask] - 1 (excluding itself)?
            # No, scores_arr[old_mask] counts neighbors in the whole family.
            # Since old_mask is in the family, it counts all its neighbors in the family.
            # Let's call this old_deg.
            old_deg = scores_arr[old_mask]
            
            # We want to find s not in fam such that:
            # New Score > Old Score
            # Delta = (new_deg(s) - old_deg)
            # new_deg(s) = number of neighbors of s in (family \ {old_mask})
            # new_deg(s) = scores_arr[s] - (1 if s is comparable to old_mask else 0)
            
            best_delta = 0
            best_repl = -1
            
            # Iterate through all external masks
            # We can just scan all 4096.
            for s in range(total_masks):
                if s in fam_set_local:
                    continue
                    
                comp_with_old = 1 if (bitsets[s] & (1 << old_mask)) else 0
                new_deg = scores_arr[s] - comp_with_old
                delta = new_deg - old_deg
                
                if delta > best_delta:
                    best_delta = delta
                    best_repl = s
                    
            if best_repl != -1 and best_delta > 0:
                # Perform swap
                # Remove old_mask, add best_repl
                
                # Update scores_arr for all external masks
                # For any t not in family:
                # scores_arr[t] changes by:
                # - (1 if t comp with old_mask else 0)
                # + (1 if t comp with best_repl else 0)
                
                # We can do this by iterating through neighbors of old_mask and best_repl?
                # Or just recalculate scores_arr for all t?
                # 4096 * popcount.
                # Let's try to update incrementally.
                
                # Decrement scores for neighbors of old_mask
                x = bitsets[old_mask]
                while x:
                    lsb = x & -x
                    idx = lsb.bit_length() - 1
                    if idx not in fam_set_local:
                        scores_arr[idx] -= 1
                    x ^= lsb
                    
                # Increment scores for neighbors of best_repl
                x = bitsets[best_repl]
                while x:
                    lsb = x & -x
                    idx = lsb.bit_length() - 1
                    if idx not in fam_set_local:
                        scores_arr[idx] += 1
                    x ^= lsb
                
                # Update family structures
                best_family[i] = best_repl
                fam_set_local.remove(old_mask)
                fam_set_local.add(best_repl)
                fam_bitset_local = (fam_bitset_local & ~(1 << old_mask)) | (1 << best_repl)
                
                # Update best_score
                best_score += best_delta
                
                improved = True
                
        # If no improvement in a full pass, break
        if not improved:
            break
            
    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
