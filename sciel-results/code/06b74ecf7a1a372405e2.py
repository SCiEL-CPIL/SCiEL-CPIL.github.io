"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct(q, n, radius):
    """
    Construct a covering code for q-ary words of length n with covering radius r.
    Uses a greedy set-cover heuristic with randomized restarts, optimized with
    NumPy for speed.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # For the specific case of q=3, n=6, radius=1
    # We know the optimal is 73. The simple direct sum gives 729 (3^6) if naive,
    # but the tetracode-based one gives 9 * 3^2 = 81.
    # We can do better.
    
    # General approach: Greedy Set Cover
    # Universe U = all q^n words
    # Sets S_w = { u in U : d(u, w) <= radius } for each w in U
    # Goal: Select minimum number of sets S_w such that their union is U.
    
    # Since n=6 and q=3, |U| = 729.
    # Pre-compute the distance matrix or neighbor lists.
    # Distance matrix D is 729x729. This is small enough.
    
    # Let's use NumPy for vectorized operations.
    import numpy as np
    
    # Generate all words
    # Represent words as integers 0..q^n-1? Or as arrays?
    # Arrays are easier for distance calculation.
    
    # Convert word index to list of digits
    def idx_to_word(idx, q, n):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w
    
    # Precompute all words as a numpy array (n, q^n)
    # Actually (q^n, n) is better for broadcasting.
    words = np.array([idx_to_word(i, q, n) for i in range(q**n)], dtype=np.uint8)
    
    # Precompute distance matrix?
    # D[i, j] = sum(words[i] != words[j])
    # This is 729x729 = ~500k entries. Very fast.
    
    # Compute pairwise Hamming distances
    # words shape: (N, n)
    # diff = words[:, np.newaxis, :] != words[np.newaxis, :, :]  # (N, N, n)
    # dist = diff.sum(axis=2)
    
    N = q**n
    # To save memory and time, we can compute distances on the fly or use the full matrix.
    # Full matrix is fine.
    
    # However, creating (729, 729, 6) bool array is ~3M bools = 3MB. Fine.
    # Summing gives (729, 729) uint8.
    
    # Let's try to be smart. We only need to know which words are within radius.
    # We can precompute the list of neighbors for each word.
    
    # neighbors[i] = list of j such that dist(i, j) <= radius
    # Or a bitmask? 729 bits = 92 bytes. 
    # Using Python integers as bitmasks is efficient.
    
    # Let's precompute neighbor masks.
    # mask[i] is an integer with bit j set if dist(i, j) <= radius.
    
    # First, compute the distance matrix using vectorization.
    # To avoid huge intermediate arrays if n was larger, but here n=6.
    
    # Alternative: Iterate through all pairs? 729*729 = 531k pairs.
    # Python loop over 531k pairs might be slow (0.5s - 1s).
    # NumPy vectorization is better.
    
    # words: (729, 6)
    # We want dist[i, j] = sum(words[i] != words[j])
    
    # Let's compute the distance matrix.
    # words_expanded = words[:, np.newaxis, :]  # (729, 1, 6)
    # words_expanded2 = words[np.newaxis, :, :] # (1, 729, 6)
    # diff = words_expanded != words_expanded2   # (729, 729, 6)
    # dist = diff.sum(axis=2)                   # (729, 729)
    
    # This is very fast in NumPy.
    
    # But wait, we need to be careful with memory if N is large.
    # Here N=729, so it's fine.
    
    # Let's do it.
    
    # Actually, we can optimize the greedy step.
    # Greedy:
    # 1. Start with uncovered = set(all words)
    # 2. While uncovered:
    #    a. Find word w that covers the most uncovered words.
    #    b. Add w to solution.
    #    c. Remove covered words from uncovered.
    
    # To find w efficiently, we can precompute the neighbor lists.
    # neighbors[i] = list of j where dist(i,j) <= radius
    
    # Let's precompute neighbors as a list of lists or arrays.
    
    # Using the distance matrix:
    # dist <= radius gives a boolean matrix B.
    # B[i, j] is True if i covers j.
    
    # We can use this B matrix.
    
    # Greedy with bitmasks:
    # uncovered_mask: integer with bits set for uncovered words.
    # For each candidate w, count = popcount(uncovered_mask & mask[w])
    # Choose w with max count.
    
    # This requires iterating over all N candidates at each step.
    # Steps ~ 73. Candidates ~ 729.
    # 73 * 729 = 53k operations.
    # Each operation is an AND and popcount.
    # Python int popcount is fast (C implementation).
    # This should be very fast (< 0.1s).
    
    # Let's implement this.
    
    # 1. Generate words and distance matrix.
    # 2. Create masks.
    # 3. Greedy loop.
    
    # Note: The problem asks for a list of codewords (lists of ints).
    # We will return the indices of the chosen words, then convert back to lists.
    
    # Let's start.
    
    # Check if we can use the known optimal 73 directly?
    # No, we must construct it. But if we find 73, it's optimal.
    # If we find 72, it's better than known? No, lower bound is 71.
    # So 72 or 73 are possible.
    
    # Let's run the greedy.
    
    # Precomputation
    # words is (N, n)
    
    # Compute distance matrix
    # To save memory, we can compute row by row?
    # No, N is small.
    
    # dist = np.zeros((N, N), dtype=np.uint8)
    # for i in range(N):
    #    dist[i] = (words[i] != words).sum(axis=1)
    # This is N * (N * n) operations = 729 * 729 * 6 = 3.2M ops.
    # Vectorized:
    
    # Let's try the full vectorization first.
    
    # words: (N, n)
    # diff = words[:, None, :] != words[None, :, :]
    # dist = diff.sum(axis=2)
    
    # This creates a (N, N, n) boolean array.
    # 729 * 729 * 6 = 3.2M bools = 3.2 MB.
    # Summing gives (N, N) uint8.
    
    # This is fine.
    
    # Let's do it.
    
    # However, if N is larger, this might be slow.
    # For q=3, n=6, N=729.
    
    # Let's implement the greedy with bitmasks.
    
    # 1. Compute dist
    # 2. Compute masks
    #    mask[i] = sum(2**j for j where dist[i, j] <= radius)
    #    This is slow in Python loop.
    #    Better: Use numpy to create the boolean matrix B = dist <= radius
    #    Then convert each row to a bitmask.
    
    # How to convert a boolean row to a bitmask efficiently?
    # row = B[i]  # (N,) bool array
    # mask = 0
    # for j in range(N):
    #     if row[j]:
    #         mask |= (1 << j)
    # This is N * N = 531k iterations. Might take ~0.5s in Python.
    
    # Alternative:
    # We can keep the boolean matrix B and use it for counting.
    # uncovered_indices = list(range(N))
    # At each step, for each candidate w, count = sum(B[w, uncovered_indices])
    # This is summing a slice of the boolean matrix.
    # B[w, uncovered] is a boolean array of size ~N.
    # sum() is fast in NumPy.
    # But we need to do this for all w.
    # So we need to compute counts for all w at once?
    # counts = B[:, uncovered_indices].sum(axis=1)
    # This creates a (N, len(uncovered)) boolean array.
    # Initially len(uncovered) = N.
    # So (N, N) boolean array again. 3.2 MB.
    # Summing gives (N,) array of counts.
    # This is vectorized and fast.
    
    # Then we pick argmax(counts).
    # Then we update uncovered_indices by removing those covered by the chosen w.
    # covered = B[chosen, uncovered_indices]
    # new_uncovered = [u for u, c in zip(uncovered_indices, covered) if not c]
    
    # This list comprehension is O(N).
    # Total steps ~ 73.
    # 73 * (N * N ops for counts + N ops for update)
    # N * N = 531k.
    # 73 * 531k = 38M ops.
    # NumPy is fast, but Python overhead for the list comprehension and argmax might add up.
    # Should be well within 110s.
    
    # Let's try this approach.
    
    # 1. Compute B (boolean matrix)
    # 2. Greedy loop
    
    # Implementation details:
    # words: np.array (N, n)
    # dist: np.array (N, N)
    # B: dist <= radius
    
    # Let's code it.
    
    # Edge case: if radius is large enough to cover everything with 1 word, return 1 word.
    # If radius >= n, any word covers all.
    
    # Let's start.
    
    # Check time budget
    if time.time() > deadline:
        # Fallback to a simple construction
        # Direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generate all words
    N = q ** n
    words = np.array([idx_to_word(i, q, n) for i in range(N)], dtype=np.uint8)
    
    # Compute distance matrix
    # Vectorized
    diff = words[:, np.newaxis, :] != words[np.newaxis, :, :]
    dist = diff.sum(axis=2)
    
    # Boolean coverage matrix
    B = dist <= radius
    
    # Greedy Set Cover
    uncovered = set(range(N))
    solution = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Find best candidate
        # Convert uncovered to list for indexing
        unc_list = list(uncovered)
        
        # Compute counts for all candidates
        # B[:, unc_list] is (N, len(unc_list))
        # Sum along axis 1
        counts = B[:, unc_list].sum(axis=1)
        
        # Pick candidate with max count
        best_idx = np.argmax(counts)
        
        # Add to solution
        solution.append(best_idx)
        
        # Remove covered words from uncovered
        # covered = B[best_idx, unc_list]
        # This is a boolean array
        # We need to remove indices where covered is True
        # new_uncovered = set()
        # for i, u in enumerate(unc_list):
        #     if not covered[i]:
        #         new_uncovered.add(u)
        
        # Faster:
        # covered_indices = [u for u, c in zip(unc_list, B[best_idx, unc_list]) if not c]
        # uncovered = set(covered_indices)
        
        # Even faster:
        # mask = B[best_idx, unc_list]
        # new_unc_list = [u for u, c in zip(unc_list, mask) if not c]
        # uncovered = set(new_unc_list)
        
        # Let's use numpy to find the indices to keep
        # keep_mask = ~B[best_idx, unc_list]
        # new_unc_list = np.array(unc_list)[keep_mask]
        # uncovered = set(new_unc_list.tolist())
        
        keep_mask = ~B[best_idx, unc_list]
        new_unc_list = np.array(unc_list)[keep_mask]
        uncovered = set(new_unc_list.tolist())
    
    # Convert solution indices to words
    # solution is a list of indices
    # We need to return a list of lists of ints
    
    # Check if we covered everything
    if uncovered:
        # If we didn't finish in time, we might have an incomplete cover.
        # We need to ensure validity.
        # The problem says "Valid when every word ... is within Hamming distance radius".
        # If we break early, it's invalid.
        # So we must complete the cover or return a valid one.
        # The greedy loop is fast, so it should finish.
        # If it doesn't, we can add all remaining words?
        # That would be a valid cover (size N).
        
        # Let's add all remaining words to be safe.
        for u in uncovered:
            solution.append(u)
    
    # Remove duplicates? Greedy shouldn't pick duplicates if we remove covered words,
    # but a word might cover itself and be picked again?
    # No, once picked, its neighbors are removed from uncovered.
    # So it won't be picked again unless it's the only way to cover something else?
    # No, the set of covered words is monotonic.
    # So no duplicates in solution.
    
    # Convert to list of lists
    result = [idx_to_word(i, q, n) for i in solution]
    
    return result

# EVOLVE-BLOCK-END
