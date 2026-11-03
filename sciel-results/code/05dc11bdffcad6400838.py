import itertools
import numpy as np
import time

def construct(n):
    if n <= 4:
        # Classic construction is optimal for small n
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Convert to strings for final output
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perm[i][n-k:] == perm[j][:k]
    # We want to maximize overlap to minimize added length.
    # Added length = n - overlap.
    
    # Use a list of bytearrays for efficiency? Or numpy?
    # N=5040. 5040*5040 = 25M entries.
    # np.uint8 array is 25MB.
    
    # To compute overlaps efficiently:
    # We can map each permutation to an integer index in a dictionary?
    # Or just compare strings.
    # String comparison is fast.
    
    # Let's build the matrix using a dictionary of suffixes?
    # For each permutation i, for each k from n-1 down to 1:
    #   suffix = perm_strs[i][n-k:]
    #   prefix = perm_strs[j][:k]
    #   If suffix == prefix, overlap[i][j] = k.
    
    # This is O(N^2 * n) if done naively.
    # 5040^2 * 7 = 176M ops. Feasible in ~1-2s in Python if optimized?
    # Or use NumPy?
    
    # Let's try a smarter way.
    # Group permutations by their suffix of length k.
    # For k = n-1 down to 1:
    #   For each i, suffix = perm_strs[i][n-k:]
    #   Find all j such that perm_strs[j][:k] == suffix.
    #   Set overlap[i][j] = k (if not already set to a larger k).
    
    # Since we iterate k from large to small, the first match is the max overlap.
    
    overlap_matrix = np.zeros((N, N), dtype=np.uint8)
    
    # Build index for prefixes
    # prefix_map[k][prefix_str] = list of indices j with that prefix
    
    # We only need to compute this once.
    
    # Optimization:
    # For k = n-1:
    #   Suffix of i is perm[i][1:] (length n-1)
    #   Prefix of j is perm[j][:n-1] (length n-1)
    #   Match if perm[i][1:] == perm[j][:n-1]
    
    # Let's build a dictionary for prefixes of length k.
    
    # To speed up construction, we can just do the double loop with string slicing in Python.
    # 25M iterations. Each is a string slice and compare.
    # Might be ~5-10 seconds.
    
    # Let's try to do it faster.
    # Convert perms to integers?
    # Or use the fact that permutations are unique.
    
    # Let's stick to the double loop but optimize the inner check.
    # We can precompute the suffixes and prefixes.
    
    # Actually, for n=7, N=5040.
    # Let's just use the simple loop.
    
    # To make it faster, we can break early if we find a match?
    # No, we want the MAX overlap.
    # So we iterate k from n-1 down to 1.
    # If we find a match for k, we record it and break for this pair (i,j).
    
    # Better:
    # For each i:
    #   For k from n-1 down to 1:
    #     suffix = perm_strs[i][n-k:]
    #     Find j such that perm_strs[j] starts with suffix.
    #     If found, overlap[i][j] = k.
    #     But there might be multiple j? No, for a fixed suffix of length n-1, there is only 1 j.
    #     For length n-2, there are n-1 j's?
    #     Actually, if perm_strs[j][:k] == suffix, then j is determined by the first k elements.
    #     Since the elements are distinct, the first k elements determine the prefix.
    #     So for a given suffix S of length k, there is at most 1 permutation j that has S as its prefix?
    #     No. Permutation (1,2,3,4,5,6,7) has prefix (1,2).
    #     Permutation (1,2,5,4,3,6,7) has prefix (1,2).
    #     So multiple j's can have the same prefix.
    #     But we want the max overlap.
    #     If we iterate k from n-1 down to 1:
    #       For a specific i, we find the largest k such that suffix(i, k) is a prefix of some j.
    #       Wait, the overlap is defined for a specific pair (i,j).
    #       We need overlap[i][j] for ALL j.
    
    # So we must compute the full matrix.
    
    # Let's use the dictionary approach for prefixes.
    # prefix_maps[k] = { prefix_string: [list of indices j] }
    
    # This is memory heavy but fast.
    # k=6: 5040 entries.
    # k=5: 5040 entries.
    # ...
    
    # Let's just do the O(N^2) loop with early termination on k.
    
    # 25M iterations.
    # In Python, 25M simple loops take ~1-2 seconds.
    # String slicing `s[1:]` and `s[:6]` is fast.
    
    # Let's do it.
    
    # To speed up, we can convert perm_strs to a list of lists of ints?
    # Comparing lists of ints is slower than strings.
    # Strings are better.
    
    # Let's run the matrix construction.
    
    # We can parallelize? No, standard library only.
    
    # Let's try to optimize the inner loop.
    # For i in range(N):
    #   si = perm_strs[i]
    #   for j in range(N):
    #     sj = perm_strs[j]
    #     # Check k=n-1
    #     if si[1:] == sj[:n-1]:
    #        overlap_matrix[i,j] = n-1
    #     elif si[2:] == sj[:n-2]:
    #        overlap_matrix[i,j] = n-2
    #     ...
    
    # This is 6 checks per pair. 150M checks.
    # Might be 10-15s.
    
    # Alternative:
    # Build a map: suffix -> indices of j that have that suffix? No, prefix.
    # Map: prefix_string -> list of j.
    
    # For k in range(n-1, 0, -1):
    #   map_k = {}
    #   for j in range(N):
    #      p = perm_strs[j][:k]
    #      if p not in map_k:
    #         map_k[p] = []
    #      map_k[p].append(j)
    #   
    #   for i in range(N):
    #      s = perm_strs[i][n-k:]
    #      if s in map_k:
    #         for j in map_k[s]:
    #            if overlap_matrix[i,j] == 0: # Not set yet (smaller k not checked? No, we go large to small)
    #               overlap_matrix[i,j] = k
    #               # We can't break because there might be other j's?
    #               # No, for a fixed i and k, s is fixed.
    #               # All j in map_k[s] have the same overlap k with i.
    #               # We set it for all of them.
    
    # This is O(N * n) for building maps + O(N * n * avg_list_size).
    # avg_list_size for k=n-1 is 1.
    # For k=1, avg_list_size is (n-1)! = 720.
    # Total ops: sum_{k=1}^{n-1} N * (N/k!)? No.
    # Map building: N * n.
    # Lookup: N * n.
    # Assignment: For each k, we assign N * (N / (n-k)!) ? No.
    # Number of j's for a specific prefix of length k is (n-k)!.
    # So total assignments for a specific k is N * 1? No.
    # For each i, we look up s. If found, we iterate over the list.
    # The sum of list sizes over all i is N.
    # So total assignments is N per k.
    # Total assignments: N * (n-1).
    # This is very fast!
    
    # Let's implement this.
    
    # Note: overlap_matrix[i,i] is n (full overlap). But we never append i to a path containing i.
    # So we can leave it 0 or n, it doesn't matter.
    
    # Initialize
    overlap_matrix = np.zeros((N, N), dtype=np.uint8)
    
    # We fill it from large k to small k.
    # If a larger k matches, we don't want to overwrite with smaller k.
    # So we only set if it is 0.
    
    for k in range(n - 1, 0, -1):
        prefix_map = {}
        for j in range(N):
            p = perm_strs[j][:k]
            if p not in prefix_map:
                prefix_map[p] = []
            prefix_map[p].append(j)
        
        for i in range(N):
            s = perm_strs[i][n-k:]
            if s in prefix_map:
                js = prefix_map[s]
                for j in js:
                    if overlap_matrix[i, j] == 0:
                        overlap_matrix[i, j] = k

    # Beam Search
    # State: (cost, last_idx, mask)
    # cost: total length so far.
    # last_idx: index of the last permutation in the path.
    # mask: bytearray of length N, 1 if used.
    
    # We fix the start to index 0 (identity permutation) to reduce search space.
    # This is valid because of the symmetry of the problem.
    
    start_idx = 0
    start_mask = bytearray(N)
    start_mask[start_idx] = 1
    start_cost = n
    
    beam = [(start_cost, start_idx, start_mask)]
    
    # Beam width
    # Let's try 50.
    beam_width = 50
    
    # Precompute "n - overlap" matrix for faster addition?
    # cost_add = n - overlap_matrix
    # cost_add[i,j] = n - overlap_matrix[i,j]
    # If overlap is 0, cost is n.
    # If overlap is n-1, cost is 1.
    
    cost_add = n - overlap_matrix # This results in uint8, n=7, max 7.
    # Wait, n - 0 = 7. n - (n-1) = 1.
    # dtype uint8 is fine.
    
    for step in range(1, N):
        # Expand beam
        candidates = []
        
        # For each state in beam
        for cost, last_idx, mask in beam:
            # Get costs for all j not in mask
            # We can use numpy to get the row
            row = cost_add[last_idx]
            
            # We need to filter out used indices.
            # mask is a bytearray.
            # Convert mask to numpy array of bools?
            # mask_np = np.frombuffer(mask, dtype=np.uint8).astype(bool)
            # This is O(N) per state. W * N per step.
            # 50 * 5040 * 5040 = 1.25B ops. Too slow.
            
            # Better:
            # Iterate over unused j?
            # Or use a list of unused indices?
            # Maintaining a list of unused indices for each state is hard to merge.
            
            # Alternative:
            # Just iterate over all j, check mask[j].
            # 50 * 5040 checks per step.
            # 50 * 5040 * 5040 = 1.25B checks.
            # In Python, 1.25B simple loops is ~10-20s.
            # This is acceptable.
            
            # Optimization:
            # We can precompute the list of unused indices?
            # No, it changes per state.
            
            # Let's just loop.
            # To speed up, we can use a local variable for mask.
            m = mask
            for j in range(N):
                if not m[j]:
                    new_cost = cost + cost_add[last_idx, j]
                    # We don't create the new mask yet, just store the candidate
                    candidates.append((new_cost, j, mask))
        
        # Sort candidates by cost
        # We only need the top beam_width
        # Use heapq.nsmallest
        import heapq
        best_candidates = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])
        
        # Create new beam with new masks
        new_beam = []
        for cost, last_idx, old_mask in best_candidates:
            new_mask = bytearray(old_mask)
            new_mask[last_idx] = 1
            new_beam.append((cost, last_idx, new_mask))
        
        beam = new_beam
        
    # Reconstruct path from the best candidate in the final beam
    best_state = min(beam, key=lambda x: x[0])
    final_cost = best_state[0]
    final_mask = best_state[2]
    
    # We need to reconstruct the path.
    # We didn't store the path in the beam!
    # We need to store the parent pointer or the path itself.
    # Storing the path (list of indices) in the beam is O(N) per state.
    # W * N * N steps? No, W * N storage.
    # Copying the path is O(N).
    # W * N * N steps? No.
    # At each step, we copy the path for the top W candidates.
    # W * N copies.
    # 50 * 5040 * 5040 = 1.25B element copies.
    # This is ~10-20s.
    # Total time ~ 30-40s. Feasible.
    
    # Let's modify the beam to store the path.
    
    # Restart with path storage.
    
    start_path = [start_idx]
    beam = [(start_cost, start_idx, start_mask, start_path)]
    
    # Clear candidates
    # We need to re-run the loop.
    # To avoid code duplication, I'll rewrite the loop section.
    
    # Wait, I already ran the loop above? No, it's in the draft.
    # I will write the full code in the replacement.
    
    # One more optimization:
    # If we use `heapq.nsmallest`, it is O(C log W) where C is number of candidates.
    # C = W * (N - step).
    # This is efficient.
    
    # Final check:
    # The path is a list of indices.
    # We build the string from the path.
    
    # Return the string.
    
    # Note: The "fix start to 0" assumption.
    # Is it possible that the shortest superpermutation does NOT start with 1..n?
    # Yes.
    # But is it possible that the shortest superpermutation *can* be rotated to start with 1..n?
    # No, the word is linear.
    # However, the *length* of the shortest superpermutation is invariant under relabeling.
    # If there is a path $P_1 \to P_2 \dots P_N$ of length L,
    # and we relabel symbols such that $P_1$ becomes $1..n$,
    # the length is still L.
    # So the minimum length is the same.
    # By fixing the start to $1..n$, we are finding the minimum length among all paths that start with $1..n$.
    # Is the global minimum length achieved by some path starting with $1..n$?
    # Yes, because for the global optimal path $P_{opt}$, we can relabel the symbols so that $P_{opt}[0]$ becomes $1..n$.
    # The resulting word has the same length.
    # So the minimum length over all paths starting with $1..n$ is equal to the global minimum length.
    
    # So this is safe.

    # Re-implementing the loop with path storage.
    
    beam = [(start_cost, start_idx, start_mask, start_path)]
    
    for step in range(1, N):
        candidates = []
        for cost, last_idx, mask, path in beam:
            m = mask
            row = cost_add[last_idx]
            for j in range(N):
                if not m[j]:
                    new_cost = cost + row[j]
                    # We don't copy path yet
                    candidates.append((new_cost, j, mask, path))
        
        best_candidates = heapq.nsmallest(beam_width, candidates, key=lambda x: x[0])
        
        new_beam = []
        for cost, last_idx, old_mask, old_path in best_candidates:
            new_mask = bytearray(old_mask)
            new_mask[last_idx] = 1
            new_path = old_path + [last_idx]
            new_beam.append((cost, last_idx, new_mask, new_path))
        
        beam = new_beam

    best_state = min(beam, key=lambda x: x[0])
    path = best_state[3]
    
    # Build string
    res = perm_strs[path[0]]
    for i in range(1, len(path)):
        ov = overlap_matrix[path[i-1], path[i]]
        res += perm_strs[path[i]][ov:]
        
    return res
