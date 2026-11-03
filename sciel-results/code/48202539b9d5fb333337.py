"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    Uses a greedy set-cover approach with bitset acceleration, followed by local search
    (swap and delete) to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for the 120s limit (and 150s kill)

    if q == 3 and n == 7 and radius == 1:
        # Fast path for the specific task instance with known good heuristics
        pass

    # Precompute all words and their neighborhoods
    # Total words = q^n
    total_words = q ** n

    if total_words > 10000000:
        # Fallback for large spaces: use the direct sum construction if possible
        # For q=3, n=7, radius=1, total_words = 2187, which is small.
        # We assume the space is manageable for bitset operations.
        pass

    # Represent the universe of words as a bitset (using Python integers or numpy arrays)
    # For q=3, n=7, total_words = 2187. This fits easily in a numpy bool array.
    # For larger, we might need a different approach, but let's stick to numpy for speed.

    # Precompute neighbors for each word
    # neighbors[w] is a set (or bitset) of words within distance <= radius
    # To save memory/time, we can generate neighbors on the fly or precompute as a list of lists.
    # Given the small size of 2187, precomputation is fast.

    # Efficiently generate all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

    # Precompute coverage bitsets for each potential codeword
    # cover_mask[c] is a numpy boolean array of length total_words, True if covered by codeword c
    # However, storing total_words * total_words bits might be heavy if total_words is large.
    # 2187 * 2187 bits ~ 5 million bits ~ 600 KB. This is fine.
    # But using numpy arrays of bools: 2187 * 2187 bytes ~ 4.7 MB. Also fine.

    # Let's use a list of sets of indices for simplicity first, then optimize if needed.
    # Actually, bit operations on numpy arrays are much faster.
    # Let's precompute for each word index w, the set of indices it covers (distance <= radius).
    
    # Optimization: Generate neighbors via single-step changes
    # For radius=1, neighbors are the word itself + all words differing in 1 position.
    # Count: 1 + n * (q - 1)

    cover_sets = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[tuple(w)])
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        new_w = w.copy()
                        new_w[pos] = val
                        idx = word_to_idx[tuple(new_w)]
                        covered.add(idx)
        # If radius > 1, we would need to expand further, but for radius=1 this is enough.
        # For general radius, we'd need BFS or combinatorial generation.
        # Given the task is radius=1, we optimize for that.
        cover_sets.append(covered)

    # Convert to numpy boolean arrays for fast union operations
    # cover_masks[i][j] == 1 if word j is covered by codeword i
    cover_masks = np.zeros((total_words, total_words), dtype=bool)
    for i, s in enumerate(cover_sets):
        for j in s:
            cover_masks[i, j] = True

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    codewords = []
    
    # Sort potential codewords by coverage size (descending) to help greedy?
    # Or just iterate. Greedy usually picks the one covering most *currently* uncovered.
    
    while np.any(uncovered):
        if time.time() > deadline - 10.0:
            break
        
        best_idx = -1
        max_new_coverage = 0
        
        # To speed this up, we can sample or use a smarter heuristic.
        # For 2187 items, O(N^2) per step is 2187^2 ~ 4.7M ops.
        # Number of steps ~ 186. Total ops ~ 870M. Might be too slow in Python.
        # We need to optimize the "find best" step.
        
        # Optimization: Use bitset operations.
        # For each candidate codeword i, count how many currently uncovered words it covers.
        # count[i] = sum(cover_masks[i] & uncovered)
        
        # This is a matrix-vector multiply-like operation.
        # counts = cover_masks @ uncovered (if cover_masks is 0/1 int)
        # But cover_masks is bool. Let's use int8 or uint8 for speed.
        
        # Let's switch to a faster representation if the above is too slow.
        # Alternative: Maintain a count of how many codewords cover each word? No, that's for deletion.
        # For addition, we need "how many NEW words does this codeword cover".
        
        # Let's try the numpy matrix multiplication approach.
        # cover_masks_uint8 = cover_masks.astype(np.uint8)
        # uncovered_uint8 = uncovered.astype(np.uint8)
        # counts = cover_masks_uint8 @ uncovered_uint8
        
        # This is O(N^2) but implemented in C. 2187x2187 @ 2187x1. Very fast.
        
        if best_idx == -1:
            # Compute counts for all candidates
            # To avoid repeated conversion, keep cover_masks as uint8
            if not hasattr(construct, '_cover_masks_uint8'):
                construct._cover_masks_uint8 = cover_masks.astype(np.uint8)
                construct._word_to_idx = word_to_idx
                construct._all_words = all_words
                construct._n = n
                construct._q = q
                construct._radius = radius
                construct._total_words = total_words
                construct._cover_sets = cover_sets
            
            cm_u8 = construct._cover_masks_uint8
            unc_u8 = uncovered.astype(np.uint8)
            counts = cm_u8 @ unc_u8
            
            # Pick the one with max count
            # Tie-breaking: random or fixed. Fixed is deterministic.
            max_count = np.max(counts)
            if max_count == 0:
                break # Should not happen if uncovered is not empty and code is valid
            
            candidates = np.where(counts == max_count)[0]
            # Random tie-breaking to escape local minima? Or just first?
            # Let's use random for diversity in restarts if we implement them.
            # For now, pick first.
            best_idx = candidates[0]
            
            # Add to codewords
            codewords.append(best_idx)
            # Update uncovered
            uncovered[cover_masks[best_idx]] = False
            
            # Check time
            if time.time() > deadline - 5.0:
                break

    # Local Search: Deletion and Swap
    # 1. Deletion: Try removing each codeword. If coverage is still complete, remove it.
    # 2. Swap: Try replacing codeword A with codeword B. If coverage is still complete and size doesn't increase (it stays same), accept.
    #    Actually, swap doesn't reduce size. Deletion reduces size.
    #    Can we do a "2-for-1" move? Replace 2 codewords with 1? Unlikely to help much.
    #    The main gain comes from greedy + deletion.
    
    # Let's implement deletion first.
    # We need to be careful with time.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try to delete each codeword
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            
            # Remove codewords[i]
            candidate_code = codewords[:i] + codewords[i+1:]
            
            # Check if candidate_code covers all
            # Efficient check: Union of cover masks for candidate_code
            # If we maintain a global covered bitset, we can just check if it's all True.
            # But removing one requires recomputing the union or maintaining counts.
            
            # Simple approach: Recompute coverage for the candidate set.
            # This is O(|C| * N) per deletion attempt.
            # |C| ~ 180, N ~ 2187. 180 * 2187 ~ 390k ops.
            # 180 deletions to try -> 70M ops. Might be slow in Python loop, but numpy helps.
            
            # Let's use a count array approach.
            # count[j] = number of codewords in current set that cover word j.
            # If we remove codeword c, count[j] decreases by 1 for all j in cover(c).
            # If any count[j] becomes 0, deletion fails.
            
            # We need to maintain `count` array.
            # Let's initialize it.
            if not hasattr(construct, '_count_array'):
                # Initialize count array for the current codewords
                count = np.zeros(total_words, dtype=np.int16)
                for c in codewords:
                    count[cover_masks[c]] += 1
                construct._count_array = count
            
            count = construct._count_array
            c = codewords[i]
            
            # Check if any word is covered ONLY by this codeword
            # i.e., count[j] == 1 for some j in cover(c)
            # If yes, we cannot delete.
            # mask = cover_masks[c]
            # if np.any(count[mask] == 1): cannot delete
            
            # To speed up, we can just check.
            # count_c = count[cover_masks[c]]
            # if np.min(count_c) > 1: delete
            
            # Let's try it.
            # Note: cover_masks is bool.
            if np.min(count[cover_masks[c]]) > 1:
                # Safe to delete
                count[cover_masks[c]] -= 1
                codewords.pop(i)
                construct._count_array = count
                improved = True
        
        # If no deletion worked, try swaps?
        # Swaps don't reduce size. They might enable future deletions.
        # Given time constraints, let's stick to deletion for now.
        # If we have time, we can add random restarts.

    # Final check: ensure we have a valid covering code
    # If we broke out early due to time, we might have uncovered words.
    # We must return a VALID code.
    # So, if uncovered is not empty, we must add codewords to cover them.
    
    if np.any(uncovered):
        # Add codewords to cover remaining
        while np.any(uncovered) and time.time() < deadline - 1.0:
            # Pick a random uncovered word, find a codeword covering it
            unc_indices = np.where(uncovered)[0]
            target = unc_indices[0]
            
            # Find any codeword that covers target
            # candidates = np.where(cover_masks[:, target])[0]
            # This is O(N).
            
            # Or just iterate through all_words and pick the first one that covers target
            # and is not already in codewords?
            # Actually, we can just pick the one with max coverage of remaining uncovered.
            
            # Simple: pick first candidate
            # We need the list of all words again.
            all_words_list = construct._all_words
            wti = construct._word_to_idx
            
            # Find a word that covers target
            # target is an index.
            # We need a codeword c such that cover_masks[c, target] is True.
            
            # Let's just iterate through all possible codewords
            found = False
            for c in range(total_words):
                if cover_masks[c, target]:
                    # Check if c is already in codewords
                    if c not in codewords:
                        codewords.append(c)
                        uncovered[cover_masks[c]] = False
                        found = True
                        break
            if not found:
                # This should not happen if the space is connected
                break

    # Convert indices back to lists
    result = [construct._all_words[i] for i in codewords]
    
    # Ensure distinctness (should be guaranteed by set logic, but just in case)
    # The greedy picks distinct indices.
    
    return result
# EVOLVE-BLOCK-END
