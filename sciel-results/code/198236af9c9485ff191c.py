"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search with multiple restarts.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    all_words = list(range(total_words))
    
    # Precompute neighbors and coverers (inverse neighbors)
    # neighbors[w] = list of words within distance <= radius from w
    # coverers[w] = list of words c such that w is within distance <= radius from c
    # Note: neighbors and coverers are the same sets, just indexed differently?
    # If d(a,b) <= r, then b is in neighbors[a] and a is in coverers[b].
    # So neighbors[w] is the set of words covered by w.
    # coverers[w] is the set of codewords that cover w.
    # These are symmetric. neighbors[w] == coverers[w] if the relation is symmetric (which Hamming distance is).
    
    neighbors = [None] * total_words
    for w in all_words:
        word = int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.append(word_to_int(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                orig_val1 = word[pos1]
                for val1 in range(q):
                    if val1 == orig_val1:
                        continue
                    for pos2 in range(pos1 + 1, n):
                        orig_val2 = word[pos2]
                        for val2 in range(q):
                            if val2 == orig_val2:
                                continue
                            new_word = word[:]
                            new_word[pos1] = val1
                            new_word[pos2] = val2
                            neigh.append(word_to_int(new_word, q))
        # Remove duplicates just in case, though logic above should be unique for radius 1.
        # For radius 2, duplicates are possible if q=2? No, positions are distinct.
        # But if q=1, no changes.
        neighbors[w] = list(set(neigh))
        
    # Precompute coverers (same as neighbors due to symmetry)
    # But for clarity, we can just use neighbors.
    # coverers[u] = [c for c in all_words if u in neighbors[c]]
    # This is expensive to compute naively.
    # But we can build it from neighbors.
    coverers = [set() for _ in range(total_words)]
    for c in all_words:
        for u in neighbors[c]:
            coverers[u].add(c)
    # Convert to lists for faster iteration if needed, or keep sets for intersection.
    # Sets are good for intersection.
    
    def greedy_init_linear():
        """
        Fast greedy: iterate through random permutation, add word if it covers any new word.
        """
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        perm = np.random.permutation(total_words)
        
        for w in perm:
            neigh = neighbors[w]
            # Check if any neighbor is not covered
            # covered[neigh] is a boolean array
            if np.any(~covered[neigh]):
                codewords.append(w)
                covered[neigh] = True
                
        return codewords

    def optimize(codewords, time_limit):
        # Initialize coverage counts
        cover_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            cover_count[neighbors[cw]] += 1
            
        cw_list = list(codewords)
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed_any = False
            # We need to be careful with indices if we remove items.
            # It's safer to iterate and build a new list, or mark for removal.
            # Given small size, rebuilding or popping is fine.
            # But popping changes indices.
            # Let's use a "to_remove" set.
            to_remove = set()
            # Shuffle to break symmetry
            indices = list(range(len(cw_list)))
            random.shuffle(indices)
            
            for i in indices:
                if i in to_remove:
                    continue
                cw = cw_list[i]
                neigh = neighbors[cw]
                # If min cover count in neigh is > 1, it's redundant
                if np.min(cover_count[neigh]) > 1:
                    to_remove.add(cw)
                    # We don't update cover_count yet, to avoid confusion?
                    # Actually, if we remove multiple, we should update counts.
                    # But if A and B are both redundant, removing A makes B even more redundant.
                    # So we can just mark all redundant ones in one pass?
                    # Yes, if min(cover_count[neigh]) > 1, it is redundant regardless of other removals.
                    # Because removing other codewords only decreases counts.
                    # So if it's > 1 now, it will be >= 1 after others are removed.
                    # Wait, if it's 2, and we remove another codeword that covers it, it becomes 1.
                    # Then it is NO LONGER redundant (it's the last cover).
                    # So we MUST update counts sequentially or carefully.
                    
            # Sequential removal is safer.
            changed = True
            while changed:
                changed = False
                indices = list(range(len(cw_list)))
                random.shuffle(indices)
                for i in indices:
                    cw = cw_list[i]
                    neigh = neighbors[cw]
                    if np.min(cover_count[neigh]) > 1:
                        cover_count[neigh] -= 1
                        cw_list.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        break # Restart loop to re-index? Or just continue?
                        # If we break, we restart the while loop.
                        # This is O(N^2) in worst case but N is small.
            return changed

        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
                
            if len(cw_list) == 0:
                break
                
            idx = random.randint(0, len(cw_list) - 1)
            old_cw = cw_list[idx]
            neigh_old = neighbors[old_cw]
            
            # Find words uniquely covered by old_cw
            # These are words u where cover_count[u] == 1 and u in neigh_old
            # We need to cover these with a new codeword.
            unique_covers = [u for u in neigh_old if cover_count[u] == 1]
            
            if len(unique_covers) == 0:
                # Should have been removed, but just in case
                cover_count[neigh_old] -= 1
                cw_list.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Find candidates that cover all unique_covers
            # candidates = intersection(coverers[u] for u in unique_covers)
            # Start with the first one
            candidates = set(coverers[unique_covers[0]])
            for u in unique_covers[1:]:
                candidates.intersection_update(coverers[u])
                
            # Remove old_cw from candidates
            candidates.discard(old_cw)
            
            if not candidates:
                # No valid swap found for this old_cw. Try another?
                # Or just continue to next iteration.
                continue
            
            # Pick a candidate
            # Heuristic: Pick one that maximizes "redundancy potential"?
            # Or just random.
            new_cw = random.choice(list(candidates))
            
            # Perform swap
            neigh_new = neighbors[new_cw]
            
            # Check if this swap is beneficial?
            # It doesn't change size.
            # It might allow further removals.
            # Let's just do it.
            cover_count[neigh_old] -= 1
            cover_count[neigh_new] += 1
            cw_list[idx] = new_cw
            cw_set.discard(old_cw)
            cw_set.add(new_cw)
            
        try_remove_redundant()
        return cw_list

    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    
    while time.time() - start_time < time_limit:
        # Allocate time for this iteration
        # Greedy is very fast now.
        # Give most time to optimize.
        # Estimate greedy time: ~0.01s.
        # Optimize time: rest.
        
        cw = greedy_init_linear()
        
        # Calculate time remaining for optimization
        elapsed = time.time() - start_time
        opt_time = time_limit - elapsed - 2 # 2s buffer
        
        if opt_time <= 0:
            break
            
        cw = optimize(cw, opt_time)
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
            
        # If we found a very good one, we might stop early?
        # Or just keep going to see if we can do better.
        
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
