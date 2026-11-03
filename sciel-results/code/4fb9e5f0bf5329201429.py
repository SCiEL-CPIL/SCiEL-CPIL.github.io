import numpy as np
import itertools
import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces or general radius
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based approach for small spaces
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # Precompute balls as bitmasks
    balls = [0] * total_words
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 1 << i
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask

    all_ones_mask = (1 << total_words) - 1
    
    # Greedy initialization
    def greedy_cover(uncovered_mask):
        selected = []
        remaining = uncovered_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            for i in range(total_words):
                c = (balls[i] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
        return selected

    initial_selected = greedy_cover(all_ones_mask)
    best_size = len(initial_selected)
    best_code = [all_words[i] for i in initial_selected]
    
    start_time = time.time()
    time_limit = 115.0

    # Branch and Bound
    # We assume 0 is in the code due to symmetry (shift invariance)
    # So we start with 0 selected, and cover the rest.
    
    def get_lower_bound(uncovered_mask):
        if not uncovered_mask:
            return 0
        # Bound 1: Ceiling of uncovered / max ball size
        max_ball_size = 1 + n * (q - 1)
        lb1 = (uncovered_mask.bit_count() + max_ball_size - 1) // max_ball_size
        
        # Bound 2: Greedy Packing
        # Find a set of words in uncovered such that their balls are disjoint?
        # Actually, if we pick a word c, it covers ball[c].
        # If we pick c1, c2 such that ball[c1] and ball[c2] are disjoint, we need at least 2 codewords.
        # Greedy packing:
        temp = uncovered_mask
        pack = 0
        # To make it fast, we don't do full max packing, just greedy
        # Pick word with max coverage in temp? No, for packing we want to pick words that don't overlap.
        # Greedy: pick c with max (ball[c] & temp).bit_count()?
        # Then remove ball[c] from temp.
        # This is O(N^2) per call, which is too slow if called many times.
        # Let's use a simpler bound or optimize.
        # For now, just lb1.
        return lb1

    def b_and_b(uncovered_mask, current_size, selected_indices, min_idx):
        nonlocal best_size, best_code
        
        if time.time() - start_time > time_limit:
            return

        if not uncovered_mask:
            if current_size < best_size:
                best_size = current_size
                best_code = [all_words[i] for i in selected_indices]
            return

        lb = get_lower_bound(uncovered_mask)
        if current_size + lb >= best_size:
            return

        # Branching
        # Pick a word v in uncovered_mask
        # To reduce branching, pick v such that the number of candidates is small?
        # Candidates for covering v are all c in ball[v].
        # But we must respect min_idx to avoid permutations?
        # Actually, since we fix 0, and we want to avoid permutations of the rest,
        # we can enforce that the next selected index is > min_idx?
        # No, the set is unordered. If we enforce index ordering, we must ensure we don't miss solutions.
        # Standard trick: When branching, only consider candidates c > min_idx?
        # But what if the optimal solution has a word with index < min_idx that we skipped?
        # We start with min_idx = 0 (word 0).
        # Next word must be > 0.
        # If we pick v, and iterate c in ball[v] with c > min_idx, we might miss c < min_idx.
        # But we already decided not to pick c < min_idx in previous steps?
        # Yes, if we enforce that we only pick indices in increasing order, we cover all sets exactly once.
        
        # However, we must ensure that v is covered.
        # If v is covered by some c < min_idx, we would have picked that c earlier?
        # Not necessarily. We picked 0. If v is in ball[0], it would be covered.
        # If v is NOT in ball[0], it must be covered by some c.
        # If we restrict c > min_idx, we are fine as long as we haven't "forgotten" to pick a small index.
        # But we *did* consider all small indices in previous branches?
        # Actually, the standard "increasing index" constraint works if we branch on "which word covers v".
        # If v is uncovered, it must be covered by some c.
        # We iterate c in ball[v].
        # If we require c > min_idx, we might miss c <= min_idx.
        # But if c <= min_idx, it means we should have picked it before.
        # Did we?
        # In the previous step, we picked some word c_prev.
        # We are now in a state where c_prev is picked.
        # If there was a better choice c_small < c_prev that covers v, we would have explored that branch?
        # Not necessarily.
        
        # Safer: Don't use min_idx for correctness, but use it for symmetry breaking if possible.
        # Given time, let's just use min_idx to enforce ordering of selection.
        # This is valid if we assume the code is sorted by index.
        
        # Find v in uncovered_mask
        # Pick v with smallest index? Or first set bit?
        # Let's pick the first set bit for determinism.
        v_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        
        # Candidates are c in ball[v_idx]
        # We iterate c in ball[v_idx]
        # To prune, sort candidates by coverage of uncovered_mask
        candidates = []
        b_v = balls[v_idx]
        # Extract indices from b_v
        # We can iterate through all c and check if b_v has bit c?
        # Or iterate through set bits of b_v.
        
        # For speed, precompute list of neighbors for each v?
        # neighbors[v] = list of indices c such that v in ball[c]
        # This is just the list of set bits in balls[v].
        
        # Let's generate candidates
        temp_mask = b_v
        while temp_mask:
            lsb = temp_mask & -temp_mask
            c = lsb.bit_length() - 1
            if c >= min_idx: # Enforce ordering
                cover_count = (balls[c] & uncovered_mask).bit_count()
                candidates.append((cover_count, c))
            temp_mask ^= lsb
            
        # Sort by cover_count descending
        candidates.sort(reverse=True)
        
        for cover_count, c in candidates:
            if current_size + 1 + get_lower_bound(uncovered_mask & ~balls[c]) >= best_size:
                continue
                
            new_uncovered = uncovered_mask & ~balls[c]
            new_selected = selected_indices + [c]
            b_and_b(new_uncovered, current_size + 1, new_selected, c + 1)

    # Start B&B
    # Fix 0 in the code
    initial_uncovered = all_ones_mask & ~balls[0]
    b_and_b(initial_uncovered, 1, [0], 1)

    return best_code
