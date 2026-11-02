import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all unique (m-1)-permutations from the current word
        # The standard construction ensures that all (m-1)! permutations appear
        # as consecutive (m-1) symbols.
        # We need to identify their order of appearance to build the next layer.
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Insert m after each permutation, overlapping as much as possible
            # Standard form: p, m, p
            block = list(p) + [m] + list(p)
            # Find max overlap with current end of new
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 5:
        return "".join(map(str, word))

    # 2. Optimization Phase
    current_word = list(word)
    L = len(current_word)
    
    # Map permutations to indices
    perm_map = {}
    idx = 0
    for p in permutations(range(1, n + 1)):
        perm_map[p] = idx
        idx += 1
    target_count = idx  # n!

    # Build initial window indices
    widx = [0] * (L - n + 1)
    for i in range(L - n + 1):
        widx[i] = perm_map[tuple(current_word[i:i + n])]

    # Counts of each permutation
    counts = [0] * target_count
    for v in widx:
        counts[v] += 1
        
    # Set of missing permutations (should be empty initially)
    missing = set()
    for i in range(target_count):
        if counts[i] == 0:
            missing.add(i)

    # Helper to check if a tuple is a permutation
    # Since n is small, we can just check len(set(t)) == n
    # But we need the index. If it's not a permutation, index is -1.
    def get_idx(tup):
        return perm_map.get(tup, -1)

    improved = True
    pass_count = 0
    max_passes = 1000 # Safety
    
    while improved and time.time() < deadline - 5 and pass_count < max_passes:
        improved = False
        pass_count += 1
        L = len(current_word)
        
        # Try to delete positions in random order
        # We can also prioritize positions that are "redundant"
        # For now, random is a good baseline
        positions = list(range(L))
        random.shuffle(positions)
        
        for pos in positions:
            if time.time() > deadline - 5:
                break
            
            L = len(current_word)
            if L <= n:
                break
            
            # Identify old windows to remove
            # Windows covering pos: start in [pos-n+1, pos]
            old_start_min = max(0, pos - n + 1)
            old_start_max = min(pos, L - n)
            
            removed_indices = []
            valid = True
            
            # Decrement counts for old windows
            for s in range(old_start_min, old_start_max + 1):
                v = widx[s]
                if v >= 0:
                    counts[v] -= 1
                    if counts[v] == 0:
                        missing.add(v)
                        removed_indices.append((s, v))
                    # If count was 1 and becomes 0, it's missing.
                    # We track these to potentially revert.
            
            # If any permutation became missing, we need to see if new windows restore it.
            # Compute new windows
            # New word length L-1
            # New windows crossing the gap:
            # Start s_new in [max(0, pos-n+1), min(pos-1, L-n-1)]
            # Note: In the new word, the gap is at pos.
            # Indices < pos are same. Indices >= pos are shifted -1.
            
            new_window_indices = []
            for s_new in range(max(0, pos - n + 1), min(pos, L - n)):
                # Construct tuple from current_word (before deletion)
                # The window in new word at s_new corresponds to:
                # current_word[s_new : pos] + current_word[pos+1 : s_new+n]
                # Length is n.
                # Let's verify:
                # If s_new < pos:
                # Part 1: current_word[s_new:pos] (length pos - s_new)
                # Part 2: current_word[pos+1 : s_new+n] (length s_new+n - (pos+1) = s_new - pos + n - 1)
                # Total length: pos - s_new + s_new - pos + n - 1 = n - 1 ??
                # Wait.
                # New word index s_new maps to old:
                # If s_new < pos, old index is s_new.
                # The window covers new indices s_new ... s_new+n-1.
                # If s_new+n-1 < pos, it's entirely before gap. (Unchanged, not in this range)
                # If s_new < pos <= s_new+n-1, it crosses.
                # The old indices are:
                # s_new ... pos-1 (from current_word)
                # pos+1 ... s_new+n-1+1 = s_new+n (from current_word, shifted)
                # So tuple is current_word[s_new:pos] + current_word[pos+1:s_new+n]
                
                if s_new < pos:
                    part1 = current_word[s_new:pos]
                    part2 = current_word[pos+1:s_new+n]
                    tup = tuple(part1 + part2)
                else:
                    # s_new >= pos? 
                    # Range was min(pos, ...). So s_new can be pos?
                    # If s_new == pos, in new word, it starts after the gap.
                    # Old index is pos+1.
                    # Window: current_word[pos+1 : pos+1+n]
                    # This is just a shift of old window at pos+1.
                    # But wait, if s_new == pos, the window is current_word[pos+1:pos+1+n].
                    # This window existed in the old word at start pos+1.
                    # It was NOT removed (unless pos+1 <= pos, which is false).
                    # So we don't need to "add" it back, it was never removed.
                    # The windows that are "new" are those that *cross* the gap.
                    # These are s_new such that s_new < pos and s_new+n-1 >= pos.
                    # i.e., s_new >= pos - n + 1 and s_new < pos.
                    # So the range is [max(0, pos-n+1), pos-1].
                    # My previous range `min(pos, L-n)` included `pos`.
                    # If s_new == pos, the window is old window at pos+1. It was not removed.
                    # So we should only iterate s_new in [max(0, pos-n+1), pos-1].
                    
                    # Let's correct the loop range.
                    pass

            # Corrected loop for new windows:
            # s_new from max(0, pos-n+1) to pos-1
            # But s_new must be a valid start in new word: s_new <= L-1-n
            new_start_min = max(0, pos - n + 1)
            new_start_max = min(pos - 1, L - 1 - n)
            
            for s_new in range(new_start_min, new_start_max + 1):
                part1 = current_word[s_new:pos]
                part2 = current_word[pos+1:s_new+n]
                tup = tuple(part1 + part2)
                v = get_idx(tup)
                if v >= 0:
                    if v in missing:
                        missing.discard(v)
                        counts[v] += 1
                        new_window_indices.append((s_new, v))
                    else:
                        counts[v] += 1
                        new_window_indices.append((s_new, v))
                else:
                    # Not a permutation. This is fine, it just doesn't contribute to coverage.
                    # But we must be careful: did we remove a valid window and replace it with an invalid one?
                    # Yes, that's allowed as long as we don't lose a *unique* permutation.
                    pass

            if len(missing) == 0:
                # Valid deletion
                # Perform deletion
                current_word.pop(pos)
                
                # Update widx
                # Old widx length L-n+1
                # New widx length L-n
                # Windows 0 to old_start_min-1 are unchanged.
                # Windows old_start_min to old_start_max are removed/changed.
                # Windows after old_start_max are shifted left by 1.
                
                # Let's rebuild widx for the affected range.
                # New length: L-1-n+1 = L-n
                new_widx = [0] * (L - n)
                
                # Copy unchanged prefix
                if old_start_min > 0:
                    new_widx[:old_start_min] = widx[:old_start_min]
                
                # Compute new windows in the affected range
                # These correspond to s_new in [new_start_min, new_start_max]
                # But also windows that were "shifted" and are now in the affected range?
                # Actually, any window in the new word that *doesn't* cross the gap is identical to an old window.
                # Windows crossing the gap are the new ones.
                # Windows before the gap: s_new < pos-n+1. Unchanged.
                # Windows after the gap: s_new >= pos. 
                #   In new word, s_new >= pos means old index s_new+1 >= pos+1.
                #   These windows were at old index s_new+1.
                #   They were not removed (since removed were <= pos).
                #   So they are just shifted.
                
                # So:
                # 1. Prefix: s_new in [0, pos-n] (if pos-n >= 0)
                #    These are same as old widx[0 : pos-n+1]
                #    Wait, old widx index s is same as new widx index s if s < pos-n+1.
                #    So new_widx[s] = widx[s] for s in [0, pos-n]
                
                # 2. Crossing: s_new in [pos-n+1, pos-1]
                #    We computed these in the loop above.
                #    We need to store them.
                
                # 3. Suffix: s_new in [pos, L-n-1]
                #    new_widx[s_new] = widx[s_new+1]
                
                # Let's implement this update.
                # We already have `new_window_indices` from the validity check, but that only stored valid permutations.
                # We need the indices for ALL windows in the crossing range, even if -1.
                # So we must recompute or store them properly.
                
                # Recompute crossing windows for widx update
                for s_new in range(new_start_min, new_start_max + 1):
                    part1 = current_word[s_new:pos] # Note: current_word is already updated!
                    # Wait, I popped pos. So current_word is the NEW word.
                    # So I can just do:
                    tup = tuple(current_word[s_new:s_new+n])
                    new_widx[s_new] = get_idx(tup)
                
                # Copy suffix
                if new_start_max + 1 < L - n:
                    # Suffix in new word starts at new_start_max + 1
                    # Corresponds to old widx starting at new_start_max + 2 ?
                    # Let's check indices.
                    # New index s_new. Old index s_old.
                    # If s_new >= pos, s_old = s_new + 1.
                    # So new_widx[s_new] = widx[s_new + 1].
                    # Range of s_new: [pos, L-n-1]
                    # Range of s_old: [pos+1, L-n]
                    # We can slice:
                    suffix_start_new = pos
                    suffix_end_new = L - n
                    suffix_start_old = pos + 1
                    suffix_end_old = L - n + 1
                    
                    new_widx[suffix_start_new:suffix_end_new] = widx[suffix_start_old:suffix_end_old]

                widx = new_widx
                improved = True
                break # Restart pass with new length
            
            else:
                # Invalid deletion, revert counts
                for s, v in removed_indices:
                    counts[v] += 1
                    # If it was missing, it's no longer missing
                    if v in missing:
                        missing.discard(v)
                for s_new, v in new_window_indices:
                    counts[v] -= 1
                    if counts[v] == 0:
                        missing.add(v)

    return "".join(map(str, current_word))
