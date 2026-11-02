import itertools
import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Generate the standard superpermutation
    # This is the recursive construction: L(n) = n! + (n-1)! + ... + 1!
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        # Extract permutations of length m-1 from the current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with the end of new_word
            overlap = 0
            max_overlap = min(len(new_word), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # For n <= 4, this is optimal, no need to search
    if n <= 4:
        return "".join(map(str, word))
    
    # 2. Local Search: Greedy Deletion with Multi-Order Passes
    # Map permutations to IDs for fast counting
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    def get_counts(w):
        counts = [0] * num_perms
        L = len(w)
        for i in range(L - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_id:
                counts[perm_to_id[p]] += 1
        return counts
    
    # Initial counts
    counts = get_counts(word)
    
    # Verify initial validity
    if any(c == 0 for c in counts):
        # Fallback to standard if something went wrong
        return "".join(map(str, word))
    
    # Helper to check if a position can be removed
    def can_remove(w, counts, pos, n, perm_to_id):
        L = len(w)
        j_min = max(0, pos - n + 1)
        j_max = min(pos, L - n)
        
        affected = {}
        for j in range(j_min, j_max + 1):
            p = tuple(w[j:j+n])
            if p in perm_to_id:
                pid = perm_to_id[p]
                affected[pid] = affected.get(pid, 0) + 1
        
        for pid, cnt in affected.items():
            if counts[pid] - cnt < 1:
                return False, affected
        return True, affected
    
    # Greedy removal pass with a specific order
    def greedy_remove_pass(w, counts, n, perm_to_id, order="fwd"):
        removals = 0
        if order == "fwd":
            i = 0
            while i < len(w):
                if time.time() > deadline:
                    break
                ok, affected = can_remove(w, counts, i, n, perm_to_id)
                if ok:
                    # Update counts
                    for pid, cnt in affected.items():
                        counts[pid] -= cnt
                    w.pop(i)
                    removals += 1
                else:
                    i += 1
        elif order == "bwd":
            i = len(w) - 1
            while i >= 0:
                if time.time() > deadline:
                    break
                ok, affected = can_remove(w, counts, i, n, perm_to_id)
                if ok:
                    for pid, cnt in affected.items():
                        counts[pid] -= cnt
                    w.pop(i)
                    removals += 1
                    i -= 1
                else:
                    i -= 1
        elif order == "random":
            # Random shuffle of indices
            indices = list(range(len(w)))
            random.shuffle(indices)
            # Note: indices become invalid after pops. 
            # To handle this correctly in random order, we'd need a more complex structure 
            # or just re-derive indices. For simplicity and speed, we'll do a few random 
            # "sweeps" where we pick a random index, try to remove, and if successful, 
            # we don't increment. If not, we increment. 
            # Actually, the simplest robust random approach is to just pick a random index 
            # from the current valid range.
            # Let's do a fixed number of random attempts per pass to avoid infinite loops 
            # if stuck, or just iterate until no removals in a full random permutation.
            # A full random permutation of indices is O(L) to generate, which is fine.
            
            # However, since indices shift, a pre-generated list of indices is invalid.
            # We will use a "random walk" style: pick random index, try remove.
            # To ensure we don't miss easy removals, we'll do a pass that tries each 
            # potential index (0 to L-1) in a random order, but checking validity 
            # against the *current* list length.
            
            # Efficient way: 
            # 1. Create a list of current indices
            # 2. Shuffle
            # 3. Iterate through shuffled list, but only if index is still valid (0 <= idx < len(w))
            #    This works because if we pop, indices > pos shift down, so they might 
            #    still be valid for the *next* item in the shuffled list if it was > pos.
            #    Wait, if we pop at pos, index `pos` is gone. Indices < pos are same. 
            #    Indices > pos become index-1.
            #    So if we have a shuffled list [5, 2, 8], and we pop at 2, the 8 in the 
            #    list now refers to the element that was at 9. The element at 8 is now at 7.
            #    This makes "random order" tricky to implement correctly with a pre-shuffled list.
            
            # Alternative: Just do a few rounds of "try random index".
            # But to be thorough, let's stick to fwd and bwd which are robust.
            # Or, we can implement a "random" pass by just picking a random index 
            # repeatedly until no progress is made in a full cycle of len(w) attempts.
            
            # Let's implement a simpler "random" pass:
            # Try to remove at random positions. If we make no removals after len(w) attempts, stop.
            attempts = 0
            max_attempts = len(w) * 2 # Safety
            while attempts < max_attempts and len(w) > 0:
                pos = random.randint(0, len(w) - 1)
                ok, affected = can_remove(w, counts, pos, n, perm_to_id)
                if ok:
                    for pid, cnt in affected.items():
                        counts[pid] -= cnt
                    w.pop(pos)
                    removals += 1
                    attempts = 0 # Reset attempts on success
                else:
                    attempts += 1
        return removals

    improved = list(word)
    counts = get_counts(improved)
    
    # Phase 1: Standard greedy removal passes (fwd, bwd, random)
    # We loop until no removals are found in a full cycle of all orders
    while time.time() < deadline - 5: # Leave buffer for final check
        changed = False
        for order in ["fwd", "bwd", "random"]:
            if time.time() > deadline - 5:
                break
            r = greedy_remove_pass(improved, counts, n, perm_to_id, order)
            if r > 0:
                changed = True
        if not changed:
            break
    
    if time.time() > deadline - 5:
        return "".join(map(str, improved))
    
    # Phase 2: Swap-based local search
    # Try swapping adjacent pairs to enable more removals
    # We'll do multiple rounds of: try all adjacent swaps, keep best improvement
    
    # Helper to verify if a word is a valid superpermutation
    def is_valid_superpermutation(w, n, perm_to_id):
        seen = set()
        L = len(w)
        for i in range(L - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_id:
                seen.add(perm_to_id[p])
                if len(seen) == num_perms:
                    return True
        return False
    
    # To make swap search faster, we'll only accept a swap if it leads to a shorter 
    # word after greedy removal, and we verify validity at the end.
    
    best_len = len(improved)
    best_word = improved[:]
    best_counts = counts[:] # Copy
    
    # We will iterate through potential swaps. 
    # Since L is ~6000, L^2 is too big. We only do adjacent swaps.
    # For each adjacent swap, we do a quick greedy removal to see if it helps.
    
    # To avoid getting stuck, we'll do multiple passes over the word.
    # In each pass, we try to improve the current word.
    
    # Optimization: Instead of full greedy removal for every swap (which is O(L*n)),
    # we can do a limited depth search or just accept if the swap itself allows a removal.
    # But the swap itself doesn't remove anything. It changes the structure.
    # Let's do: Swap -> Greedy Removal (1 pass fwd) -> If shorter, accept.
    
    # Time budget for swap search
    swap_deadline = deadline - 10
    last_improvement_time = time.time()
    
    # We'll do a hill-climbing approach.
    # Current state: improved, counts
    # We try swaps in a random order or sequential order.
    # If a swap + 1 greedy pass leads to a shorter word, we accept it and restart scan.
    
    # To prevent infinite loops on same-length states, we only accept if strictly shorter.
    
    # Let's use a more aggressive strategy:
    # 1. Try a random adjacent swap.
    # 2. Do 1 fwd greedy pass.
    # 3. If length decreased, accept new state.
    # 4. If no improvement after X attempts, try a different random seed or strategy.
    
    # Actually, sequential scan with acceptance is more deterministic and often effective.
    # Let's do sequential scan for swaps, accepting any improvement.
    
    current_word = improved
    current_counts = counts
    
    while time.time() < swap_deadline:
        improved_this_pass = False
        L = len(current_word)
        # Try swaps in a random order to avoid bias? 
        # Or sequential? Sequential is O(L). Random is O(L) if we pick L random swaps.
        # Let's do sequential first.
        
        for i in range(L - 1):
            if time.time() > swap_deadline:
                break
            
            # Skip if time is tight
            if time.time() > swap_deadline - 0.1:
                break
                
            # Perform swap
            current_word[i], current_word[i+1] = current_word[i+1], current_word[i]
            
            # Rebuild counts (expensive, but necessary for correctness of greedy removal)
            # Optimization: Only rebuild counts if the swap actually changes the set of windows?
            # It always changes windows i-n+1 to i+1.
            # But for simplicity, let's rebuild.
            # Wait, rebuilding counts is O(L). Doing this for L swaps is O(L^2).
            # For L=6000, L^2 = 36M. In Python, this might be too slow for 100s.
            # 36M operations is ~1-2 seconds in C, but Python is slower. Maybe 10-20s?
            # Risky.
            
            # Alternative: Don't do full greedy removal for every swap.
            # Just check if the swap allows *immediate* removals?
            # Or, use a faster validity check.
            
            # Let's try a different approach: 
            # Only do swap search if we have plenty of time.
            # And use a faster method: 
            # Instead of full greedy removal, just do 1 pass of "try to remove at i, i+1, i+2".
            # This is heuristic.
            
            # Let's stick to the safer, but potentially slower, full greedy removal 
            # but limit the number of swap attempts.
            
            # Actually, let's just do the swap, rebuild counts, and do ONE fwd greedy pass.
            # If it's too slow, we'll break out.
            
            new_counts = get_counts(current_word)
            
            # Do 1 fwd greedy pass
            test_word = current_word
            test_counts = new_counts
            r = greedy_remove_pass(test_word, test_counts, n, perm_to_id, "fwd")
            
            if len(test_word) < len(current_word):
                # Accept improvement
                current_word = test_word
                current_counts = test_counts
                improved_this_pass = True
                # Restart scan from beginning as indices shifted
                break
            else:
                # Undo swap
                current_word[i], current_word[i+1] = current_word[i+1], current_word[i]
        
        if not improved_this_pass:
            # No improvement in this full pass. 
            # We could try random swaps now, or stop.
            # Let's try a few random swaps to escape local minima.
            random_improved = False
            for _ in range(50): # 50 random swap attempts
                if time.time() > swap_deadline:
                    break
                if len(current_word) < 2:
                    break
                i = random.randint(0, len(current_word) - 2)
                current_word[i], current_word[i+1] = current_word[i+1], current_word[i]
                new_counts = get_counts(current_word)
                test_word = current_word
                test_counts = new_counts
                r = greedy_remove_pass(test_word, test_counts, n, perm_to_id, "fwd")
                if len(test_word) < len(current_word):
                    current_word = test_word
                    current_counts = test_counts
                    random_improved = True
                    break
                else:
                    current_word[i], current_word[i+1] = current_word[i+1], current_word[i]
            
            if not random_improved:
                # Stuck. Stop swap search.
                break
        
        # Update best
        if len(current_word) < best_len:
            best_len = len(current_word)
            best_word = current_word[:]
            best_counts = current_counts[:]

    # Final verification and return
    # Ensure we return a valid superpermutation
    # The greedy removal logic ensures validity, but let's double check if time permits
    if time.time() < deadline - 2:
        if not is_valid_superpermutation(best_word, n, perm_to_id):
            # Fallback to original word if best_word is invalid (should not happen)
            return "".join(map(str, word))
    
    return "".join(map(str, best_word))
