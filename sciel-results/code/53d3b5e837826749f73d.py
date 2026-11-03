import itertools
import time
import random

def construct(q, n, radius):
    # If parameters don't match the specific optimized case, fallback to seed logic
    if not (q == 3 and n == 6 and radius == 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Fallback for small n
            return [list(tail) for tail in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 110
    
    # Precomputation
    all_words = [list(itertools.product(range(q), repeat=n))]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (distance <= 1)
    neighbors = []
    for w in all_words:
        nb = [w]
        for i in range(n):
            for val in range(q):
                if val == w[i]: continue
                new_w = w.copy()
                new_w[i] = val
                nb.append(tuple(new_w))
        neighbors.append([word_to_idx[tnb] for tnb in nb])
        
    # Convert neighbors to sets for faster "in" check if needed, 
    # but list is fine for size 13.
    
    # Initialize code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    code_indices = set(word_to_idx[c] for c in initial_code)
    
    cover_count = [0] * (q**n)
    for c_idx in code_indices:
        for nb_idx in neighbors[c_idx]:
            cover_count[nb_idx] += 1
            
    def is_redundant(c_idx):
        for nb_idx in neighbors[c_idx]:
            if cover_count[nb_idx] == 1:
                return False
        return True

    def remove_c(c_idx):
        code_indices.remove(c_idx)
        for nb_idx in neighbors[c_idx]:
            cover_count[nb_idx] -= 1

    def add_c(c_idx):
        code_indices.add(c_idx)
        for nb_idx in neighbors[c_idx]:
            cover_count[nb_idx] += 1

    random.seed(42) # Deterministic
    
    while time.time() < deadline:
        removed_any = False
        # Iterate over a copy because we modify code_indices
        for c_idx in list(code_indices):
            if is_redundant(c_idx):
                remove_c(c_idx)
                removed_any = True
        
        if not removed_any:
            swapped = False
            # Try swaps
            # To avoid getting stuck in a local minimum of "no valid swaps",
            # we can try to move a codeword to a random position.
            # Or try to move it to a position that covers critical words.
            
            # Let's try a few random swaps
            for _ in range(200):
                if time.time() > deadline: break
                if not code_indices: break
                
                c_idx = random.choice(list(code_indices))
                c_tuple = all_words[c_idx]
                new_c_tuple = list(c_tuple)
                
                # Perturb
                num_changes = random.choice([1, 2])
                for _ in range(num_changes):
                    pos = random.randint(0, n-1)
                    old_val = new_c_tuple[pos]
                    new_val = random.choice([v for v in range(q) if v != old_val])
                    new_c_tuple[pos] = new_val
                    
                new_c_idx = word_to_idx[tuple(new_c_tuple)]
                
                if new_c_idx == c_idx or new_c_idx in code_indices:
                    continue
                
                # Check validity
                critical_words = []
                for nb_idx in neighbors[c_idx]:
                    if cover_count[nb_idx] == 1:
                        critical_words.append(nb_idx)
                        
                if not critical_words:
                    continue # Shouldn't happen if pruning worked
                
                valid_swap = True
                for w_idx in critical_words:
                    if w_idx not in neighbors[new_c_idx]:
                        valid_swap = False
                        break
                        
                if valid_swap:
                    remove_c(c_idx)
                    add_c(new_c_idx)
                    swapped = True
                    break
                    
            if not swapped:
                # Optional: Try a "kick" - remove a random codeword, 
                # find best replacement to cover uncovered words, add it, prune.
                # This allows escaping local minima where no single swap works.
                
                # Identify a codeword to "kick"
                # Maybe the one with lowest "redundancy" (most critical words)?
                # Or just random.
                c_idx = random.choice(list(code_indices))
                
                # Remove it
                remove_c(c_idx)
                
                # Find uncovered words
                uncovered = [i for i in range(q**n) if cover_count[i] == 0]
                
                if not uncovered:
                    # It was redundant, but we didn't catch it? 
                    # Or we just removed it and it's fine.
                    # Prune again in next loop.
                    pass
                else:
                    # Find a word that covers the most uncovered words
                    best_c = -1
                    best_cover = -1
                    # We can't check all 729 words against all uncovered efficiently?
                    # 729 * |uncovered|. |uncovered| is small usually.
                    # Let's try all words.
                    for cand_idx in range(q**n):
                        if cand_idx in code_indices: continue
                        # Count how many uncovered words are in N(cand_idx)
                        # We can just check if cand_idx covers all uncovered?
                        # No, we want to maximize coverage to allow pruning.
                        # Actually, if we can cover ALL uncovered words, we restore validity.
                        # If we can't cover all, the code is invalid.
                        # So we MUST find a cand_idx that covers all uncovered words.
                        
                        covers_all = True
                        for u in uncovered:
                            if u not in neighbors[cand_idx]:
                                covers_all = False
                                break
                            
                        if covers_all:
                            # This is a valid replacement
                            add_c(cand_idx)
                            swapped = True # Treat as a swap/kick
                            break
                            
                    if not swapped:
                        # No single word covers all uncovered words.
                        # This means the removal of c_idx created a "hole" 
                        # that cannot be filled by a single codeword.
                        # We must restore c_idx.
                        add_c(c_idx)
                        
    return [list(all_words[c_idx]) for c_idx in code_indices]
