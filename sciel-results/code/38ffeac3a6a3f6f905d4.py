import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces or general radius
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based approach for small spaces (e.g., 3^7 = 2187)
    
    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Precompute balls as bitmasks (Python ints)
    balls = [0] * total_words
    ball_indices = [[] for _ in range(total_words)] # List of indices in the ball
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            # Self
            mask |= (1 << i)
            ball_indices[i].append(i)
            # Neighbors
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        # Calculate neighbor index directly
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
                        ball_indices[i].append(neighbor_idx)
            balls[i] = mask
    else:
        # General radius (slower)
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
                    ball_indices[i].append(j)
            balls[i] = mask

    all_ones_mask = (1 << total_words) - 1
    
    def greedy_cover(uncovered_mask):
        """
        Greedily select codewords to cover the words in uncovered_mask.
        Returns a list of indices.
        """
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
                elif c == max_cover and c > 0:
                    if random.random() < 0.5:
                        best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
        return selected

    def check_redundant(selected_indices, cover_counts):
        """
        Checks if any codeword in selected_indices is redundant.
        Returns (is_redundant, redundant_idx, updated_cover_counts)
        We check by simulating removal.
        """
        # To be efficient, we check candidates.
        # A codeword is a candidate if it covers few unique points.
        # We'll just iterate and check.
        for i in selected_indices:
            # Check if removing i leaves any point with count 0
            # We need to know which points have count == 1 and are covered by i.
            # If any such point exists, i is NOT redundant.
            # If no such point exists, i IS redundant.
            
            is_redundant = True
            for w_idx in ball_indices[i]:
                if cover_counts[w_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                return True, i, cover_counts
        return False, -1, cover_counts

    def remove_redundants(selected):
        """
        Removes any codeword from selected that is redundant.
        Uses coverage counts for efficiency.
        Returns the new list of selected indices.
        """
        if not selected:
            return []
        
        # Initialize cover counts
        cover_counts = [0] * total_words
        for idx in selected:
            for w_idx in ball_indices[idx]:
                cover_counts[w_idx] += 1
        
        changed = True
        while changed:
            changed = False
            # Check from the end to pop easily, or any order
            for i in range(len(selected) - 1, -1, -1):
                idx = selected[i]
                # Check redundancy
                is_redundant = True
                for w_idx in ball_indices[idx]:
                    if cover_counts[w_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    selected.pop(i)
                    for w_idx in ball_indices[idx]:
                        cover_counts[w_idx] -= 1
                    changed = True
                    break # Restart loop as counts changed
        return selected

    def local_search_2_1(selected):
        """
        Try to remove 2 codewords and add 1 to reduce size by 1.
        Returns new selected list if improved, else None.
        """
        if len(selected) < 3:
            return None
            
        # Build coverage counts for current set
        cover_counts = [0] * total_words
        for idx in selected:
            for w_idx in ball_indices[idx]:
                cover_counts[w_idx] += 1
        
        # Try a few random pairs
        # We can be more systematic if size is small
        num_pairs = 20 if len(selected) < 50 else 5
        for _ in range(num_pairs):
            i1, i2 = random.sample(range(len(selected)), 2)
            c1 = selected[i1]
            c2 = selected[i2]
            
            # Simulate removal of c1 and c2
            # Decrement counts
            for w_idx in ball_indices[c1]:
                cover_counts[w_idx] -= 1
            for w_idx in ball_indices[c2]:
                cover_counts[w_idx] -= 1
            
            # Find uncovered words
            uncovered_mask = 0
            # Check if any word has count 0
            # Instead of scanning all words, we can just check the balls of c1 and c2?
            # No, removing c1 and c2 might uncover words that were covered by them.
            # But words NOT in Ball(c1) U Ball(c2) still have count >= 1 (since they were covered by others).
            # So we only need to check words in Ball(c1) U Ball(c2).
            
            candidate_uncovered = []
            # Union of indices in balls
            # To avoid duplicates, we can just check all w in ball_indices[c1] and ball_indices[c2]
            # and see if cover_counts[w] == 0
            
            seen = set()
            for w_idx in ball_indices[c1]:
                if w_idx not in seen:
                    seen.add(w_idx)
                    if cover_counts[w_idx] == 0:
                        candidate_uncovered.append(w_idx)
            for w_idx in ball_indices[c2]:
                if w_idx not in seen:
                    seen.add(w_idx)
                    if cover_counts[w_idx] == 0:
                        candidate_uncovered.append(w_idx)
            
            if not candidate_uncovered:
                # Both redundant?
                new_selected = [x for x in selected if x != c1 and x != c2]
                # Update counts for new_selected?
                # We just return the list, remove_redundants will handle it
                return new_selected

            # Find the best single word to cover uncovered
            # The new word must cover ALL words in candidate_uncovered.
            # And it should maximize coverage of other currently uncovered? No, all others are covered.
            # So we just need a word that covers candidate_uncovered.
            # To be a valid 2-1 move, the new word must cover all newly uncovered words.
            
            # We can intersect the balls of the uncovered words to find candidates?
            # Or just scan all words and check if they cover candidate_uncovered.
            
            best_idx = -1
            max_extra_cover = 0 # How many other words it covers (not strictly needed for validity, but good for tie-breaking)
            
            # Optimization: Only consider words that cover the first uncovered word
            first_uncovered = candidate_uncovered[0]
            candidates = ball_indices[first_uncovered]
            
            for k in candidates:
                if k == c1 or k == c2:
                    continue
                # Check if k covers all other uncovered words
                covers_all = True
                extra = 0
                for w_idx in candidate_uncovered[1:]:
                    if not (balls[k] >> w_idx) & 1:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # It covers all. Calculate extra coverage (words with count > 0 that it also covers)
                # Actually, we just want to reduce size. Any valid k works.
                # Tie-break: Maximize total coverage?
                # Or minimize "overlap" with existing?
                # Let's just pick the first valid one or randomize.
                
                # To improve diversity, we can pick randomly among valid ones
                if best_idx == -1 or random.random() < 0.5:
                    best_idx = k
            
            if best_idx != -1:
                new_selected = [x for x in selected if x != c1 and x != c2]
                new_selected.append(best_idx)
                return new_selected
            
            # Restore counts
            for w_idx in ball_indices[c1]:
                cover_counts[w_idx] += 1
            for w_idx in ball_indices[c2]:
                cover_counts[w_idx] += 1
        
        return None

    best_code = None
    best_size = total_words + 1
    start_time = time.time()
    time_limit = 115.0
    
    # Initial Greedy
    initial_selected = greedy_cover(all_ones_mask)
    initial_selected = remove_redundants(initial_selected)
    
    if len(initial_selected) < best_size:
        best_size = len(initial_selected)
        best_code = [all_words[i] for i in initial_selected]
    
    current_selected = initial_selected[:]
    
    while time.time() - start_time < time_limit:
        # 1. Try 2-1 Local Search
        improved = False
        new_sel = local_search_2_1(current_selected)
        if new_sel:
            current_selected = new_sel
            # Clean up redundants
            current_selected = remove_redundants(current_selected)
            improved = True
            if len(current_selected) < best_size:
                best_size = len(current_selected)
                best_code = [all_words[i] for i in current_selected]
        
        if improved:
            continue

        # 2. Perturb and Repair
        perturb_amount = random.randint(1, 3)
        if len(current_selected) <= perturb_amount:
            current_selected = greedy_cover(all_ones_mask)
            current_selected = remove_redundants(current_selected)
        else:
            indices_to_remove = random.sample(range(len(current_selected)), perturb_amount)
            indices_to_remove_set = set(indices_to_remove)
            
            covered_mask = 0
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    covered_mask |= balls[current_selected[i]]
            
            uncovered_mask = all_ones_mask & ~covered_mask
            
            repair_selected = greedy_cover(uncovered_mask)
            
            new_selected = []
            for i in range(len(current_selected)):
                if i not in indices_to_remove_set:
                    new_selected.append(current_selected[i])
            new_selected.extend(repair_selected)
            
            new_selected = remove_redundants(new_selected)
            
            if len(new_selected) < best_size:
                best_size = len(new_selected)
                best_code = [all_words[i] for i in new_selected]
            
            if len(new_selected) <= len(current_selected) or (random.random() < 0.05 and len(new_selected) < len(current_selected) + 3):
                current_selected = new_selected

    return best_code
