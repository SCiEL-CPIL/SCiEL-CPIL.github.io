"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    """
    Constructs the shortest superpermutation for n symbols.
    For n=7, uses the permutation-ordering (TSP-like) approach with hill-climbing
    on the overlap matrix, followed by character-level local search.
    For n<=6, uses the standard recursive construction.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    if n == 1:
        return "1"
    
    if n <= 6:
        word = _build_recursive(n)
        return "".join(map(str, word))
    
    if n == 7:
        word = _construct_n7(deadline)
        return "".join(map(str, word))
    
    # Fallback for n > 7: standard recursive construction
    word = _build_recursive(n)
    return "".join(map(str, word))


def _build_recursive(n):
    """
    Builds the standard recursive superpermutation:
    SP(1) = "1"
    SP(m) = concat over all (m-1)-perms p in order of appearance in SP(m-1):
             p + [m] + p
    with maximum overlap between consecutive blocks.
    """
    if n == 1:
        return [1]
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap
            max_possible_overlap = min(len(new_word), len(block), m - 1)
            overlap = 0
            for k in range(max_possible_overlap, -1, -1):
                if new_word[len(new_word) - k:] == block[:k]:
                    overlap = k
                    break
            
            new_word.extend(block[overlap:])
        
        word = new_word
    
    return word


def _construct_n7(deadline):
    """
    Constructs a superpermutation for n=7 using permutation-ordering optimization.
    Step 1: Build SP(6) and extract the 720 permutations in order of appearance.
    Step 2: Compute the overlap matrix M[i][j] = longest suffix of perm i matching prefix of perm j.
    Step 3: Find a good ordering of the 720 permutations via hill-climbing (maximize total overlap).
    Step 4: Construct the word from the best ordering.
    Step 5: Apply character-level local search to remove redundant characters.
    """
    from itertools import permutations
    
    # Step 1: Build SP(6)
    word_6 = [1]
    for m in range(2, 7):
        perms, seen = [], set()
        for i in range(len(word_6) - m + 2):
            window = tuple(word_6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_6 = new
    
    # Extract all 720 permutations of 1..6 in order of appearance in SP(6)
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 1):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
    
    num_perms = len(perms_6)
    
    # Step 2: Compute overlap matrix
    # M[i][j] = longest k such that suffix of perms_6[i] of length k == prefix of perms_6[j] of length k
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
    
    # Step 3: Hill-climbing to find good ordering
    # The total length of the constructed word is:
    #   sum of block lengths - sum of overlaps
    # Each block has length 6 + 1 + 6 = 13
    # Total length = 720 * 13 - sum of overlaps between consecutive permutations
    # So we want to maximize the sum of overlaps.
    
    random.seed(42)
    
    # Start with the natural order (order of appearance in SP(6))
    order = list(range(num_perms))
    
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i + 1]]
        return score
    
    best_order = order[:]
    best_score = calc_score(order)
    
    # Hill-climbing with multiple strategies
    current_order = order[:]
    current_score = best_score
    
    while time.time() < deadline - 30.0:  # Leave 30s for local search
        # Try random moves
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap adjacent
            i = random.randint(0, num_perms - 2)
            new_order = current_order[:]
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            
            # Compute delta score
            old_score = 0
            if i > 0:
                old_score += M[current_order[i - 1]][current_order[i]]
            old_score += M[current_order[i]][current_order[i + 1]]
            if i + 1 < num_perms - 1:
                old_score += M[current_order[i + 1]][current_order[i + 2]]
            
            new_score = 0
            if i > 0:
                new_score += M[new_order[i - 1]][new_order[i]]
            new_score += M[new_order[i]][new_order[i + 1]]
            if i + 1 < num_perms - 1:
                new_score += M[new_order[i + 1]][new_order[i + 2]]
            
            delta = new_score - old_score
            if delta > 0:
                current_order = new_order
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
        elif move_type < 0.7:
            # Move element i to position j
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j:
                continue
            
            new_order = current_order[:]
            item = new_order.pop(i)
            new_order.insert(j, item)
            
            # Compute delta score (only affected positions)
            old_score = 0
            if i > 0:
                old_score += M[current_order[i - 1]][current_order[i]]
            if i < num_perms - 1:
                old_score += M[current_order[i]][current_order[i + 1]]
            if j > 0 and j != i:
                old_score += M[current_order[j - 1]][current_order[j]]
            if j < num_perms - 1 and j + 1 != i:
                old_score += M[current_order[j]][current_order[j + 1]]
            
            new_score = 0
            if i > 0:
                new_score += M[current_order[i - 1]][current_order[i + 1] if i < num_perms - 1 else -1] if i < num_perms - 1 else 0
            if i < num_perms - 1:
                new_score += M[current_order[i]][current_order[i + 1]]
            # This is getting complex; just recalc full score for simplicity when needed
            # For now, use a simpler approach: recalc score for the new order
            full_new_score = calc_score(new_order)
            delta = full_new_score - current_score
            
            if delta > 0:
                current_order = new_order
                current_score = full_new_score
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
        else:
            # 2-opt: reverse a segment
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            new_order = current_order[:]
            new_order[i:j + 1] = new_order[i:j + 1][::-1]
            
            # Compute delta score
            old_score = 0
            if i > 0:
                old_score += M[current_order[i - 1]][current_order[i]]
            old_score += M[current_order[i]][current_order[i + 1]]
            if j < num_perms - 1:
                old_score += M[current_order[j]][current_order[j + 1]]
            
            new_score = 0
            if i > 0:
                new_score += M[new_order[i - 1]][new_order[i]]
            new_score += M[new_order[i]][new_order[i + 1]]
            if j < num_perms - 1:
                new_score += M[new_order[j]][new_order[j + 1]]
            
            delta = new_score - old_score
            if delta > 0:
                current_order = new_order
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
    
    # Step 4: Construct word from best ordering
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            max_ov = min(len(word), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]
    
    # Step 5: Character-level local search
    word = _local_search(word, 7, deadline)
    
    return word


def _local_search(word, n, deadline):
    """
    Local search to remove redundant characters from the superpermutation.
    A character can be removed if no permutation becomes uncovered.
    """
    random.seed(42)
    
    # Build index mapping from permutation tuple to index
    from itertools import permutations
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Build coverage counts
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            idx = perm_to_idx.get(window)
            if idx is not None:
                counts[idx] += 1
    
    best_word = list(word)
    best_len = len(word)
    
    # Multiple passes of removal
    for iteration in range(50000):
        if time.time() > deadline - 5.0:
            break
        
        # Try to remove a random character
        found_removal = False
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            if len(word) <= n:
                break
            
            pos = random.randint(0, len(word) - 1)
            
            # Find affected windows (those that include position pos)
            start_lo = max(0, pos - n + 1)
            start_hi = min(pos, len(word) - n)
            
            if start_lo > start_hi:
                continue
            
            affected_indices = []
            for s in range(start_lo, start_hi + 1):
                window = tuple(word[s:s + n])
                if len(set(window)) == n:
                    idx = perm_to_idx.get(window)
                    if idx is not None:
                        affected_indices.append(idx)
            
            # Check if any affected permutation would lose its last coverage
            safe = True
            for idx in affected_indices:
                if counts[idx] == 1:
                    safe = False
                    break
            
            if safe:
                # Remove the character
                word.pop(pos)
                for idx in affected_indices:
                    counts[idx] -= 1
                found_removal = True
                if len(word) < best_len:
                    best_word = list(word)
                    best_len = len(word)
                break
        
        if not found_removal:
            # Try a different strategy: swap two adjacent characters
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                
                if len(word) < 2:
                    break
                
                pos = random.randint(0, len(word) - 2)
                
                # Compute affected windows before swap
                start_lo = max(0, pos - n + 1)
                start_hi = min(pos + 1, len(word) - n)
                
                if start_lo > start_hi:
                    continue
                
                before_counts = {}
                for s in range(start_lo, start_hi + 1):
                    window = tuple(word[s:s + n])
                    if len(set(window)) == n:
                        idx = perm_to_idx.get(window)
                        if idx is not None:
                            before_counts[idx] = before_counts.get(idx, 0) + 1
                
                # Perform swap
                word[pos], word[pos + 1] = word[pos + 1], word[pos]
                
                # Compute affected windows after swap
                after_counts = {}
                for s in range(start_lo, start_hi + 1):
                    window = tuple(word[s:s + n])
                    if len(set(window)) == n:
                        idx = perm_to_idx.get(window)
                        if idx is not None:
                            after_counts[idx] = after_counts.get(idx, 0) + 1
                
                # Check if swap is safe: no permutation goes from 1 to 0
                safe = True
                for idx in before_counts:
                    if before_counts[idx] == 1 and after_counts.get(idx, 0) == 0:
                        safe = False
                        break
                
                if not safe:
                    # Revert swap
                    word[pos], word[pos + 1] = word[pos + 1], word[pos]
                else:
                    # Update counts
                    for idx in before_counts:
                        counts[idx] -= before_counts[idx]
                    for idx in after_counts:
                        counts[idx] += after_counts[idx]
                    improved = True
                    break
            
            if not improved:
                # No improvement possible in this iteration; break to avoid infinite loop
                # But first, try a few more random removals
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    if len(word) <= n:
                        break
                    
                    pos = random.randint(0, len(word) - 1)
                    start_lo = max(0, pos - n + 1)
                    start_hi = min(pos, len(word) - n)
                    
                    if start_lo > start_hi:
                        continue
                    
                    affected_indices = []
                    for s in range(start_lo, start_hi + 1):
                        window = tuple(word[s:s + n])
                        if len(set(window)) == n:
                            idx = perm_to_idx.get(window)
                            if idx is not None:
                                affected_indices.append(idx)
                    
                    safe = True
                    for idx in affected_indices:
                        if counts[idx] == 1:
                            safe = False
                            break
                    
                    if safe:
                        word.pop(pos)
                        for idx in affected_indices:
                            counts[idx] -= 1
                        if len(word) < best_len:
                            best_word = list(word)
                            best_len = len(word)
                        break
                else:
                    # Truly stuck
                    break
    
    return best_word
# EVOLVE-BLOCK-END
