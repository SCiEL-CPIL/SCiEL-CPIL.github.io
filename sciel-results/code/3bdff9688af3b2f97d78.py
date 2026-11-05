"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _compute_overlap(current, block):
    """Compute maximum overlap: longest suffix of current matching prefix of block."""
    max_check = min(len(current), len(block))
    for k in range(max_check, 0, -1):
        if current[-k:] == block[:k]:
            return k
    return 0

def _build_block(p, m):
    """Build the block for permutation p with new symbol m."""
    return list(p) + [m] + list(p)

def _build_word_from_order(perms, m):
    """Build the word by appending blocks in the given order."""
    word = []
    for p in perms:
        block = _build_block(p, m)
        ov = _compute_overlap(word, block)
        word.extend(block[ov:])
    return word

def _greedy_order(perms, m):
    """Greedy nearest-neighbor: always pick the next perm that maximizes overlap."""
    remaining = list(perms)
    if not remaining:
        return []
    order = [remaining.pop(0)]
    current_block = _build_block(order[0], m)
    word_suffix = list(current_block)
    
    while remaining:
        best_idx = -1
        best_overlap = -1
        for i, p in enumerate(remaining):
            block = _build_block(p, m)
            ov = _compute_overlap(word_suffix, block)
            if ov > best_overlap:
                best_overlap = ov
                best_idx = i
        if best_idx < 0:
            best_idx = 0
        chosen = remaining.pop(best_idx)
        order.append(chosen)
        new_block = _build_block(chosen, m)
        word_suffix = word_suffix + new_block[best_overlap:]
        # Keep suffix bounded for efficiency
        if len(word_suffix) > 2 * m + 2:
            word_suffix = word_suffix[-(2 * m + 2):]
    return order

def _two_opt_improve(order, perms_set, m, max_iterations=50, deadline=None):
    """Improve ordering using 2-opt moves."""
    n_perms = len(order)
    if n_perms < 2:
        return order
    
    # Precompute blocks
    blocks = [_build_block(p, m) for p in order]
    
    def total_length(order_idx):
        """Compute total word length for given order of block indices."""
        total = len(blocks[order_idx[0]])
        for i in range(1, len(order_idx)):
            prev_block = blocks[order_idx[i-1]]
            curr_block = blocks[order_idx[i]]
            ov = _compute_overlap(prev_block, curr_block)
            total += len(curr_block) - ov
        return total
    
    current_order = list(range(n_perms))
    current_len = total_length(current_order)
    no_improve = 0
    iter_count = 0
    
    while no_improve < 10 and iter_count < max_iterations:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(n_perms - 1):
            if deadline and time.time() > deadline:
                break
            for j in range(i + 1, n_perms):
                if deadline and time.time() > deadline:
                    break
                # Try reversing segment [i..j]
                new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                new_len = total_length(new_order)
                if new_len < current_len:
                    current_order = new_order
                    current_len = new_len
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
        iter_count += 1
    
    return [order[i] for i in current_order]

def _swap_improve(order, perms_set, m, max_passes=20, deadline=None):
    """Improve ordering using adjacent swaps."""
    n_perms = len(order)
    if n_perms < 2:
        return order
    
    blocks = [_build_block(p, m) for p in order]
    
    def total_length(order_idx):
        total = len(blocks[order_idx[0]])
        for i in range(1, len(order_idx)):
            prev_block = blocks[order_idx[i-1]]
            curr_block = blocks[order_idx[i]]
            ov = _compute_overlap(prev_block, curr_block)
            total += len(curr_block) - ov
        return total
    
    current_order = list(range(n_perms))
    current_len = total_length(current_order)
    
    for _ in range(max_passes):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(n_perms - 1):
            if deadline and time.time() > deadline:
                break
            # Try swapping i and i+1
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_len = total_length(new_order)
            if new_len < current_len:
                current_order = new_order
                current_len = new_len
                improved = True
        if not improved:
            break
    
    return [order[i] for i in current_order]

def _build_superperm(n):
    """Build superpermutation with optimized ordering at each level."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        
        # Extract all distinct (m-1)-permutations from current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        if not perms:
            break
        
        # For small m, use standard order
        if m <= 4:
            new_word = []
            for p in perms:
                block = _build_block(p, m)
                ov = _compute_overlap(new_word, block)
                new_word.extend(block[ov:])
            word = new_word
            continue
        
        # For larger m, use optimization
        # Greedy initialization
        greedy_order = _greedy_order(perms, m)
        
        # Apply swap improvement
        improved_order = _swap_improve(greedy_order, set(perms), m, 
                                        max_passes=30, deadline=deadline)
        
        # Build word from improved order
        new_word = _build_word_from_order(improved_order, m)
        
        # Try 2-opt if time permits
        if time.time() < deadline - 20:
            two_opt_order = _two_opt_improve(improved_order, set(perms), m,
                                              max_iterations=20, deadline=deadline)
            two_opt_word = _build_word_from_order(two_opt_order, m)
            if len(two_opt_word) < len(new_word):
                new_word = two_opt_word
        
        word = new_word
    
    return word

def _verify_superperm(word, n):
    """Verify that all permutations of 1..n appear in word."""
    if len(word) < n:
        return False
    perms_found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            perms_found.add(window)
    return len(perms_found) == _factorial(n)

def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Handle small cases directly
    if n <= 1:
        return "1"
    
    # Build with optimization
    word = _build_superperm(n)
    
    # If we ran out of time and the word might be incomplete, fall back
    # Verify if possible
    if n <= 6:
        if not _verify_superperm(word, n):
            # Fallback to standard construction
            word = _standard_superperm(n)
    
    return "".join(map(str, word))

def _standard_superperm(n):
    """Standard recursive construction as fallback."""
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
            overlap = next((o for o in range(min(len(new), len(block)), -1, -1) 
                          if new[len(new) - o:] == block[:o]), 0)
            new.extend(block[overlap:])
        word = new
    return word

# EVOLVE-BLOCK-END
