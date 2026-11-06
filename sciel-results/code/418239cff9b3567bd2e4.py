"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _score_bitmask(mask, n=512):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d all in mask."""
    count = 0
    # d can range from 1 to 255 (since 2d < 512 needed for a+2d <= 511, so d <= 255)
    for d in range(1, 256):
        # Check if mask & (mask >> d) & (mask >> (2*d)) has any bit set
        if mask & (mask >> d) & (mask >> (2 * d)):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    best_set = set(range(1, 33))
    best_score = _score_bitmask(sum(1 << (x - 1) for x in best_set))
    
    # Try a few structured initializations
    # 4x8 grid
    grid48 = set()
    for i in range(4):
        for j in range(8):
            grid48.add(1 + i + 8 * j)
    s = _score_bitmask(sum(1 << (x - 1) for x in grid48))
    if s > best_score:
        best_score = s
        best_set = grid48
    
    # 8x4 grid
    grid84 = set()
    for i in range(8):
        for j in range(4):
            grid84.add(1 + i + 4 * j)
    s = _score_bitmask(sum(1 << (x - 1) for x in grid84))
    if s > best_score:
        best_score = s
        best_set = grid84
    
    # 2x16 grid
    grid216 = set()
    for i in range(2):
        for j in range(16):
            grid216.add(1 + i + 16 * j)
    s = _score_bitmask(sum(1 << (x - 1) for x in grid216))
    if s > best_score:
        best_score = s
        best_set = grid216
    
    # 16x2 grid
    grid162 = set()
    for i in range(16):
        for j in range(2):
            grid162.add(1 + i + 2 * j)
    s = _score_bitmask(sum(1 << (x - 1) for x in grid162))
    if s > best_score:
        best_score = s
        best_set = grid162
    
    # 1x32 (original)
    # Already covered
    
    # Local search: simulated annealing with add/remove/swap moves
    current_set = set(best_set)
    current_mask = sum(1 << (x - 1) for x in current_set)
    current_score = best_score
    
    all_elements = list(range(1, 513))
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if temp < 0.01:
            temp = 1.0  # Restart with higher temperature if stuck
        
        # Choose a random move
        move_type = random.random()
        
        if len(current_set) < 32 and move_type < 0.4:
            # Add an element
            candidates = [x for x in all_elements if x not in current_set]
            if not candidates:
                continue
            new_elem = random.choice(candidates)
            new_set = current_set | {new_elem}
            new_mask = current_mask | (1 << (new_elem - 1))
            new_score = _score_bitmask(new_mask)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 0.01)):
                current_set = new_set
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
        elif len(current_set) > 32 and move_type < 0.7:
            # Remove an element
            if not current_set:
                continue
            old_elem = random.choice(list(current_set))
            new_set = current_set - {old_elem}
            new_mask = current_mask & ~(1 << (old_elem - 1))
            new_score = _score_bitmask(new_mask)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 0.01)):
                current_set = new_set
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
        else:
            # Swap: remove one, add one
            if len(current_set) < 32:
                # Just add
                candidates = [x for x in all_elements if x not in current_set]
                if candidates:
                    new_elem = random.choice(candidates)
                    new_set = current_set | {new_elem}
                    new_mask = current_mask | (1 << (new_elem - 1))
                    new_score = _score_bitmask(new_mask)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 0.01)):
                        current_set = new_set
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
            elif len(current_set) > 32:
                # Just remove
                old_elem = random.choice(list(current_set))
                new_set = current_set - {old_elem}
                new_mask = current_mask & ~(1 << (old_elem - 1))
                new_score = _score_bitmask(new_mask)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 0.01)):
                    current_set = new_set
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
            else:
                # Exact 32: swap
                old_elem = random.choice(list(current_set))
                candidates = [x for x in all_elements if x not in current_set]
                if candidates:
                    new_elem = random.choice(candidates)
                    new_set = (current_set - {old_elem}) | {new_elem}
                    new_mask = (current_mask & ~(1 << (old_elem - 1))) | (1 << (new_elem - 1))
                    new_score = _score_bitmask(new_mask)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 0.01)):
                        current_set = new_set
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
        
        temp *= cooling
    
    return sorted(best_set)
# EVOLVE-BLOCK-END

