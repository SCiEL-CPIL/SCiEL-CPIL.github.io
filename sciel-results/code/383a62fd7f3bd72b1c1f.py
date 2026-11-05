import time
import numpy as np
import random
from itertools import product

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    space_size = q ** n
    
    # Convert index to word
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    # Powers of q
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q
    
    # Precompute neighbors as bitsets (Python integers)
    # Each neighbor set is represented as a bitmask
    neighbors = [0] * space_size
    neighbor_list = [[] for _ in range(space_size)]
    
    for idx in range(space_size):
        mask = 1 << idx
        lst = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n-1-pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n-1-pos]
                nidx = idx + delta
                mask |= (1 << nidx)
                lst.append(nidx)
        neighbors[idx] = mask
        neighbor_list[idx] = lst
    
    # Coverage count array
    cov = np.zeros(space_size, dtype=np.int32)
    in_set = np.zeros(space_size, dtype=bool)
    
    def compute_cov(code):
        """Compute coverage array from a list of codeword indices."""
        c = np.zeros(space_size, dtype=np.int32)
        for ci in code:
            for nb in neighbor_list[ci]:
                c[nb] += 1
        return c
    
    def is_valid(code):
        """Check if code covers all space."""
        c = np.zeros(space_size, dtype=np.int32)
        for ci in code:
            for nb in neighbor_list[ci]:
                c[nb] += 1
        return np.all(c > 0)
    
    # === Phase 1: Generate initial code using group code structure ===
    # For ternary codes, use a linear code + cosets approach
    # Start with a simple greedy, then improve
    
    # Greedy set cover with bitset speedup
    uncovered_mask = (1 << space_size) - 1  # All bits set
    initial_code = []
    
    # Precompute gain for each codeword
    while uncovered_mask != 0:
        best_gain = -1
        best_idx = -1
        for idx in range(space_size):
            gain = bin(uncovered_mask & neighbors[idx]).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                if gain == 0:
                    break
        
        if best_gain == 0:
            break
        
        initial_code.append(best_idx)
        uncovered_mask &= ~neighbors[best_idx]
        
        if time.time() - t_start > deadline - 30:
            break
    
    # Ensure validity
    if uncovered_mask != 0:
        while uncovered_mask != 0:
            # Find an uncovered position
            for i in range(space_size):
                if (uncovered_mask >> i) & 1:
                    initial_code.append(i)
                    uncovered_mask &= ~neighbors[i]
                    break
            if time.time() - t_start > deadline - 30:
                break
    
    # === Phase 2: Local search with bitset operations ===
    current_code = initial_code[:]
    current_cov = compute_cov(current_code)
    for ci in current_code:
        in_set[ci] = True
    
    best_code = current_code[:]
    best_size = len(current_code)
    
    def try_remove(cov, in_set_mask, code, c_idx):
        """Try to remove codeword at c_idx. Returns (success, new_code, new_cov, new_in_set)."""
        c = code[c_idx]
        # Check if all neighbors of c have cov > 1
        for nb in neighbor_list[c]:
            if cov[nb] <= 1:
                return False, None, None, None
        new_code = code[:]
        new_code.pop(c_idx)
        new_cov = cov.copy()
        for nb in neighbor_list[c]:
            new_cov[nb] -= 1
        new_in_set = in_set_mask.copy()
        new_in_set[c] = False
        return True, new_code, new_cov, new_in_set
    
    def try_swap(cov, in_set_mask, code, c_idx, new_c):
        """Try to swap codeword at c_idx with new_c."""
        if in_set_mask[new_c]:
            return False, None, None, None
        c = code[c_idx]
        if c == new_c:
            return False, None, None, None
        # Check: for each neighbor of c with cov==1, it must be covered by new_c
        new_c_mask = neighbors[new_c]
        for nb in neighbor_list[c]:
            if cov[nb] == 1:
                if not (new_c_mask >> nb) & 1:
                    return False, None, None, None
        new_code = code[:]
        new_code[c_idx] = new_c
        new_cov = cov.copy()
        for nb in neighbor_list[c]:
            new_cov[nb] -= 1
        for nb in neighbor_list[new_c]:
            new_cov[nb] += 1
        new_in_set = in_set_mask.copy()
        new_in_set[c] = False
        new_in_set[new_c] = True
        return True, new_code, new_cov, new_in_set
    
    def full_prune(code, cov, in_set_mask):
        """Aggressively remove all redundant codewords."""
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                success, new_code, new_cov, new_in_set = try_remove(cov, in_set_mask, code, i)
                if success:
                    code, cov, in_set_mask = new_code, new_cov, new_in_set
                    changed = True
                    break
        return code, cov, in_set_mask
    
    # Initial pruning
    current_code, current_cov, in_set = full_prune(current_code, current_cov, in_set)
    best_code = current_code[:]
    best_size = len(current_code)
    
    rng = random.Random(42)
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - t_start < deadline - 5:
        made_progress = False
        
        # Strategy 1: Try to remove a random codeword
        if len(current_code) > 0:
            i = rng.randrange(len(current_code))
            success, new_code, new_cov, new_in_set = try_remove(current_cov, in_set, current_code, i)
            if success:
                current_code, current_cov, in_set = new_code, new_cov, new_in_set
                if len(current_code) < best_size:
                    best_code = current_code[:]
                    best_size = len(current_code)
                no_improve = 0
                made_progress = True
        
        # Strategy 2: Try a random swap
        if not made_progress and len(current_code) > 0:
            i = rng.randrange(len(current_code))
            new_c = rng.randrange(space_size)
            success, new_code, new_cov, new_in_set = try_swap(current_cov, in_set, current_code, i, new_c)
            if success:
                # Accept swap with some probability to explore
                if rng.random() < 0.3:
                    current_code, current_cov, in_set = new_code, new_cov, new_in_set
                    # Try to prune after swap
                    current_code, current_cov, in_set = full_prune(current_code, current_cov, in_set)
                    if len(current_code) < best_size:
                        best_code = current_code[:]
                        best_size = len(current_code)
                    no_improve = 0
                    made_progress = True
        
        # Strategy 3: When stuck, try a multi-swap kick
        if not made_progress and no_improve > max_no_improve:
            # Remove 1-2 random codewords and add 1-2 random ones
            kick_size = min(2, len(current_code))
            removed = []
            for _ in range(kick_size):
                i = rng.randrange(len(current_code))
                removed.append(current_code.pop(i))
            for _ in range(kick_size):
                new_c = rng.randrange(space_size)
                if not in_set[new_c]:
                    current_code.append(new_c)
                    in_set[new_c] = True
            # Recompute coverage
            current_cov = compute_cov(current_code)
            in_set = np.zeros(space_size, dtype=bool)
            for ci in current_code:
                in_set[ci] = True
            # Prune
            current_code, current_cov, in_set = full_prune(current_code, current_cov, in_set)
            if len(current_code) < best_size:
                best_code = current_code[:]
                best_size = len(current_code)
            no_improve = 0
            made_progress = True
        
        if not made_progress:
            no_improve += 1
        
        # Periodic full prune
        if no_improve % 20 == 0:
            current_code, current_cov, in_set = full_prune(current_code, current_cov, in_set)
            if len(current_code) < best_size:
                best_code = current_code[:]
                best_size = len(current_code)
    
    # Final validation
    final_cov = compute_cov(best_code)
    if np.any(final_cov == 0):
        # Fallback
        return [idx_to_word(c) for c in initial_code]
    return [idx_to_word(c) for c in best_code]
