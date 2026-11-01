import time
import random
import sys

# Constants
Q = 3
N = 7
RADIUS = 1
TOTAL_WORDS = 3 ** N  # 2187


def decode_word(index):
    """Convert integer index to list of digits."""
    digits = []
    for _ in range(N):
        digits.append(index % Q)
        index //= Q
    return digits


def encode_word(digits):
    """Convert list of digits to integer index."""
    val = 0
    for i in range(N - 1, -1, -1):
        val = val * Q + digits[i]
    return val


# Precompute balls
balls = [0] * TOTAL_WORDS
words = [decode_word(i) for i in range(TOTAL_WORDS)]

for i in range(TOTAL_WORDS):
    w = words[i]
    mask = 1 << i
    for pos in range(N):
        orig_val = w[pos]
        for new_val in range(Q):
            if new_val == orig_val:
                continue
            new_w = w[:]
            new_w[pos] = new_val
            j = encode_word(new_w)
            mask |= (1 << j)
    balls[i] = mask

FULL_MASK = (1 << TOTAL_WORDS) - 1


def greedy_construct(random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    uncovered = FULL_MASK
    code = []
    
    while uncovered:
        best_idx = -1
        best_count = -1
        best_candidates = []
        
        for i in range(TOTAL_WORDS):
            cov = (balls[i] & uncovered).bit_count()
            if cov > best_count:
                best_count = cov
                best_idx = i
                best_candidates = [i]
            elif cov == best_count and cov > 0:
                best_candidates.append(i)
        
        if best_count == 0:
            break
            
        selected = random.choice(best_candidates)
        code.append(selected)
        uncovered &= ~balls[selected]
        
    return code


def reduce_code(code):
    """Remove redundant codewords."""
    if not code:
        return code
    
    changed = True
    while changed:
        changed = False
        counts = [0] * TOTAL_WORDS
        for i in code:
            m = balls[i]
            while m:
                lowbit = m & -m
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                m ^= lowbit
        
        new_code = []
        for i in code:
            redundant = True
            m = balls[i]
            while m:
                lowbit = m & -m
                idx = lowbit.bit_length() - 1
                if counts[idx] == 1:
                    redundant = False
                    break
                m ^= lowbit
            
            if redundant:
                m = balls[i]
                while m:
                    lowbit = m & -m
                    idx = lowbit.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lowbit
                changed = True
            else:
                new_code.append(i)
        
        code = new_code
    
    return code


def swap_improve(code, random_seed=None):
    """Try to swap codewords to reduce code size."""
    if random_seed is not None:
        random.seed(random_seed)
    
    # Convert to set for faster lookup
    code_set = set(code)
    code_list = list(code)
    
    # Compute coverage counts
    counts = [0] * TOTAL_WORDS
    for i in code_list:
        m = balls[i]
        while m:
            lowbit = m & -m
            idx = lowbit.bit_length() - 1
            counts[idx] += 1
            m ^= lowbit
    
    changed = True
    max_iter = 100
    iter_count = 0
    
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        
        # Try to remove a codeword and replace with a better one
        for idx, i in enumerate(code_list):
            # Check if we can remove i
            m = balls[i]
            removable = True
            for pos in range(N):
                # This is a simplified check
                pass
            
            # More direct: check if all words covered by i are covered by others
            m = balls[i]
            while m:
                lowbit = m & -m
                j = lowbit.bit_length() - 1
                if counts[j] == 1:
                    removable = False
                    break
                m ^= lowbit
            
            if not removable:
                continue
            
            # Try to find a replacement that covers more uncovered territory
            # First, temporarily remove i
            m_remove = balls[i]
            while m_remove:
                lowbit = m_remove & -m_remove
                j = lowbit.bit_length() - 1
                counts[j] -= 1
                m_remove ^= lowbit
            
            # Find best replacement
            best_j = -1
            best_gain = -1
            for j in range(TOTAL_WORDS):
                if j in code_set:
                    continue
                gain = (balls[j] & ~balls[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            
            if best_j >= 0 and best_gain > 0:
                # Apply swap
                code_list[idx] = best_j
                code_set.remove(i)
                code_set.add(best_j)
                
                # Update counts
                m_add = balls[best_j]
                while m_add:
                    lowbit = m_add & -m_add
                    j = lowbit.bit_length() - 1
                    counts[j] += 1
                    m_add ^= lowbit
                
                changed = True
            else:
                # Revert
                m_restore = balls[i]
                while m_restore:
                    lowbit = m_restore & -m_restore
                    j = lowbit.bit_length() - 1
                    counts[j] += 1
                    m_restore ^= lowbit
    
    return code_list


def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # 10s margin before 120s limit
    
    best_code = None
    best_size = float('inf')
    
    # Initial run with fixed seed
    code = greedy_construct(0)
    code = reduce_code(code)
    code = swap_improve(code, 0)
    code = reduce_code(code)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code
    
    # Random restarts
    seed = 1
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        code = swap_improve(code, seed)
        code = reduce_code(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            if best_size <= 186:
                break
        seed += 1
    
    # Convert indices to lists of digits
    return [words[i] for i in best_code]


if __name__ == "__main__":
    res = construct(3, 7, 1)
    print(f"Size: {len(res)}")
    covered = 0
    for i in range(len(res)):
        covered |= balls[encode_word(res[i])]
    if covered == FULL_MASK:
        print("Valid")
    else:
        print("Invalid")
