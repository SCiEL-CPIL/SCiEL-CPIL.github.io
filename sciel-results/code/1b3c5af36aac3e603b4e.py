import itertools
import random
import time
from collections import Counter

def construct(q, n, radius):
    # Assume q=3, n=6, radius=1 based on task description
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    
    # Fallback generic greedy
    return _construct_generic(q, n, radius)

def _construct_ternary_6_1():
    """
    Structural change: Formulate as maximum entropy / cross-entropy minimization.
    
    We use a bitset representation for fast set operations.
    Each codeword covers a ball of 19 words. We represent the coverage as a bitset.
    We start with a tetracode-based initial set (size 81).
    
    The "maximum entropy" aspect is addressed by minimizing the cross-entropy 
    between the empirical coverage distribution and a uniform target. 
    In practice, this translates to:
    1. Removing codewords that are "redundant" (i.e., their removal doesn't uncover any points).
    2. Swapping codewords to reduce the "entropy" of the coverage distribution, 
       i.e., making the coverage more uniform (though for a covering code, we just need coverage >= 1).
    
    However, a more direct structural change is to use a **greedy algorithm based on maximum coverage** 
    followed by **local search with bitset operations**.
    
    Let's implement a robust bitset-based local search.
    """
    N = 729
    L = 6
    Q = 3
    
    # Precompute balls as bitsets
    # ball[i] is a bitset of all words within distance 1 of word i
    balls = [0] * N
    for i in range(N):
        word = []
        temp = i
        for _ in range(L):
            word.append(temp % Q)
            temp //= Q
        word = word[::-1]
        
        mask = 1 << i
        for pos in range(L):
            for new_val in range(Q):
                if new_val == word[pos]:
                    continue
                new_word = word.copy()
                new_word[pos] = new_val
                new_idx = 0
                for d in new_word:
                    new_idx = new_idx * Q + d
                mask |= 1 << new_idx
        balls[i] = mask

    # Initial code: Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for t0 in range(3):
            for t1 in range(3):
                word = h + [t0, t1]
                idx = 0
                for d in word:
                    idx = idx * Q + d
                initial_code.add(idx)

    code = set(initial_code)
    coverage = [0] * N
    
    for c in code:
        mask = balls[c]
        # Iterate over bits in mask
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage[idx] += 1
            m ^= lsb

    def remove_word(c):
        mask = balls[c]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage[idx] -= 1
            m ^= lsb

    def add_word(c):
        mask = balls[c]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage[idx] += 1
            m ^= lsb

    def is_valid():
        return all(x > 0 for x in coverage)

    def try_remove(c):
        remove_word(c)
        if is_valid():
            code.remove(c)
            return True
        else:
            add_word(c)
            return False

    def try_swap(c_old, c_new):
        if c_new in code:
            return False
        remove_word(c_old)
        add_word(c_new)
        if is_valid():
            code.remove(c_old)
            code.add(c_new)
            return True
        else:
            remove_word(c_new)
            add_word(c_old)
            return False

    start_time = time.time()
    deadline = 145.0
    best_size = len(code)
    best_code = set(code)
    
    # Simulated Annealing / Randomized Local Search
    temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Precompute list of all words for random sampling
    all_words = list(range(N))
    
    while time.time() < deadline and temp > min_temp:
        # Phase 1: Greedy removal
        changed = True
        while changed:
            changed = False
            # Randomize order to avoid getting stuck in same local min
            code_list = list(code)
            random.shuffle(code_list)
            for c in code_list:
                if time.time() > deadline:
                    break
                if c not in code:
                    continue
                if try_remove(c):
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = set(code)
                        
        if time.time() > deadline:
            break

        # Phase 2: Random swaps (Simulated Annealing style)
        for _ in range(50):
            if time.time() > deadline:
                break
            if not code:
                break
            c_old = random.choice(list(code))
            c_new = random.choice(all_words)
            
            # Only accept if it reduces size (impossible for swap) or maintains validity
            # Actually, swap keeps size same. We need to allow invalid intermediate states?
            # No, we stick to valid moves only, but we can use "kick" moves.
            
            # Let's try a "kick": remove two, add one? Or remove one, add two?
            # To reduce size, we need to remove more than we add.
            # Since swap keeps size, we need to find a way to reduce size.
            
            # Wait, if we can't remove any single word, we might need to swap to a position 
            # that allows a removal.
            
            if c_new in code:
                continue
                
            # Try swap
            if try_swap(c_old, c_new):
                # After swap, try to remove again
                code_list = list(code)
                random.shuffle(code_list)
                for c in code_list:
                    if time.time() > deadline:
                        break
                    if c not in code:
                        continue
                    if try_remove(c):
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = set(code)
                break # Restart phase 1 loop if a swap was successful?
        
        temp *= cooling_rate

    # Final greedy removal
    code_list = list(code)
    random.shuffle(code_list)
    for c in code_list:
        if time.time() > deadline:
            break
        if c not in code:
            continue
        if try_remove(c):
            if len(code) < best_size:
                best_size = len(code)
                best_code = set(code)

    final_code = best_code if len(best_code) < len(code) else code
    
    def to_word(val):
        word = []
        for _ in range(L):
            word.append(val % Q)
            val //= Q
        return word[::-1]

    return [to_word(c) for c in final_code]

def _construct_generic(q, n, radius):
    total = q ** n
    words = [list(map(int, itertools.product(range(q), repeat=n))[i]) for i in range(total)]
    
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    covered = [False] * total
    code = []
    remaining = set(range(total))
    
    while remaining:
        best_w = None
        best_count = -1
        for w_idx in remaining:
            w = words[w_idx]
            count = 0
            for r_idx in remaining:
                if dist(w, words[r_idx]) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_w = w_idx
                
        code.append(words[best_w])
        w = words[best_w]
        new_remaining = set()
        for r_idx in remaining:
            if dist(w, words[r_idx]) > radius:
                new_remaining.add(r_idx)
        remaining = new_remaining
        
    return code
