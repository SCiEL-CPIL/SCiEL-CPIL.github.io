"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _cover_ternary_6_1()
    return _generic_greedy(q, n, radius)

def _cover_ternary_6_1():
    q = 3
    n = 6
    total_words = q ** n
    
    # 1. Initialize with Direct Sum construction (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(h + list(tail))
            
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
        
    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w

    code_ints = [word_to_int(w) for w in initial_code]
    
    # Precompute neighbors (distance <= 1)
    neighbors = [[] for _ in range(total_words)]
    neighbor_sets = [None] * total_words
    for i in range(total_words):
        w = int_to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                neighbors[i].append(word_to_int(w))
            w[pos] = orig
        neighbor_sets[i] = set(neighbors[i])

    # State
    in_code = [False] * total_words
    cover_count = [0] * total_words
    code_list = list(code_ints)
    code_set = set(code_ints)
    
    for c in code_ints:
        in_code[c] = True
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    def try_remove(c):
        # Returns True if removed
        for nb in neighbors[c]:
            if cover_count[nb] == 1:
                return False
        for nb in neighbors[c]:
            cover_count[nb] -= 1
        in_code[c] = False
        code_set.remove(c)
        return True

    def try_swap(c_old, c_new):
        # Returns True if swapped
        critical_words = [nb for nb in neighbors[c_old] if cover_count[nb] == 1]
        if critical_words:
            if not set(critical_words).issubset(neighbor_sets[c_new]):
                return False
        
        for nb in neighbors[c_old]:
            cover_count[nb] -= 1
        in_code[c_old] = False
        code_set.remove(c_old)
        
        for nb in neighbors[c_new]:
            cover_count[nb] += 1
        in_code[c_new] = True
        code_set.add(c_new)
        return True

    start_time = time.time()
    deadline = start_time + 140.0
    
    # Phase 1: Greedy Removal
    # Shuffle to randomize removal order
    random.shuffle(code_list)
    # We need to be careful with modifying code_list while iterating
    # Make a copy of indices or just iterate and check validity
    i = 0
    while i < len(code_list):
        if time.time() > deadline:
            break
        c = code_list[i]
        if c in code_set: # Might have been removed? No, we remove from list too.
            if try_remove(c):
                code_list.pop(i)
                # Do not increment i, check next element at same index
            else:
                i += 1
        else:
            i += 1

    # Phase 2: Local Search
    while time.time() < deadline:
        if not code_list:
            break
            
        success = False
        
        # Try Removal
        if code_list:
            idx = random.randint(0, len(code_list) - 1)
            c = code_list[idx]
            if try_remove(c):
                code_list.pop(idx)
                success = True
                
        # Try Swap
        if not success and code_list:
            idx = random.randint(0, len(code_list) - 1)
            c_old = code_list[idx]
            
            # Find a valid c_new
            # Random sampling
            while True:
                c_new = random.randint(0, total_words - 1)
                if c_new not in code_set:
                    break
                    
            if try_swap(c_old, c_new):
                code_list[idx] = c_new
                success = True
                
        # If nothing worked, we just loop again.
        # To avoid infinite tight loop on failure, we could add a small sleep or just rely on time check.
        # The time check is inside the loop.

    final_code_ints = code_list
    final_code = [int_to_word(c) for c in final_code_ints]
    return final_code

def _generic_greedy(q, n, radius):
    # Fallback
    return []
# EVOLVE-BLOCK-END
