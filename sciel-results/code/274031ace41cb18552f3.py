import numpy as np
import time
import itertools
import random

Q = 3
N = 7
RADIUS = 1
TOTAL_WORDS = Q ** N  # 2187

def int_to_base3(num, length):
    digits = [0] * length
    for i in range(length):
        digits[i] = num % Q
        num //= Q
    return digits

# Precompute base-3 representations
base3_words = [int_to_base3(w, N) for w in range(TOTAL_WORDS)]

# Build neighbor lists and coverage matrix
# neighbors[i] = list of indices j such that dist(i, j) <= 1
neighbors = [[] for _ in range(TOTAL_WORDS)]
for i in range(TOTAL_WORDS):
    wi = base3_words[i]
    neighbors[i].append(i)
    for pos in range(N):
        orig_val = wi[pos]
        for new_val in range(Q):
            if new_val == orig_val:
                continue
            wj = wi.copy()
            wj[pos] = new_val
            val = 0
            for k in range(N):
                val = val * Q + wj[k]
            neighbors[i].append(val)

# Convert to numpy arrays for fast operations
neighbors_arr = [np.array(neighbors[i], dtype=np.int32) for i in range(TOTAL_WORDS)]

# Build sparse representation: for each word, which codeword positions cover it
# Actually, let's build a matrix M where M[i,j]=1 if dist(i,j)<=1
# But 2187x2187 bool matrix is ~4.7M entries, which is fine
M = np.zeros((TOTAL_WORDS, TOTAL_WORDS), dtype=np.int8)
for i in range(TOTAL_WORDS):
    for j in neighbors[i]:
        M[i, j] = 1

# Also build reverse: for each codeword c, which words does it cover?
# That's just M[:, c] which is the column

def greedy_construct():
    uncovered = np.ones(TOTAL_WORDS, dtype=bool)
    code = []
    
    while np.any(uncovered):
        uncovered_indices = np.nonzero(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
            
        best_word = -1
        max_covered = 0
        
        for w in uncovered_indices:
            cnt = np.sum(uncovered[neighbors_arr[w]])
            if cnt > max_covered:
                max_covered = cnt
                best_word = w
        
        if best_word == -1:
            break
            
        code.append(best_word)
        uncovered[neighbors_arr[best_word]] = False
        
    return code

def local_search_stochastic(code, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    code = list(code)
    
    code_set = set(code)
    code_mask = np.zeros(TOTAL_WORDS, dtype=bool)
    for c in code:
        code_mask[c] = True
    
    # Compute initial cover count
    cover_count = np.zeros(TOTAL_WORDS, dtype=np.int32)
    for c in code:
        cover_count += M[:, c]
    
    best_code = list(code)
    best_size = len(code)
    
    no_improve_counter = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if time.time() > deadline - 2:
            break
            
        if no_improve_counter > max_no_improve:
            # Reset and try a different strategy
            no_improve_counter = 0
            max_no_improve = 10000
            # Try to remove a random redundant codeword
            removable = []
            for c in code:
                test_counts = cover_count - M[:, c]
                if np.all(test_counts > 0):
                    removable.append(c)
            if removable:
                c_to_remove = random.choice(removable)
                code.remove(c_to_remove)
                code_mask[c_to_remove] = False
                cover_count -= M[:, c_to_remove]
                if len(code) < best_size:
                    best_code = list(code)
                    best_size = len(code)
                no_improve_counter = 0
            else:
                # Try a random swap
                c_to_swap = random.choice(code)
                w_to_add = random.randint(0, TOTAL_WORDS - 1)
                if w_to_add in code_set:
                    continue
                new_counts = cover_count - M[:, c_to_swap] + M[:, w_to_add]
                if np.all(new_counts > 0):
                    # Can we remove something?
                    for c_prime in code:
                        if c_prime == c_to_swap:
                            continue
                        test = new_counts - M[:, c_prime]
                        if np.all(test > 0):
                            code.remove(c_to_swap)
                            code.remove(c_prime)
                            code.append(w_to_add)
                            code_mask[c_to_swap] = False
                            code_mask[c_prime] = False
                            code_mask[w_to_add] = True
                            cover_count = test
                            if len(code) < best_size:
                                best_code = list(code)
                                best_size = len(code)
                            no_improve_counter = 0
                            break
            continue
        
        # Try removal first
        improved = False
        removable = []
        for c in code:
            test_counts = cover_count - M[:, c]
            if np.all(test_counts > 0):
                removable.append(c)
        
        if removable:
            c_to_remove = random.choice(removable)
            code.remove(c_to_remove)
            code_mask[c_to_remove] = False
            cover_count -= M[:, c_to_remove]
            if len(code) < best_size:
                best_code = list(code)
                best_size = len(code)
            no_improve_counter = 0
            improved = True
            continue
        
        if not improved:
            # Try swap: remove c, add w, then try to remove c_prime
            # To be efficient, sample a few c and w
            c_to_swap = random.choice(code)
            
            # Find candidates w that when added (after removing c) still cover
            # We need new_counts = cover_count - M[:, c] + M[:, w] > 0 for all
            # This is expensive to check all w. Let's sample.
            found_swap = False
            for attempt in range(20):
                w_to_add = random.randint(0, TOTAL_WORDS - 1)
                if w_to_add in code_set:
                    continue
                new_counts = cover_count - M[:, c_to_swap] + M[:, w_to_add]
                if not np.all(new_counts > 0):
                    continue
                
                # Now try to remove a c_prime from code \ {c_to_swap}
                for c_prime in code:
                    if c_prime == c_to_swap:
                        continue
                    test = new_counts - M[:, c_prime]
                    if np.all(test > 0):
                        code.remove(c_to_swap)
                        code.remove(c_prime)
                        code.append(w_to_add)
                        code_mask[c_to_swap] = False
                        code_mask[c_prime] = False
                        code_mask[w_to_add] = True
                        cover_count = test
                        if len(code) < best_size:
                            best_code = list(code)
                            best_size = len(code)
                        no_improve_counter = 0
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if not found_swap:
                no_improve_counter += 1
    
    return best_code

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    start_time = time.time()
    deadline = 140.0
    
    # 1. Greedy
    code = greedy_construct()
    
    # 2. Local Search
    time_elapsed = time.time() - start_time
    remaining_time = deadline - time_elapsed
    
    if remaining_time > 5:
        code = local_search_stochastic(code, remaining_time)
    
    # Final Verification
    mask = np.zeros(TOTAL_WORDS, dtype=bool)
    for c in code:
        mask[c] = True
    cover_count = np.sum(M[mask], axis=0)
    if np.any(cover_count == 0):
        code = greedy_construct()
    
    # Convert to list of lists
    result = []
    for c in code:
        result.append(int_to_base3(c, n))
    
    return result

if __name__ == "__main__":
    pass
