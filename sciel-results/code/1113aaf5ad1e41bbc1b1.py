import numpy as np
import itertools
import time
import random

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls
# ball_matrix[c, w] = 1 if w is within distance 1 of c
ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
ball_indices = []  # list of arrays, each containing indices in the ball

for i in range(NUM_WORDS):
    c = all_words[i]
    ball_indices.append([i])
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball_indices[-1].append(idx)
            ball_matrix[i, idx] = 1

# Convert to list of arrays for fast indexing
ball_indices = [np.array(bi, dtype=np.int32) for bi in ball_indices]

def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

    # 1. Greedy Initialization
    covered = np.zeros(NUM_WORDS, dtype=bool)
    code = []
    while not np.all(covered):
        scores = ball_matrix.dot(~covered)
        c = int(np.argmax(scores))
        if scores[c] == 0:
            break
        code.append(c)
        covered[ball_matrix[c] == 1] = True
    
    best_code = [list(all_words[c]) for c in code]
    best_size = len(code)
    
    # Helper to compute coverage counts
    def get_coverage(code_list):
        cov = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code_list:
            cov[ball_indices[c]] += 1
        return cov

    # Helper to try to reduce a code
    def try_reduce(code_list, cov_counts):
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                if np.all(cov_counts[ball_indices[c]] > 1):
                    cov_counts[ball_indices[c]] -= 1
                    code_list.pop(i)
                    changed = True
        return len(code_list)

    # Initial reduction
    cov_counts = get_coverage(code)
    code = try_reduce(code, cov_counts)
    best_size = len(code)
    best_code = [list(all_words[c]) for c in code]

    # Local Search Loop
    max_iter = 100000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = [list(all_words[c]) for c in code]
            
        if not code:
            break
            
        # Try to find a swap that enables reduction
        i_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[i_to_remove]
        
        # Temporarily remove it
        code.pop(i_to_remove)
        cov_counts[ball_indices[c_remove]] -= 1
        
        # Check if valid
        if np.all(cov_counts > 0):
            # It was redundant. We've effectively reduced the code by 1.
            # Update best
            if len(code) < best_size:
                best_size = len(code)
                best_code = [list(all_words[c]) for c in code]
            continue
        
        # Find uncovered words
        uncovered = np.where(cov_counts == 0)[0]
        
        # We need to find a codeword c_add that covers ALL uncovered words.
        # And c_add must not be in the code.
        
        first_uncov = uncovered[0]
        candidates = np.where(ball_matrix[:, first_uncov] == 1)[0]
        
        in_code_set = set(code)
        
        valid_candidates = []
        for c in candidates:
            if c in in_code_set:
                continue
            if np.all(ball_matrix[c, uncovered] == 1):
                valid_candidates.append(c)
                
        if not valid_candidates:
            # Revert
            code.append(c_remove)
            cov_counts[ball_indices[c_remove]] += 1
            continue
            
        # Pick a random valid candidate
        c_add = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
        
        # Perform swap
        code.append(c_add)
        cov_counts[ball_indices[c_add]] += 1
        
        # Now try to reduce
        try_reduce(code, cov_counts)
        
    return best_code
