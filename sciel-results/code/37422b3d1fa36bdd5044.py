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

# Precompute balls as bitsets (Python ints)
# ball_bits[c] is an integer where bit i is 1 if word i is within distance 1 of codeword c
ball_bits = [0] * NUM_WORDS
ball_indices = [None] * NUM_WORDS

for i in range(NUM_WORDS):
    c = all_words[i]
    bits = 1 << i
    indices = [i]
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            bits |= (1 << idx)
            indices.append(idx)
    ball_bits[i] = bits
    ball_indices[i] = indices

def get_initial_code():
    """Greedy set cover using bitsets."""
    covered_bits = 0
    code = []
    all_bits = (1 << NUM_WORDS) - 1
    
    # Calculate initial scores: how many bits does each ball have? (Constant 19)
    # We need to pick the one that adds the most NEW bits.
    # score[c] = popcount(ball_bits[c] & ~covered_bits)
    
    while covered_bits != all_bits:
        best_c = -1
        best_score = -1
        # Randomize tie-breaking slightly or just iterate
        # For speed, we can sample or iterate all 729. 729 * 80 steps is small.
        for c in range(NUM_WORDS):
            new_bits = ball_bits[c] & ~covered_bits
            # popcount
            score = bin(new_bits).count('1')
            if score > best_score:
                best_score = score
                best_c = c
            elif score == best_score and best_c != -1 and random.random() < 0.1:
                # Small randomization
                best_c = c
        
        if best_score == 0:
            break # Should not happen if cover exists
            
        code.append(best_c)
        covered_bits |= ball_bits[best_c]
        
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if q is None or n is None or radius is None:
            return []
        # Simple greedy for general case
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
    deadline = start_time + 140  # 10s safety margin
    
    # Initialize with greedy cover
    current_code = get_initial_code()
    best_code = current_code.copy()
    best_size = len(best_code)
    
    # Local Search with Swaps
    # We maintain a list of codeword indices
    # cov_counts[w] = number of codewords in current_code that cover w
    
    def compute_cov_counts(code):
        counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            for w in ball_indices[c]:
                counts[w] += 1
        return counts

    def is_valid_cover(code):
        # Check if union of balls covers all words
        covered = 0
        for c in code:
            covered |= ball_bits[c]
        return covered == (1 << NUM_WORDS) - 1

    # Optimization: We can check validity faster using the swap logic itself.
    # But for initial check and safety, bitset OR is fast.
    
    num_restarts = 0
    while time.time() < deadline:
        # Local search phase on current_code
        changed = True
        while changed and time.time() < deadline:
            changed = False
            cov_counts = compute_cov_counts(current_code)
            
            # 1. Try to remove redundant codewords
            removed_any = False
            # Shuffle to break symmetry
            indices = list(range(len(current_code)))
            random.shuffle(indices)
            
            for idx in indices:
                c = current_code[idx]
                # Check if all words in ball(c) are covered by others
                # i.e., cov_counts[w] > 1 for all w in ball_indices[c]
                redundant = True
                for w in ball_indices[c]:
                    if cov_counts[w] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove c
                    current_code.pop(idx)
                    changed = True
                    removed_any = True
                    break # Restart loop to recompute cov_counts
            
            if removed_any:
                continue # Restart inner while loop
            
            # 2. Try swaps
            # If no removals, try to swap a codeword to a better position
            # We look for a codeword c_old and a candidate c_new not in code
            # such that B(c_new) covers all unique points of c_old.
            
            swap_done = False
            indices = list(range(len(current_code)))
            random.shuffle(indices)
            
            for idx in indices:
                c_old = current_code[idx]
                
                # Find unique points of c_old
                unique_bits = 0
                for w in ball_indices[c_old]:
                    if cov_counts[w] == 1:
                        unique_bits |= (1 << w)
                
                if unique_bits == 0:
                    # Should have been removed in step 1, but just in case
                    current_code.pop(idx)
                    changed = True
                    swap_done = True
                    break
                
                # Find c_new not in current_code such that unique_bits is subset of ball_bits[c_new]
                # To speed up, we can iterate through all c_new and check
                # We can also filter c_new by checking if it covers at least one unique bit
                # But simple check is fast enough for 729 candidates.
                
                # Optimization: Only consider c_new that are "close" or random?
                # Let's iterate all, but break early if found.
                # We need to ensure c_new is not in current_code.
                # Creating a set of current_code is O(|C|).
                code_set = set(current_code)
                
                found_swap = False
                # Randomize order of candidates?
                candidates = list(range(NUM_WORDS))
                random.shuffle(candidates)
                
                for c_new in candidates:
                    if c_new in code_set:
                        continue
                    # Check if unique_bits is subset of ball_bits[c_new]
                    if (unique_bits & ~ball_bits[c_new]) == 0:
                        # Valid swap
                        current_code[idx] = c_new
                        changed = True
                        swap_done = True
                        found_swap = True
                        break
                
                if found_swap:
                    break # Restart inner while loop to recompute cov_counts
            
            if swap_done:
                continue
        
        # Update best
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code.copy()
            # If we hit 73, we can try to go lower, but 73 is best known.
            # We can stop early if we want, but let's keep searching.
            
        # Perturbation: Randomly swap or add/remove to escape local minima
        if time.time() > deadline:
            break
            
        # Simple perturbation: Randomly pick a codeword and replace it with a random other
        # Or add a random codeword and then run reduction again?
        # Let's do a random swap perturbation
        if len(current_code) > 0:
            idx = random.randint(0, len(current_code) - 1)
            old_c = current_code[idx]
            new_c = random.randint(0, NUM_WORDS - 1)
            if new_c not in current_code:
                current_code[idx] = new_c
            else:
                # Just randomize a bit
                pass
        else:
            current_code = get_initial_code()

    # Verify best code
    if not is_valid_cover(best_code):
        # Fallback to greedy if something went wrong
        best_code = get_initial_code()
        
    return [list(all_words[c]) for c in best_code]
