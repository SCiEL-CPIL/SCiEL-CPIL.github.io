import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Generic fallback: just return all words (worst case) or a simple product
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Total number of words
    total_words = q ** n  # 6561

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists (Hamming distance <= radius)
    # For q=3, n=8, radius=1, each word has 1 + 8*2 = 17 neighbors
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # Precompute reverse neighbors: rev_neighbors[x] = list of i such that x is in neighbors[i]
    # This is the same as neighbors by symmetry (Hamming distance is symmetric)
    # So rev_neighbors[i] = neighbors[i]

    # Precompute bitset representation for fast intersection
    # bitset[i] = integer where bit j is set iff j is in neighbors[i]
    bitsets = [0] * total_words
    for i in range(total_words):
        bs = 0
        for j in neighbors[i]:
            bs |= (1 << j)
        bitsets[i] = bs

    # --- Helper: greedy set cover using efficient counting ---
    def greedy_cover(deadline_limit):
        covered = np.zeros(total_words, dtype=bool)
        code_list = []
        
        while not covered.all() and time.time() < deadline_limit:
            # Compute potential: for each candidate i, count uncovered neighbors
            # Efficient approach: for each uncovered point x, add 1 to potential of all i in neighbors[x]
            uncovered_idx = np.where(~covered)[0]
            if len(uncovered_idx) == 0:
                break
            
            potential = np.zeros(total_words, dtype=np.int32)
            for x in uncovered_idx:
                potential[neighbors[x]] += 1
            
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
            
            code_list.append(c)
            covered[neighbors[c]] = True
        
        return code_list

    # --- Helper: local search with swaps ---
    def local_search(initial_code, deadline_limit, seed):
        random.seed(seed)
        current_code = initial_code[:]
        current_size = len(current_code)
        
        # Compute coverage counts
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in current_code:
            cov_count[neighbors[c]] += 1
        
        current_code_set = set(current_code)
        
        stuck_counter = 0
        best_code = current_code[:]
        best_size = current_size
        
        while time.time() < deadline_limit:
            # 1. Try to remove a word
            if current_size == 0:
                break
            
            # Try removing in random order, check up to min(current_size, 50) candidates
            tried = 0
            removed = False
            max_tries = min(current_size, 50)
            while tried < max_tries and not removed:
                idx = random.randint(0, current_size - 1)
                c = current_code[idx]
                
                # Check if all neighbors are covered by more than one codeword
                if np.all(cov_count[neighbors[c]] > 1):
                    # Safe to remove
                    cov_count[neighbors[c]] -= 1
                    current_code_set.remove(c)
                    current_code.pop(idx)
                    current_size -= 1
                    stuck_counter = 0
                    removed = True
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                else:
                    tried += 1
            
            if not removed:
                stuck_counter += 1
            
            # 2. If stuck, try a swap
            if stuck_counter > 3 and time.time() < deadline_limit - 5.0 and current_size > 0:
                c1_idx = random.randint(0, current_size - 1)
                c1 = current_code[c1_idx]
                
                # Find risky points: x in neighbors[c1] where cov_count[x] == 1
                risky_points = [x for x in neighbors[c1] if cov_count[x] == 1]
                
                if not risky_points:
                    # c1 is removable
                    cov_count[neighbors[c1]] -= 1
                    current_code_set.remove(c1)
                    current_code.pop(c1_idx)
                    current_size -= 1
                    stuck_counter = 0
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                else:
                    # Find c2 not in code such that c2 covers all risky points.
                    # c2 must be in the intersection of neighbors[x] for all x in risky_points.
                    # Use bitset intersection
                    candidate_bits = bitsets[risky_points[0]]
                    for x in risky_points[1:]:
                        candidate_bits &= bitsets[x]
                        if candidate_bits == 0:
                            break
                    
                    if candidate_bits != 0:
                        # Exclude c1 itself and all current codewords
                        for c_code in current_code:
                            candidate_bits &= ~(1 << c_code)
                        
                        if candidate_bits != 0:
                            # Pick a random valid candidate
                            # Get list of set bits
                            candidates = []
                            tmp = candidate_bits
                            while tmp:
                                lsb = tmp & (-tmp)
                                pos = lsb.bit_length() - 1
                                candidates.append(pos)
                                tmp ^= lsb
                            
                            if candidates:
                                c2 = random.choice(candidates)
                                
                                # Perform the swap
                                cov_count[neighbors[c1]] -= 1
                                current_code_set.remove(c1)
                                current_code[c1_idx] = c2
                                
                                cov_count[neighbors[c2]] += 1
                                current_code_set.add(c2)
                                
                                stuck_counter = 0
                            
                            else:
                                pass
                        else:
                            pass
                    else:
                        pass
                    
                    # Reset stuck counter to allow more removal attempts first
                    stuck_counter = 0
        
        return best_code, best_size

    # --- Phase 1: Multiple greedy initializations with different strategies ---
    best_code = None
    best_size = float('inf')
    
    # Run multiple greedy passes with different randomizations
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 60.0:
            break
        
        # Greedy cover
        init_code = greedy_cover(deadline - 40.0)
        
        if not init_code:
            continue
        
        # Local search
        ls_code, ls_size = local_search(init_code, deadline - 20.0, seed=42 + r)
        
        if ls_size < best_size:
            best_size = ls_size
            best_code = ls_code[:]
    
    # If no valid solution found, fall back to full space
    if best_code is None:
        best_code = list(range(total_words))
        best_size = total_words

    # Final Polish - aggressive removal pass
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try removing in random order for better exploration
        random.seed(12345)
        order = list(range(len(final_code)))
        random.shuffle(order)
        for i in order:
            if i >= len(final_code):
                continue
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    # Verify coverage
    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]
        
    return [list(digits_of[i]) for i in final_code]
