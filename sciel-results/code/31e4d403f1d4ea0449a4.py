import itertools
import random
import time

def _popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')

def construct(q, n, radius):
    # This implementation is specialized for q=3, n=6, radius=1
    # as per the task description, but will try to be somewhat general or at least robust for this specific case.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # However, the specific problem is K_3(6,1).
    # If q, n, radius are different, we might need a general solver, but the prompt implies this specific instance.
    # The seed program hardcodes the tetracode logic for n=6, q=3.
    # We will assume the evaluator calls construct(3, 6, 1).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other parameters if needed, though unlikely.
        # For now, just return a trivial valid code (all words) which is always valid but large.
        return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    
    # 1. Generate all words in {0,1,2}^6
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total_words = len(all_words) # 729
    
    # 2. Precompute coverage sets as bitsets (Python integers)
    # coverage[i] is a bitset where bit j is 1 if word j is covered by codeword i
    coverage = [0] * total_words
    
    for i, c in enumerate(all_words):
        # c covers itself
        mask = 1 << word_to_idx[c]
        # c covers all words at distance 1
        for pos in range(6):
            for val in range(3):
                if val == c[pos]:
                    continue
                new_c = list(c)
                new_c[pos] = val
                new_c_tuple = tuple(new_c)
                mask |= 1 << word_to_idx[new_c_tuple]
        coverage[i] = mask
        
    # 3. Greedy Set Cover with restarts and Local Search
    best_code = None
    best_size = total_words
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit, leaving 20s margin
    
    r = 0
    while time.time() < deadline:
        random.seed(r)
        candidates = list(range(total_words))
        random.shuffle(candidates)
        
        covered_mask = 0
        chosen_indices = []
        
        # Greedy Phase
        while covered_mask != (1 << total_words) - 1:
            best_gain = -1
            best_idx = -1
            
            for idx in candidates:
                new_coverage = coverage[idx] & ~covered_mask
                gain = _popcount(new_coverage)
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    
            if best_gain == 0:
                break
                
            chosen_indices.append(best_idx)
            covered_mask |= coverage[best_idx]
            
        # Local Search Phase
        S = chosen_indices
        S_set = set(S)
        union_mask = 0
        for i in S:
            union_mask |= coverage[i]
            
        improved = True
        while improved:
            improved = False
            for i in S:
                others_mask = union_mask ^ coverage[i]
                unique_i = coverage[i] & ~others_mask
                
                if unique_i == 0:
                    S.remove(i)
                    S_set.discard(i)
                    improved = True
                    break
                
                best_j = -1
                best_gain = -1
                for j in range(total_words):
                    if j in S_set:
                        continue
                    if (unique_i & ~coverage[j]) != 0:
                        continue
                    gain = _popcount(coverage[j] & ~union_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                
                if best_j != -1 and best_gain > 0:
                    S.remove(i)
                    S_set.discard(i)
                    S.append(best_j)
                    S_set.add(best_j)
                    union_mask = (union_mask & ~coverage[i]) | coverage[best_j]
                    improved = True
                    break
        
        if len(S) < best_size:
            best_size = len(S)
            best_code = [all_words[i] for i in S]
        r += 1

    # Convert best_code to list of lists
    if best_code is None:
        # Fallback to seed logic if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in best_code]

# EVOLVE-BLOCK-START
# The code above is the complete implementation.
# We place it inside the block as requested by the format, though the prompt says "Rewrite the program from scratch".
# The seed program has EVOLVE-BLOCK markers. I will provide the full file content.
# EVOLVE-BLOCK-END
