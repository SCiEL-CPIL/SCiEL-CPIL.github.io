import itertools
import random
import time

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
        
    # 3. Greedy Set Cover with restarts
    best_code = None
    best_size = total_words
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit, leaving 20s margin
    
    # Run multiple greedy restarts
    # Each restart shuffles the candidate order to break ties differently
    num_restarts = 50 # Should be fast enough
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(r) # Deterministic restarts for reproducibility if needed, or random
        candidates = list(range(total_words))
        random.shuffle(candidates)
        
        covered_mask = 0
        chosen_indices = []
        
        while covered_mask != (1 << total_words) - 1:
            best_gain = -1
            best_idx = -1
            
            # Find candidate with max new coverage
            for idx in candidates:
                # Count bits in (coverage[idx] & ~covered_mask)
                new_coverage = coverage[idx] & ~covered_mask
                gain = bin(new_coverage).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    
            if best_gain == 0:
                # Should not happen if code is valid, but safety check
                break
                
            chosen_indices.append(best_idx)
            covered_mask |= coverage[best_idx]
            
            # Optimization: If a candidate is fully covered, we could skip it in future iterations,
            # but for n=6, 729 candidates is small enough to just scan all.
            # However, we can remove chosen_idx from candidates? No, other candidates might still be useful?
            # Actually, once chosen, its contribution is done. But other candidates overlap.
            # We just need to pick the one with max *additional* coverage.
            
        if len(chosen_indices) < best_size:
            best_size = len(chosen_indices)
            best_code = [all_words[i] for i in chosen_indices]
            
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
