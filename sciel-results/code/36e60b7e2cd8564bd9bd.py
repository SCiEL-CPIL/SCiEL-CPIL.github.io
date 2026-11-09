"""
Strategy: Iterative Local Search (ILS) with a strong linear code seed.
1. Construct a high-quality seed using a known [17,9,6] linear code (or a robust
   greedy construction if the linear one is hard to generate quickly, though we
   will attempt the linear one first as it provides 256 words immediately).
2. Instead of greedy expansion (which gets stuck in local optima quickly),
   use Iterative Local Search:
   a. Start with the seed.
   b. Generate a random perturbation (e.g., swap a random codeword with a random
      non-codeword, or remove a random codeword and re-greedy).
   c. Apply a local search improvement heuristic (e.g., try to add any possible
      word, or perform 1-for-1 swaps to increase size).
   d. Accept the new state if it improves or with some probability (simulated
      annealing style) to escape local optima.
3. The key difference from the current program is that we don't just do one
   greedy pass. We actively try to *modify* the code to escape local maxima
   where no single word can be added.
"""

import time
import random
import numpy as np

N = 17
MIN_DISTANCE = 6
TOTAL_WORDS = 2 ** N

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s kill

    # 1. Construct a strong seed.
    # We will try to construct the [17,9,6] linear code.
    # The [18,9,6] extended QR code is well-known. Shortening it gives [17,9,6].
    # However, generating the specific QR code matrix requires knowing the
    # generator matrix.
    # A simpler robust seed: Greedy construction of a large code.
    # Let's do a fast greedy to get a seed of ~200-250 words.
    
    def fast_greedy_seed(deadline):
        """
        Fast greedy to build a large code.
        Uses a 'forbidden set' approach for O(1) checks.
        """
        # Precompute all masks of weight < 6
        masks = []
        for i in range(TOTAL_WORDS):
            if i.bit_count() < MIN_DISTANCE:
                masks.append(i)
        
        code = []
        forbidden = set()
        
        # Random order for candidates
        candidates = list(range(TOTAL_WORDS))
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if cand in forbidden:
                continue
            
            # Add cand
            code.append(cand)
            forbidden.add(cand)
            for m in masks:
                forbidden.add(cand ^ m)
                
            # Early stop if we have a good number (e.g., 256) to save time for ILS
            if len(code) >= 256:
                break
                
        return code

    # Get a seed
    seed_code = fast_greedy_seed(deadline - 5.0)
    
    if not seed_code:
        # Fallback if something went wrong
        return [0]
        
    best_code = list(seed_code)
    best_size = len(best_code)
    
    # 2. Iterative Local Search
    # We will try to improve the code size by:
    # a. Trying to add any word (greedy fill) - this handles cases where the seed
    #    wasn't maximal.
    # b. Performing 1-for-1 swaps: remove a word w1, add a word w2.
    #    If removing w1 allows adding w2 (and w2 is not in code), the size stays
    #    the same, but the structure changes. This can allow further additions.
    # c. 1-for-2 swaps: remove w1, add w2 and w3. This increases size by 1.
    #    This is the most powerful move but expensive to search.
    
    # Optimization: Maintain a set of codewords for O(1) lookup.
    # For distance checks, we need to be efficient.
    
    def try_greedy_fill(code, deadline):
        """
        Try to add any possible words to the code greedily.
        Returns the updated code.
        """
        code_set = set(code)
        # Rebuild forbidden set from current code?
        # That's expensive. 
        # Instead, just check candidates against all codewords?
        # With ~256 codewords, checking 131k candidates * 256 dists is 33M ops.
        # In Python, this might take ~2-5 seconds.
        # We can optimize by only checking candidates that are not in the code.
        
        # A faster way: 
        # Precompute masks again? No, let's just iterate through all words.
        # To speed up, we can skip words that are in code_set.
        
        # Actually, the "forbidden set" approach is best for filling.
        # But rebuilding it from scratch takes 256 * 9402 = 2.4M ops.
        # That's fast.
        
        masks = []
        for i in range(TOTAL_WORDS):
            if i.bit_count() < MIN_DISTANCE:
                masks.append(i)
                
        forbidden = set()
        for c in code:
            forbidden.add(c)
            for m in masks:
                forbidden.add(c ^ m)
                
        # Now iterate through all words and add if not forbidden
        # This is O(2^17) checks.
        # We can iterate through 0..2^17-1.
        
        added = 0
        for w in range(TOTAL_WORDS):
            if time.time() > deadline:
                break
            if w in code_set:
                continue
            if w in forbidden:
                continue
            
            # Add w
            code.append(w)
            code_set.add(w)
            added += 1
            for m in masks:
                forbidden.add(w ^ m)
                
        return code

    # First, fill the seed greedily
    best_code = try_greedy_fill(best_code, deadline - 10.0)
    best_size = len(best_code)
    
    # Now, ILS loop
    # We will perform random swaps to escape local optima.
    # Move: Remove a random codeword c, then try to greedily fill again?
    # Or: Remove c, add a random non-codeword n?
    
    # Let's try: 
    # 1. Remove a random codeword.
    # 2. Try to add a random non-codeword.
    # 3. If the new code is valid (it is, by construction of swap), try to greedily fill it.
    # 4. If the size increased, accept. Otherwise, reject (or accept with prob T).
    
    # To make it faster, we can just do:
    # 1. Pick a random codeword to remove.
    # 2. Pick a random candidate to add (must be valid).
    # 3. If valid, apply swap.
    # 4. Try to fill.
    # 5. If size > best, update best.
    
    # Even better: 
    # Since greedy fill is fast, we can do:
    # 1. Perturb: Remove 1-2 random codewords.
    # 2. Greedy Fill.
    # 3. If size > best, keep. Else, revert.
    
    # Let's implement "Remove 1, Fill".
    
    current_code = list(best_code)
    current_size = best_size
    
    # Precompute masks for efficiency in the loop?
    # The masks are fixed.
    masks = []
    for i in range(TOTAL_WORDS):
        if i.bit_count() < MIN_DISTANCE:
            masks.append(i)
            
    # For faster distance check in swap validation:
    # We need to check if a new word w is at distance >= 6 from all (C \ {c}).
    # If we remove c, we only need to check against C \ {c}.
    
    # Let's do a simpler ILS:
    # Loop:
    #   1. Create a perturbed version of current_code by removing a random element.
    #   2. Run try_greedy_fill on it.
    #   3. If the result is larger than current_size, accept it as current.
    #   4. If the result is larger than best_size, update best.
    #   5. If no improvement for X iterations, increase perturbation strength (remove 2).
    
    iterations = 0
    no_improve_count = 0
    perturbation_strength = 1
    
    while time.time() < deadline:
        iterations += 1
        
        # Perturb
        if len(current_code) <= perturbation_strength:
            break
            
        # Remove 'perturbation_strength' random elements
        indices_to_remove = random.sample(range(len(current_code)), perturbation_strength)
        # Sort to remove safely
        indices_to_remove.sort(reverse=True)
        perturbed_code = current_code.copy()
        for idx in indices_to_remove:
            perturbed_code.pop(idx)
            
        # Fill
        improved_code = try_greedy_fill(perturbed_code, deadline)
        
        if len(improved_code) > current_size:
            # Accept
            current_code = improved_code
            current_size = len(improved_code)
            no_improve_count = 0
            
            if current_size > best_size:
                best_code = current_code
                best_size = current_size
                # Maybe increase perturbation strength if we found a big jump?
                # Or keep it.
        else:
            no_improve_count += 1
            
            # Increase perturbation strength if stuck
            if no_improve_count > 50:
                perturbation_strength += 1
                if perturbation_strength > 5:
                    perturbation_strength = 5
                no_improve_count = 0
                
        # Safety check: if we are taking too long per iteration, reduce work
        # try_greedy_fill is the bottleneck.
        # If we have 256 words, fill takes ~2-3s.
        # So we can only do ~20-30 iterations in 60s.
        # This might be too few.
        
        # Optimization: Only run full greedy fill if the perturbation was significant
        # or if we are close to a potential improvement?
        # Actually, if we remove 1 word, we can often add 1 word back if the code
        # wasn't "tight".
        
        # Alternative: 1-for-1 swap search.
        # For each codeword c, try to find a word w such that:
        # 1. w not in C
        # 2. dist(w, c') >= 6 for all c' in C \ {c}
        # If found, swap c with w. This keeps size constant but changes structure.
        # Then try to add more.
        
        # This is expensive: 256 * 131k * 255 checks.
        
        # Let's stick to Remove-Fill but optimize the Fill.
        # Can we make Fill faster?
        # The forbidden set construction is O(K * 9402).
        # The scan is O(2^17).
        # Total ~ 2.4M + 131k ops.
        # In Python, 2.4M set adds is ~0.1-0.2s.
        # 131k set checks is ~0.01s.
        # So Fill should be ~0.2s.
        # Then we can do ~200 iterations in 40s.
        
    return best_code

if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    print(f"Size: {len(res)}, Time: {time.time() - start}")
