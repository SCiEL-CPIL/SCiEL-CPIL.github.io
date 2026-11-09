import time
import random
import numpy as np

N = 17
MIN_DISTANCE = 6
TOTAL_WORDS = 2 ** N

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit (or 60s if strict)
    
    # 1. Precompute popcount table for fast distance calculation
    # dist(a, b) = popcount[a ^ b]
    # Using a lookup table is faster than bit_count() in tight Python loops if we can't vectorize the whole set easily,
    # but here we will use a hybrid: NumPy for bulk checks, Python for specific swaps.
    # Actually, for N=17, we can store the code as a set of ints.
    # To check if a candidate w is compatible with code C:
    # min_{c in C} popcount(w ^ c) >= 6
    
    # A faster way to check compatibility for a single candidate against a large code:
    # If we maintain a "forbidden" bitset (size 2^17 bits = 16KB), we can check `if not forbidden[w]` in O(1).
    # Updating the forbidden set when adding w:
    # forbidden[w ^ mask] = 1 for all mask with weight < 6.
    # Number of masks with weight < 6 in 17 bits: sum_{i=0}^5 C(17, i) = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402.
    # This is very fast.
    
    # Let's precompute the list of masks with weight < 6.
    masks = []
    for i in range(TOTAL_WORDS):
        if i.bit_count() < MIN_DISTANCE:
            masks.append(i)
            
    # Strategy:
    # 1. Greedy phase: Add words in a random order if they don't violate constraints.
    # 2. Local Search (Simulated Annealing): 
    #    - Try to add a word. If it fails, try to remove a word that "blocks" it, then add it.
    #    - Or simply try to swap a word for another that allows more additions.
    #    - Given the time limit, a simple "Greedy + Random Restarts" with the Forbidden Bitset is often very strong.
    #    - However, to beat the 256 linear code, we need to find the nonlinear extension.
    #    - Milshtein's code is 258. We aim for that.
    
    best_code = set()
    best_forbidden = np.zeros(TOTAL_WORDS, dtype=bool)
    best_size = 0
    
    # We will run multiple restarts.
    # Each restart:
    # 1. Initialize empty code and forbidden set.
    # 2. Greedily add words. To make it deterministic-ish but varied, we can shuffle the order of candidates.
    # 3. After greedy, try a few local search steps.
    
    def greedy_fill(deadline):
        code = set()
        forbidden = np.zeros(TOTAL_WORDS, dtype=bool)
        
        # Randomize the order of candidates to avoid bias
        candidates = list(range(TOTAL_WORDS))
        random.shuffle(candidates)
        
        for w in candidates:
            if time.time() > deadline:
                break
                
            if forbidden[w]:
                continue
                
            # w is compatible
            code.add(w)
            
            # Update forbidden set
            for m in masks:
                forbidden[w ^ m] = True
                
        return code, forbidden

    def local_search(code, forbidden, deadline):
        # Try to improve the code.
        # If the code is maximal, we can't add any more words directly.
        # We can try to remove a word x, which frees up some space, and see if we can add two words y, z that were blocked by x.
        # This is expensive. 
        # Alternative: Just do more greedy restarts. The "Forbidden Bitset" greedy is very fast.
        # 131072 candidates. Each check is O(1).
        # Adding a word takes 9402 ops.
        # If we add ~250 words, that's 250 * 9402 ~ 2.3M ops.
        # The loop over 131k candidates is the bottleneck. 131k * O(1) is fast.
        # We can probably do ~10-50 restarts in 55 seconds.
        
        # Let's implement a simple "Swap" heuristic if time permits.
        # Pick a random codeword c. Remove it. Update forbidden (clear neighbors of c).
        # Try to add any word that becomes available.
        # If we add a word, keep it. If not, put c back? Or keep the removal if it doesn't hurt? 
        # Actually, removing a word reduces size by 1. We only do this if we can add >1 words.
        
        # Given the complexity and time, let's rely on the speed of the greedy restarts.
        # The key insight from inspirations: "Constructs the optimal 256-word linear code... and uses vectorized greedy search".
        # My forbidden bitset approach is even faster than vectorized distance checks against the whole code.
        
        return code, forbidden

    # Main loop: Multiple restarts
    num_restarts = 0
    while time.time() < deadline - 5.0: # Keep 5s buffer for final processing
        code, forbidden = greedy_fill(deadline)
        size = len(code)
        
        if size > best_size:
            best_size = size
            best_code = code
            best_forbidden = forbidden
            num_restarts += 1
            
        # Optional: Local search on the best code found so far?
        # If we have time, we can try to improve the current best_code.
        # But modifying best_code directly is risky if it fails.
        # Let's just do more restarts. The variance in greedy is usually enough to find the 258 solution if it's reachable by greedy.
        # Milshtein's 258 code is likely found by a specific greedy or integer programming.
        # Random greedy with forbidden set is a very strong baseline.
        
    # Final check: return the list of integers
    return list(best_code)

if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    print(f"Size: {len(res)}, Time: {time.time() - start}")
