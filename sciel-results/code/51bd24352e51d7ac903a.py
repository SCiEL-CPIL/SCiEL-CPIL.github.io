import time
import random
import numpy as np

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 5.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

def _hamming(a, b):
    return (a ^ b).bit_count()

def _check_min_distance(words, min_dist):
    n = len(words)
    # Optimization: for large n, this is O(n^2). 
    # We only call this on the final result to be safe.
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_dist:
                return False
    return True

def _generate_all_vectors_up_to_weight(N, max_weight):
    """
    Generate all N-bit vectors with weight <= max_weight.
    Returns a list of integers.
    """
    vectors = [0] # weight 0
    # Weights 1, 2, 3
    # Using combinations from itertools is efficient enough for N=18
    from itertools import combinations
    
    for w in range(1, max_weight + 1):
        for comb in combinations(range(N), w):
            v = 0
            for bit in comb:
                v |= (1 << bit)
            vectors.append(v)
    return vectors

def _expand_forbidden_set(word, N, min_dist, forbidden_set):
    """
    Update the forbidden set (a set of integers) with all words 
    within Hamming distance < min_dist of 'word'.
    """
    # For min_dist=6, we need to add all words at distance 1, 2, 3, 4, 5.
    # This is expensive if done naively for every word.
    # However, we are doing this in a greedy loop.
    # Optimization: We only need to check if a candidate is in the forbidden set.
    # Instead of expanding the forbidden set for every added word (which is huge),
    # we can just check the candidate against all existing words.
    # But checking against all existing words is O(N) per candidate.
    # Expanding forbidden set is O(2^d) per word added, which is ~32 for d=5? 
    # No, number of words within dist 5 is sum(C(18, i) for i=0..5) = 1 + 18 + 153 + 816 + 3060 + 8568 = ~12616.
    # This is too big to pre-compute and store for every word if we have 500 words.
    
    # Alternative Strategy for Speed:
    # 1. Precompute all "low weight" vectors (weight <= 3). 
    #    Any two vectors of weight <= 3 have distance <= 6? 
    #    Wait. If v1 has weight 3 and v2 has weight 3, and they are disjoint, distance is 6.
    #    If they share 1 bit, distance is 4 (bad).
    #    So we cannot just take ALL low weight vectors. We must pack them.
    
    # Let's stick to a robust Greedy with a "Forbidden Set" that is a BITSET of size 2^18?
    # 2^18 = 262,144 bits = 32 KB. This is very small!
    # We can maintain a global `forbidden` array of booleans or a bitarray.
    # Initially, forbidden[0] = True (if we start with 0).
    # When we add a word W, we mark all words within distance < 6 as forbidden.
    # To do this fast, we can precompute the "neighborhood" of radius 5 for any word?
    # No, the neighborhood depends on the word.
    
    # However, we can iterate through all 2^18 words and check distance?
    # 2^18 * 500 words = 131 million operations. Might be too slow in Python.
    
    # Better: Use the property that we are looking for a code.
    # Let's use a standard Greedy Independent Set on the graph where edges connect dist < 6.
    # But we don't want to build the graph.
    
    # Revised Strategy:
    # 1. Start with a strong seed (Reed-Muller or Golay shortening).
    # 2. Use a "Bitset Forbidden" approach but optimize the expansion.
    #    Actually, for N=18, we can just iterate candidates from 0 to 2^18-1.
    #    For each candidate, check if it's forbidden.
    #    If not, add it, and update forbidden set.
    #    Updating forbidden set: 
    #       For the new word W, find all X such that d(W, X) < 6.
    #       This can be done by iterating over all masks of weight 1..5 and XORing with W.
    #       Number of such masks is ~12600.
    #       If we add 500 words, total ops = 500 * 12600 = 6.3 million.
    #       This is very fast in Python!
    
    pass

def _reed_muller_extended():
    """
    Generate the [16, 5, 6] extended Reed-Muller code.
    """
    words = []
    for a in range(16):
        for b in range(2):
            word = 0
            for x in range(16):
                bit = ((a & x).bit_count() + b) % 2
                if bit:
                    word |= (1 << x)
            # Extend to 18 bits? The task says length 18.
            # The seed program generated 16-bit words? 
            # "Generate the [16, 5, 6] extended Reed-Muller code."
            # But N=18. 
            # The seed program's `_reed_muller_extended` returns 16-bit words?
            # Let's check the seed code: `word |= (1 << x)` where x is 0..15.
            # So it returns 16-bit words. 
            # But the task requires N=18.
            # The seed program likely pads or the evaluator handles it?
            # "Each integer encodes one N-bit binary codeword".
            # If I return 16-bit words, are they valid 18-bit words? 
            # Yes, leading zeros.
            # So a 16-bit word is effectively an 18-bit word with 2 leading zeros.
            return words
    return words

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()
    
    # 1. Precompute masks for distance < 6
    # We need to mark all neighbors of a word W.
    # Neighbors are W XOR mask, where weight(mask) < MIN_DIST.
    # So masks of weight 1, 2, 3, 4, 5.
    
    # Generate all masks of weight 1..5
    # Using itertools
    from itertools import combinations
    
    masks = []
    for w in range(1, MIN_DIST): # 1 to 5
        for comb in combinations(range(N), w):
            m = 0
            for bit in comb:
                m |= (1 << bit)
            masks.append(m)
            
    # Total masks: sum(C(18, i) for i=1..5) = 18 + 153 + 816 + 3060 + 8568 = 12615
    # This is a manageable list.
    
    # 2. Initialize Forbidden Bitset
    # Size 2^18 = 262144
    # Using a bytearray or list of booleans? 
    # A numpy array of uint8 is fast.
    # Or just a set of integers? 
    # Set of integers is O(1) lookup, but memory overhead is high.
    # 262,144 integers in a set is ~2MB. Fine.
    # But updating a set with 12k elements per word is O(12k).
    # Using a bitarray (numpy) might be faster for bulk operations?
    # Actually, simple list of booleans `forbidden = [False] * (1<<N)` is fastest for indexing in Python.
    
    max_val = 1 << N
    forbidden = [False] * max_val
    
    code = []
    
    # 3. Seed with a good code
    # Let's start with 0 (all zeros).
    # Then add all other words greedily?
    # Or start with Reed-Muller?
    # Reed-Muller has 32 words. 
    # If we start with RM, we mark their neighbors as forbidden.
    # Then we scan 0..2^18-1 for the first available word.
    
    # Let's try starting with just 0, and greedy packing.
    # This is a "Greedy Independent Set" on the graph.
    # The order matters. 
    # Standard greedy: sort candidates by some heuristic (e.g., weight) and add if possible.
    # Or just scan 0, 1, 2...
    # Scanning 0, 1, 2... tends to pick low-weight words first.
    # Low-weight words are "compatible" with many other low-weight words?
    # No, two weight-1 words have distance 2 (if distinct). So they conflict.
    # Weight 0 and Weight 1 have distance 1. Conflict.
    # So we can't just take all low weights.
    
    # Strategy: 
    # 1. Add 0. Mark neighbors.
    # 2. Iterate v from 1 to 2^18-1.
    #    If not forbidden[v]:
    #       Add v to code.
    #       Mark all neighbors of v as forbidden.
    
    # This is O(2^N * K) where K is number of words added * masks per word.
    # 2^18 * (1 + 1) + 500 * 12615 ~ 262k + 6.3M = 6.5M ops.
    # Very fast.
    
    # Let's run this greedy scan.
    
    # Add 0
    forbidden[0] = True
    code.append(0)
    
    # Mark neighbors of 0
    for m in masks:
        forbidden[m] = True
    # Note: 0 is already marked.
    
    # Scan
    for v in range(1, max_val):
        if time.time() > DEADLINE:
            break
            
        if not forbidden[v]:
            # Add v
            code.append(v)
            forbidden[v] = True
            
            # Mark neighbors
            for m in masks:
                neighbor = v ^ m
                forbidden[neighbor] = True
                
    # This simple greedy might yield a decent code, but maybe not optimal.
    # Can we do better?
    # The "Order" of scanning determines the result.
    # Scanning 0, 1, 2... is biased towards low weights.
    # Maybe we should use a randomized order?
    # Or multiple restarts with different orders?
    
    # Let's try a few different strategies:
    # 1. Standard Greedy (0, 1, 2...)
    # 2. Reverse Greedy (2^N-1, 2^N-2...)
    # 3. Random Greedy (shuffle order)
    
    best_code = code
    best_size = len(code)
    
    # Try Reverse
    if time.time() < DEADLINE - 10:
        forbidden = [False] * max_val
        code_rev = []
        
        # Start with max_val - 1
        start_v = max_val - 1
        forbidden[start_v] = True
        code_rev.append(start_v)
        for m in masks:
            forbidden[start_v ^ m] = True
            
        for v in range(max_val - 2, -1, -1):
            if time.time() > DEADLINE:
                break
            if not forbidden[v]:
                code_rev.append(v)
                forbidden[v] = True
                for m in masks:
                    forbidden[v ^ m] = True
                    
        if len(code_rev) > best_size:
            best_code = code_rev
            best_size = len(code_rev)
            
    # Try Randomized Greedy (a few times)
    rng = random.Random(42)
    for i in range(5):
        if time.time() > DEADLINE - 5:
            break
            
        forbidden = [False] * max_val
        code_rand = []
        
        # Create a shuffled list of all words? 
        # 2^18 = 262k integers. Shuffling takes time but is OK.
        # Or just pick random candidates until we find one?
        # "Pick random candidate" is slower if the code is dense.
        # Better: Shuffle the list of all 2^18 words.
        
        all_words = list(range(max_val))
        rng.shuffle(all_words)
        
        for v in all_words:
            if time.time() > DEADLINE:
                break
            if not forbidden[v]:
                code_rand.append(v)
                forbidden[v] = True
                for m in masks:
                    forbidden[v ^ m] = True
                    
        if len(code_rand) > best_size:
            best_code = code_rand
            best_size = len(code_rand)
            
    # Final Polish: Try to add more words to the best code found?
    # The greedy algorithm already scans all words, so it's "maximal" for that order.
    # We can't add more without removing some.
    # Local search (swap) could improve, but is complex.
    # Given the time, the greedy packing is likely strong.
    
    # Verification
    if not _check_min_distance(best_code, MIN_DIST):
        # Fallback
        return [0]
        
    return best_code

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")
    # Check distance
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if _hamming(res[i], res[j]) < 6:
                valid = False
                break
        if not valid: break
    print(f"Valid: {valid}")
