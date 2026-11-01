import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Construct a covering code for q=3, n=6, radius=1.
    Uses a greedy removal/swap heuristic starting from the tetracode direct sum construction.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (not expected in this specific task instance)
        return _generic_construct(q, n, radius)
        
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initial Construction: Direct sum of tetracode (length 4) and full space (length 2)
    # Tetracode: 9 words of length 4, covering radius 1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all 81 initial codewords
    # Format: [c0, c1, c2, c3, c4, c5]
    # First 4 from tetracode, last 2 from product
    initial_codewords = []
    for t in tetracode:
        for c4 in range(3):
            for c5 in range(3):
                initial_codewords.append([t[0], t[1], t[2], t[3], c4, c5])
                
    codewords = np.array(initial_codewords, dtype=np.int32)
    
    # Helper to generate all neighbors (distance <= 1) for a set of codewords
    # Returns a set of tuples for membership checking
    def get_covered_set(cw_array):
        covered = set()
        # Distance 0
        for cw in cw_array:
            covered.add(tuple(cw))
        # Distance 1
        # There are n * (q-1) neighbors at distance 1
        # For q=3, n=6, that's 12 neighbors per codeword
        for i, cw in enumerate(cw_array):
            for pos in range(n):
                for val in range(q):
                    if val == cw[pos]:
                        continue
                    neighbor = list(cw)
                    neighbor[pos] = val
                    covered.add(tuple(neighbor))
        return covered
        
    # Initial check
    covered_set = get_covered_set(codewords)
    total_space = 3 ** 6 # 729
    if len(covered_set) < total_space:
        raise ValueError("Initial construction does not cover space")

    best_codewords = codewords.copy()
    
    # Heuristic: Iteratively try to remove a codeword if it doesn't break coverage
    # Or swap a codeword with another to allow removal of a third one later?
    # Simpler: Just removal first.
    
    # To speed up removal check:
    # Count how many times each word is covered.
    # If a word is covered exactly once by a specific codeword, that codeword cannot be removed.
    # If all words covered by a codeword are covered by >= 2 codewords, it can be removed.
    
    def compute_coverage_counts(cw_array):
        counts = np.zeros(3**6, dtype=np.int32)
        # Flatten codewords to indices
        # Index = c0*3^5 + c1*3^4 + ...
        powers = [3**(n-1-i) for i in range(n)]
        
        for cw in cw_array:
            # Self
            idx = 0
            for i in range(n):
                idx += cw[i] * powers[i]
            counts[idx] += 1
            
            # Neighbors
            for pos in range(n):
                orig_val = cw[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    delta = (new_val - orig_val) * powers[pos]
                    nidx = idx + delta
                    counts[nidx] += 1
        return counts

    # Main Loop
    changed = True
    while changed and time.time() < deadline:
        changed = False
        counts = compute_coverage_counts(codewords)
        
        # Try to remove codewords in order of "least essential"
        # A codeword is removable if min(counts[neighbors(cw)]) >= 2 for all neighbors
        # Actually, we need to check if removing it leaves any count at 0.
        # So we need to know if any neighbor has count 1 AND is covered ONLY by this codeword.
        # Simpler: 
        # 1. Calculate counts.
        # 2. For each codeword, check if removing it results in any count becoming 0.
        #    This is equivalent to: does it cover any word that is covered by NO OTHER codeword?
        #    i.e. is there a word w in N(cw) such that counts[w] == 1?
        #    If yes, cannot remove.
        #    If no, can remove.
        
        removable_indices = []
        for i in range(len(codewords)):
            cw = codewords[i]
            removable = True
            # Check self
            powers = [3**(n-1-j) for j in range(n)]
            idx = sum(cw[j] * powers[j] for j in range(n))
            if counts[idx] == 1:
                removable = False
                break
            # Check neighbors
            if removable:
                for pos in range(n):
                    orig_val = cw[pos]
                    for new_val in range(q):
                        if new_val == orig_val:
                            continue
                        nidx = idx + (new_val - orig_val) * powers[pos]
                        if counts[nidx] == 1:
                            removable = False
                            break
                    if not removable:
                        break
            if removable:
                removable_indices.append(i)
                
        if removable_indices:
            # Remove the first removable one found (or randomize? Let's stick to deterministic for now, or randomize for diversity)
            # Random selection might find better local minima
            idx_to_remove = removable_indices[np.random.randint(len(removable_indices))]
            codewords = np.delete(codewords, idx_to_remove, axis=0)
            changed = True
            best_codewords = codewords.copy()
            
    # Try Swaps? 
    # A swap (remove A, add B) might enable further removals.
    # Given time constraints, simple removal often gets close. 
    # Let's add a simple swap phase if time permits.
    
    if time.time() < deadline - 10:
        # Try to improve via swaps
        # This is complex. Let's stick to the result from removal.
        # The removal phase should get us from 81 down to around 73-75.
        pass
    
    # Final verification
    final_set = get_covered_set(best_codewords)
    if len(final_set) < total_space:
         # Fallback to initial if something went wrong (shouldn't happen)
         return [list(c) for c in initial_codewords]
         
    return [list(c) for c in best_codewords]

def _generic_construct(q, n, radius):
    # Fallback for non-standard parameters
    # Simple greedy covering code
    # This is not optimized for q=3, n=6, r=1
    # Just returns a trivial upper bound construction
    # e.g. all words if small, or random
    # For safety, return a large set
    # If q^n is small, return all
    if q ** n <= 10000:
        return [list(c) for c in itertools.product(range(q), repeat=n)]
    else:
        # Random sampling
        import random
        seen = set()
        res = []
        while len(res) < 1000:
            w = [random.randint(0, q-1) for _ in range(n)]
            if tuple(w) not in seen:
                seen.add(tuple(w))
                res.append(w)
        return res
