# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code using a deterministic algebraic construction 
    followed by vectorized local search optimization.
    
    For q=3, n=8, radius=1:
    1. Start with a structured code derived from the ternary Golay code or 
       a linear code with good covering properties.
    2. Use a vectorized local search to remove redundant codewords.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave margin

    # Specific handling for the task parameters
    if q == 3 and n == 8 and radius == 1:
        # Strategy:
        # 1. Generate a strong initial code. 
        #    A known good construction for K_3(8,1) uses the product of a [4,2,3]_3 code 
        #    and a [4,2,3]_3 code? No, that gives 81*81=6561.
        #    The best known is 486. 
        #    486 = 2 * 3^5. 
        #    One construction: Take the ternary Golay code G(11,6) and shorten/puncture?
        #    Actually, a simpler approach for a strong seed:
        #    Use a linear code C of length 8 over F_3 with dimension k.
        #    If we pick a code with minimum distance d, it doesn't automatically cover.
        #    
        #    Let's try a constructive seed based on the "football pool" solution structure.
        #    A common strong seed is the set of all words of the form:
        #    (a, b, a+b, a+2b, c, d, c+d, c+2d) for a,b,c,d in F_3.
        #    This is the product of two [4,2,3] MDS codes.
        #    Size: 3^4 = 81 codewords.
        #    Does this cover?
        #    The [4,2,3] code over F_3 is a perfect code for radius 1?
        #    Ball size radius 1 in F_3^4 is 1 + 4*2 = 9.
        #    Space size 3^4 = 81.
        #    81/9 = 9. So a perfect code of size 9 exists.
        #    The [4,2,3] code has 3^2 = 9 codewords.
        #    Yes, the [4,2,3] MDS code over F_3 is a perfect 1-error correcting code.
        #    Therefore, the product code C = C4 x C4 is a perfect code for F_3^8 with radius 2?
        #    Wait. The radius of the product of two perfect radius-1 codes is 2?
        #    If x is in F_3^8, split into x1, x2 in F_3^4.
        #    There exists c1 in C4 with d(x1, c1) <= 1.
        #    There exists c2 in C4 with d(x2, c2) <= 1.
        #    Then d(x, c1||c2) <= 2.
        #    So C = C4 x C4 covers F_3^8 with radius 2.
        #    We need radius 1.
        #    So this seed (81 words) does NOT cover with radius 1.
        
        #    We need a code that covers with radius 1.
        #    Lower bound: 3^8 / (1 + 8*2) = 6561 / 17 ~ 386.
        #    Best known: 486.
        
        #    Let's try a different seed.
        #    Consider the code formed by taking the perfect code C4 (size 9) 
        #    and "lifting" it?
        
        #    Alternative: Greedy set cover is reliable but slow if not optimized.
        #    Let's implement a highly optimized vectorized greedy set cover.
        #    The key is to avoid the O(N^2) memory if possible, or use bitsets.
        #    N = 6561. N^2 = 43M. A boolean matrix of 43M is ~43MB. This is fine.
        
        #    Generate all words
        all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int8)
        num_words = len(all_words)
        
        # Precompute the "ball" for each word as a list of indices? 
        # Or a boolean matrix?
        # Boolean matrix is 6561 x 6561.
        # Let's build the incidence matrix M where M[i, j] = 1 if d(i, j) <= radius.
        
        # Efficient distance computation for radius 1:
        # d(i, j) <= 1 means they are equal or differ in exactly one position.
        
        # Initialize M as identity
        M = np.eye(num_words, dtype=bool)
        
        # For each position p, and each possible change, mark connections.
        # We can vectorize this.
        # For a fixed position p, we can create a permutation of indices that swaps values.
        
        # Let's use a different representation: 
        # For each word i, the words it covers are:
        # i itself, and for each p in 0..n-1, and each v in 0..q-1 (v != i[p]), 
        # the word j formed by replacing i[p] with v.
        
        # We can compute these neighbors efficiently.
        
        # Let's build the coverage matrix column by column or using broadcasting?
        # Broadcasting 6561x8x3 might be heavy but let's try a smarter way.
        
        # Method:
        # 1. Create a mapping from word tuple to index.
        word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
        
        # 2. For each word i, find its neighbors.
        #    To speed this up, we can iterate through positions and use vectorized lookups?
        #    Actually, 6561 * 8 * 2 = ~100k lookups. This is very fast in Python.
        
        neighbors = [[] for _ in range(num_words)]
        for i in range(num_words):
            neighbors[i].append(i) # self
            w = all_words[i]
            for p in range(n):
                for v in range(q):
                    if v == w[p]:
                        continue
                    # Create new word
                    # Instead of copying the whole array, we can compute the index directly
                    # Index of a word in base-q representation.
                    # all_words is generated in lexicographic order of product(range(q), repeat=n).
                    # This corresponds to base-q number where last element is least significant?
                    # product(range(q), repeat=n) generates (0,...,0), (0,...,1)...
                    # So index = sum(w[k] * q^(n-1-k)).
                    
                    # Let's verify index calculation.
                    # w = [w0, w1, ..., wn-1]
                    # idx = w0*q^(n-1) + w1*q^(n-2) + ... + wn-1
                    old_val = w[p]
                    new_val = v
                    # Change in index: (new_val - old_val) * q^(n-1-p)
                    delta_idx = (new_val - old_val) * (q ** (n - 1 - p))
                    j = i + delta_idx
                    neighbors[i].append(j)
        
        # Build sparse matrix or boolean matrix for fast coverage updates.
        # Since we do greedy, we need to know which words are covered.
        # We maintain a boolean array 'covered' of size num_words.
        # When we select a codeword i, we mark all neighbors[i] as covered.
        
        # To find the best next codeword, we need to count how many NEW words it covers.
        # new_covered_count(i) = sum(1 for j in neighbors[i] if not covered[j])
        
        # We can maintain an array 'uncovered_count' for each potential codeword?
        # No, that's what we are trying to compute.
        
        # Efficient Greedy:
        # 1. Maintain 'covered' boolean array.
        # 2. Maintain 'remaining_uncovered' count for each word? No, that's for the words themselves.
        # 3. For each candidate codeword i, we need count of neighbors not yet covered.
        
        # Let's use a vectorized approach for the greedy step.
        # If we have a boolean matrix M (6561x6561), we can do:
        # uncovered_mask = ~covered
        # scores = (M & uncovered_mask).sum(axis=1)
        # This is a matrix-vector multiplication or similar.
        # 6561x6561 matrix multiplication with a vector is fast in NumPy.
        
        # Let's build M as a dense boolean matrix.
        M = np.zeros((num_words, num_words), dtype=bool)
        for i in range(num_words):
            M[i, neighbors[i]] = True
            
        # Greedy Set Cover
        selected = []
        covered = np.zeros(num_words, dtype=bool)
        
        # Optimization: Precompute the sum of each row? No, it changes.
        
        while not covered.all() and time.time() < deadline:
            # Calculate score for each candidate
            # M is (N, N), uncovered is (N,)
            # We want for each row i, sum of M[i, :] where uncovered is True.
            # This is equivalent to M @ uncovered? No, M is boolean.
            # (M & uncovered[None, :]).sum(axis=1)
            
            # To avoid creating a huge temporary array (N x N) every iteration,
            # we can use the property that M is fixed.
            # However, N=6561, N^2=43M. 43M booleans is 43MB. 
            # Creating this temp array 486 times might be slow.
            
            # Alternative: Maintain a counter for each codeword.
            # Let 'potential[i]' be the number of uncovered words covered by codeword i.
            # Initially potential[i] = len(neighbors[i]) = 17 (for radius 1, q=3, n=8).
            # Wait, is it always 17?
            # Ball size = 1 + n*(q-1) = 1 + 8*2 = 17.
            # Yes, for any word in q-ary space, the ball size is constant.
            
            # So initially, all codewords cover 17 words.
            # When we select a codeword c, we mark its 17 neighbors as covered.
            # For every other codeword i that shares a neighbor with c, its potential decreases.
            
            # We can maintain 'potential' array.
            # When we select c:
            # 1. covered[neighbors[c]] = True
            # 2. For each neighbor j of c:
            #    For each codeword i that covers j (i.e., j in neighbors[i]):
            #       potential[i] -= 1
            #       (if it was previously uncovered)
            
            # This requires knowing, for each word j, which codewords cover it.
            # This is the transpose of M. Let M_T = M.T.
            # The set of codewords covering j is M_T[:, j] (or M[j, :] since M is symmetric? 
            # Distance is symmetric, so M is symmetric. M[i,j]=1 iff M[j,i]=1).
            
            # So, when we cover word j:
            # We need to decrement potential[i] for all i such that M[i, j] == 1.
            # i.e., for all i in neighbors[j].
            
            # Algorithm:
            potential = np.full(num_words, 17, dtype=np.int32) # Initial potential
            covered = np.zeros(num_words, dtype=bool)
            selected = []
            
            # To pick the best, we need argmax(potential).
            # Ties can be broken arbitrarily or by index.
            
            while not covered.all() and time.time() < deadline:
                # Find best candidate
                best_idx = np.argmax(potential)
                if potential[best_idx] == 0:
                    break # Should not happen if coverage is possible
                
                selected.append(best_idx)
                
                # Update covered and potentials
                # Get neighbors of best_idx
                nbrs = neighbors[best_idx]
                
                # Mark as covered
                for j in nbrs:
                    if not covered[j]:
                        covered[j] = True
                        # Decrement potential of all codewords that cover j
                        # These are the neighbors of j
                        # We only care about codewords not yet selected?
                        # Actually, if a codeword is already selected, we don't care about its potential.
                        # But we can just decrement it, it won't be selected again (we can mask it).
                        
                        for i in neighbors[j]:
                            potential[i] -= 1
                
                # Optional: Set potential of selected codewords to 0 to avoid re-selection
                # (Though greedy set cover usually doesn't re-select, argmax might pick it if potential is high? 
                # No, its potential would have dropped, but let's be safe).
                potential[best_idx] = -1

            codewords = [all_words[i].tolist() for i in selected]
            
            # Local Search: Try to remove redundant codewords
            # If we remove codeword c, do we still cover all words?
            # This is equivalent to checking if the union of balls of other codewords covers the space.
            # Or simpler: Check if any word is ONLY covered by c.
            # If a word j is covered only by c, we cannot remove c.
            
            # We can maintain a 'cover_count' array.
            # cover_count[j] = number of selected codewords that cover j.
            # If min(cover_count) >= 2, we can potentially remove any codeword? No.
            # If min(cover_count) >= 1, the code is valid.
            # To remove c, we need to ensure that for all j in neighbors[c], cover_count[j] > 1.
            
            # Let's build cover_count.
            cover_count = np.zeros(num_words, dtype=np.int32)
            for i in selected:
                for j in neighbors[i]:
                    cover_count[j] += 1
            
            # Now try to remove codewords.
            # We can iterate through selected codewords and try to remove them.
            # If removing c keeps min(cover_count) >= 1 (after decrementing for c's neighbors), remove it.
            
            # To speed up, we can sort selected codewords by "criticality" or just iterate.
            # A good heuristic: try to remove codewords that cover the most "redundant" words.
            # Or just iterate through the list.
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for idx_in_selected in range(len(selected)):
                    if time.time() > deadline - 5:
                        break
                    
                    c = selected[idx_in_selected]
                    nbrs_c = neighbors[c]
                    
                    # Check if any neighbor is covered only by c
                    # i.e., cover_count[j] == 1 for any j in nbrs_c
                    # If so, c is essential.
                    
                    # Vectorized check:
                    # counts = cover_count[nbrs_c]
                    # if np.any(counts == 1): continue
                    
                    # If no neighbor has count 1, we can remove c.
                    if not np.any(cover_count[nbrs_c] == 1):
                        # Remove c
                        for j in nbrs_c:
                            cover_count[j] -= 1
                        selected.pop(idx_in_selected)
                        improved = True
                        break # Restart loop as indices changed
            
            # Final verification
            final_covered = np.zeros(num_words, dtype=bool)
            for i in selected:
                final_covered[neighbors[i]] = True
            
            if not final_covered.all():
                # Fallback if something went wrong (should not happen)
                return all_words.tolist()
            
            return codewords

    # General fallback
    return _greedy_cover_general(q, n, radius, deadline)

def _greedy_cover_general(q, n, radius, deadline):
    """General greedy set cover for arbitrary q, n, radius."""
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int8)
    num_words = len(all_words)
    
    if num_words > 10000:
        # Fallback for large spaces: return a sparse grid
        # This is a heuristic, might not be optimal but valid if dense enough
        # For radius 1, we need density roughly 1/(1+n(q-1))
        # This is complex. Just return first N words? No, must cover.
        # For general case, if too big, we might timeout.
        # Let's just return all words to be safe, though score will be bad.
        return all_words.tolist()

    # Compute neighbors
    neighbors = [[] for _ in range(num_words)]
    for i in range(num_words):
        neighbors[i].append(i)
        w = all_words[i]
        for p in range(n):
            for v in range(q):
                if v == w[p]:
                    continue
                delta_idx = (v - w[p]) * (q ** (n - 1 - p))
                j = i + delta_idx
                neighbors[i].append(j)

    # Greedy with potential tracking
    # Ball size might vary if radius > 1 or boundaries? 
    # In Z_q^n with Hamming distance, ball size is constant for all words.
    ball_size = 1
    # Calculate ball size
    for r in range(1, radius + 1):
        # C(n, r) * (q-1)^r
        ball_size += int(np.math.comb(n, r)) * ((q - 1) ** r)
        
    potential = np.full(num_words, ball_size, dtype=np.int32)
    covered = np.zeros(num_words, dtype=bool)
    selected = []
    
    while not covered.all() and time.time() < deadline:
        best_idx = np.argmax(potential)
        if potential[best_idx] == 0:
            break
        
        selected.append(best_idx)
        nbrs = neighbors[best_idx]
        
        for j in nbrs:
            if not covered[j]:
                covered[j] = True
                for i in neighbors[j]:
                    potential[i] -= 1
        
        potential[best_idx] = -1

    # Local search
    cover_count = np.zeros(num_words, dtype=np.int32)
    for i in selected:
        for j in neighbors[i]:
            cover_count[j] += 1
            
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx_in_selected in range(len(selected)):
            if time.time() > deadline - 5:
                break
            c = selected[idx_in_selected]
            nbrs_c = neighbors[c]
            if not np.any(cover_count[nbrs_c] == 1):
                for j in nbrs_c:
                    cover_count[j] -= 1
                selected.pop(idx_in_selected)
                improved = True
                break

    return [all_words[i].tolist() for i in selected]
# EVOLVE-BLOCK-END
