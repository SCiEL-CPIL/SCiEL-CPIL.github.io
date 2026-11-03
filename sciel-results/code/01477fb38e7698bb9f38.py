import numpy as np
import time
import itertools

# Precomputation for q=3, n=7
Q = 3
N = 7
RADIUS = 1
TOTAL_WORDS = Q ** N  # 2187

# Precompute neighbors matrix M where M[i, j] = 1 if dist(i, j) <= RADIUS
# We can generate this efficiently.
# Convert integer to base-3 digits
def int_to_base3(num, length):
    digits = [0] * length
    for i in range(length):
        digits[i] = num % Q
        num //= Q
    return digits

def hamming_distance(a, b):
    return sum(1 for i in range(N) if a[i] != b[i])

# Build M
# M is a 2D numpy array of booleans, shape (TOTAL_WORDS, TOTAL_WORDS)
M = np.zeros((TOTAL_WORDS, TOTAL_WORDS), dtype=bool)
words_list = list(range(TOTAL_WORDS))
base3_words = [int_to_base3(w, N) for w in words_list]

for i in range(TOTAL_WORDS):
    wi = base3_words[i]
    # i itself
    M[i, i] = True
    # neighbors at distance 1
    for pos in range(N):
        orig_val = wi[pos]
        for new_val in range(Q):
            if new_val == orig_val:
                continue
            # Create new word
            wj = wi.copy()
            wj[pos] = new_val
            # Convert back to int
            val = 0
            for k in range(N):
                val = val * Q + wj[k]
            M[i, val] = True

# Greedy Construction
def greedy_construct():
    uncovered = np.ones(TOTAL_WORDS, dtype=bool)
    code = []
    
    # To speed up greedy, we can precompute the list of neighbors for each word
    # neighbors[i] = list of indices j such that M[i, j] is True
    neighbors = [np.nonzero(M[i])[0] for i in range(TOTAL_WORDS)]
    
    while np.any(uncovered):
        best_word = -1
        max_covered = 0
        
        # Find word in uncovered that covers most uncovered words
        # This is O(TOTAL_WORDS * avg_degree)
        # We can optimize by only checking uncovered words
        uncovered_indices = np.nonzero(uncovered)[0]
        
        for w in uncovered_indices:
            # Count how many neighbors of w are in uncovered
            # neighbors[w] is an array of indices
            # We need count of neighbors[w] that are True in uncovered
            # Using numpy: np.sum(uncovered[neighbors[w]])
            cnt = np.sum(uncovered[neighbors[w]])
            if cnt > max_covered:
                max_covered = cnt
                best_word = w
        
        if best_word == -1:
            break
            
        code.append(best_word)
        # Update uncovered
        uncovered[neighbors[best_word]] = False
        
    return code

def local_search(code, time_limit):
    # code is a list of integers
    # cover_count[i] = number of codewords covering word i
    # code_mask[i] = 1 if i in code
    
    code_set = set(code)
    code_mask = np.zeros(TOTAL_WORDS, dtype=bool)
    for c in code:
        code_mask[c] = True
        
    # Initial cover count
    cover_count = np.sum(M[code_mask], axis=0)
    
    # Precompute neighbor arrays for fast access
    # neighbors[i] is already computed in greedy, but let's ensure it's available
    # We'll use the global neighbors list
    
    start_time = time.time()
    
    # 1. Removal phase
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit - 10:
            break
        for c in list(code):
            if time.time() - start_time > time_limit - 10:
                break
            # Check if removing c keeps cover_count > 0
            # New cover count = cover_count - M[:, c]
            # We only need to check if min(cover_count - M[:, c]) > 0
            # But we only care about words that were covered by c
            # Actually, just check np.all(cover_count - M[:, c] > 0)
            # Optimization: M[:, c] is the column for c.
            # We can check if any word has cover_count == 1 and is a neighbor of c.
            # If yes, c is critical.
            # If no, c is redundant.
            
            # Let's do the full check for simplicity, it's fast enough with numpy
            # M[:, c] is a boolean vector
            new_counts = cover_count - M[:, c].astype(int)
            if np.all(new_counts > 0):
                # Remove c
                code.remove(c)
                code_mask[c] = False
                cover_count = new_counts
                changed = True
                # Restart loop to check for more removals
                break
                
    # 2. Swap phase
    # We try to swap a codeword c with a non-codeword w
    # Such that the new code covers the space
    # AND allows removal of some other codeword c'
    
    # To make this efficient, we iterate through current codewords
    # and potential replacements
    
    # We need a way to quickly find redundant codewords in the new state
    # Let's just do the swap, check coverage, then try removals.
    
    # To avoid O(N^2) swaps, we can prioritize swaps that cover "rare" words.
    # But for now, random or systematic sweep.
    
    # Systematic sweep of c in code
    # For each c, try all w not in code
    # This is |C| * (TOTAL - |C|) iterations.
    # 190 * 2000 = 380,000 iterations.
    # Each iteration:
    # 1. Compute new_counts = cover_count - M[:, c] + M[:, w]
    # 2. Check np.all(new_counts > 0)
    # 3. If yes, try to remove any c' from (code \ {c}) \ {w}
    #    For each c', check if new_counts - M[:, c'] > 0
    #    This is expensive inside the loop.
    
    # Optimization: 
    # Only accept swap if it reduces size.
    # So we must find a c' that becomes redundant.
    
    # Let's modify the inner loop to be smarter.
    # Instead of checking all c', we can identify which words are "critically covered" by the new code.
    # A word x is critically covered if new_counts[x] == 1.
    # The codeword covering x must be the one we just added w, or one of the existing ones.
    # If new_counts[x] == 1, then the codeword covering x is critical.
    # If the critical codeword is w, then w is critical (bad for removal of w, but we are keeping w).
    # If the critical codeword is c', then c' is critical.
    # If NO word has new_counts == 1, then all codewords are redundant? No, that means every word is covered by >= 2 codewords.
    # In that case, ANY codeword can be removed.
    
    # So:
    # 1. Compute new_counts.
    # 2. Check min(new_counts) > 0.
    # 3. If min(new_counts) >= 2, then we can remove ANY codeword from the rest. So size decreases by 1.
    # 4. If min(new_counts) == 1, find the words x where new_counts[x] == 1.
    #    For each such x, find the codeword c_x that covers it (new_counts[x]==1 implies only one codeword covers x).
    #    If c_x is w, then w is critical. We cannot remove w.
    #    If c_x is some c', then c' is critical.
    #    We can remove a codeword c'' if c'' is NOT critical.
    #    So if there exists a c'' in (code \ {c}) \ {w} that is not critical, we can remove it.
    
    # How to find c_x?
    # We can precompute for each word x, the list of codewords covering it.
    # But this list changes.
    # Alternative: 
    # If min(new_counts) == 1, let critical_words = np.where(new_counts == 1)[0].
    # For each x in critical_words:
    #   Find c_x such that M[x, c_x] is True and c_x is in new_code.
    #   Since new_counts[x] == 1, there is exactly one such c_x.
    #   We can find it by checking which codeword in new_code has M[x, c] == True.
    #   This is O(|C|) per critical word.
    #   Number of critical words is small?
    
    # Let's implement this.
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit - 5:
            break
        
        # Create a list of current codewords to iterate over
        # We might modify code, so we work on a copy or handle carefully
        current_code = list(code)
        
        for c in current_code:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Try replacing c with w
            # To speed up, we can sample w or iterate all
            # Iterating all 2187 is fine if inner loop is fast
            
            # We need to skip w if w is in code
            # code_mask is updated
            
            for w in range(TOTAL_WORDS):
                if code_mask[w]:
                    continue
                    
                # Compute new_counts
                # cover_count - M[:, c] + M[:, w]
                # To avoid creating new arrays, we can do:
                # new_counts = cover_count.copy()
                # new_counts -= M[:, c]
                # new_counts += M[:, w]
                # This is 3 array ops of size 2187. Very fast.
                
                new_counts = cover_count - M[:, c].astype(int) + M[:, w].astype(int)
                
                if not np.all(new_counts > 0):
                    continue
                
                # Check if we can remove some c'
                # Case 1: min(new_counts) >= 2
                if np.min(new_counts) >= 2:
                    # We can remove any c' from current_code \ {c}
                    # Let's pick the first one
                    # Actually, we should pick one that is "least useful" to maintain robustness?
                    # For now, just remove the first one found
                    for c_prime in current_code:
                        if c_prime == c:
                            continue
                        # Remove c_prime
                        # Update state
                        code.remove(c_prime)
                        code_mask[c_prime] = False
                        # Update cover_count to new_counts - M[:, c_prime]
                        cover_count = new_counts - M[:, c_prime].astype(int)
                        # Add w
                        code.append(w)
                        code_mask[w] = True
                        # Note: cover_count is already updated for w in new_counts
                        # But we subtracted c_prime.
                        # So current cover_count corresponds to (code \ {c, c_prime}) \ {c} + {w} - {c_prime} ?
                        # Let's trace:
                        # Old code C.
                        # New code C' = (C \ {c}) \ {c_prime} + {w}.
                        # cover_count_old = sum(M[:, C])
                        # new_counts = cover_count_old - M[:, c] + M[:, w]
                        # final_counts = new_counts - M[:, c_prime]
                        # This matches C'.
                        
                        improved = True
                        # Break out of loops to restart search with new code
                        break
                    if improved:
                        break
                        
                else:
                    # min(new_counts) == 1
                    # Find critical words
                    critical_words = np.where(new_counts == 1)[0]
                    
                    # Identify critical codewords
                    # A codeword x is critical if it is the ONLY one covering some word.
                    # Let's find which codewords are critical.
                    critical_codewords = set()
                    
                    # We need to map each critical word to its covering codeword in C'
                    # C' = (C \ {c}) + {w}
                    # For each x in critical_words:
                    #   Find y in C' such that M[x, y] is True.
                    #   Since new_counts[x] == 1, there is exactly one such y.
                    
                    # To make this fast, we can precompute for each word x, the list of neighbors.
                    # But we need to check which neighbor is in C'.
                    # This is O(deg(x)) per critical word.
                    
                    for x in critical_words:
                        # Find y in C' covering x
                        # We can iterate through C'
                        # C' is small (~190)
                        # neighbors[x] is ~22
                        # Intersection of neighbors[x] and C'
                        # Since new_counts[x] == 1, the intersection has size 1.
                        
                        # Let's just check w and the other codewords
                        # Check w first
                        if M[x, w]:
                            critical_codewords.add(w)
                        else:
                            # Must be one of the others
                            # We can find it by checking which codeword in C \ {c} covers x
                            # We can precompute a list of codewords for each word? No, it changes.
                            # Just iterate C \ {c}
                            found = False
                            for c_other in current_code:
                                if c_other == c:
                                    continue
                                if M[x, c_other]:
                                    critical_codewords.add(c_other)
                                    found = True
                                    break
                            # If not found, something is wrong (should not happen)
                            
                    # Now, can we remove a non-critical codeword?
                    # We want to remove c_prime from C \ {c}
                    # c_prime must NOT be in critical_codewords
                    # Also, w must NOT be removed (we just added it)
                    
                    for c_prime in current_code:
                        if c_prime == c:
                            continue
                        if c_prime in critical_codewords:
                            continue
                        
                        # Can we remove c_prime?
                        # Yes, if it's not critical.
                        # Perform removal
                        code.remove(c_prime)
                        code_mask[c_prime] = False
                        cover_count = new_counts - M[:, c_prime].astype(int)
                        code.append(w)
                        code_mask[w] = True
                        
                        improved = True
                        break
                    if improved:
                        break
            if improved:
                break
                
    return code

def construct(q, n, radius):
    # Verify parameters
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed-like construction for other parameters
        # Just return a simple construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin
    
    # 1. Greedy
    code = greedy_construct()
    
    # 2. Local Search
    # Allocate remaining time to local search
    time_elapsed = time.time() - start_time
    remaining_time = deadline - time_elapsed
    
    if remaining_time > 5:
        code = local_search(code, remaining_time)
        
    # Convert to list of lists
    result = []
    for c in code:
        result.append(int_to_base3(c, n))
        
    return result

# Ensure the function is available
# The evaluator calls construct(q, n, radius)

if __name__ == "__main__":
    # Simple test
    pass
