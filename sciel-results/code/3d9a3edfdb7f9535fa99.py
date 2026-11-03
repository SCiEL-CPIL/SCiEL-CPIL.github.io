"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    For q=3, n=8, r=1, we use a greedy construction followed by local search.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple construction for other parameters
        # For general case, use a direct product construction if possible
        if q == 3 and radius == 1:
            # Use tetracode based construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n >= 4:
                return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
            else:
                # For n < 4, just return all words
                return [list(word) for word in product(range(3), repeat=n)]
        else:
            # General fallback: return all words (trivial covering code)
            return [list(word) for word in product(range(q), repeat=n)]
    
    # Specialized algorithm for q=3, n=8, radius=1
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Precompute all 3^8 = 6561 words as integers 0..6560
    # Each word is represented as a base-3 integer
    total = 3 ** n  # 6561
    
    # Precompute neighbors: for each word, compute all words within Hamming distance 1
    # A word covers itself and its 8*n = 72 neighbors at distance 1
    # Total covered per codeword: 1 + 8*2 = 17 (since for each position, 2 other symbols)
    
    # Convert word index to base-3 digits
    def index_to_digits(idx, length=n, base=q):
        digits = []
        for _ in range(length):
            digits.append(idx % base)
            idx //= base
        return digits
    
    def digits_to_index(digits, base=q):
        idx = 0
        for d in reversed(digits):
            idx = idx * base + d
        return idx
    
    # Precompute the list of all neighbors (including self) for each word
    # neighbor_list[i] = list of indices within Hamming distance 1 of word i
    neighbor_list = []
    for i in range(total):
        digits = index_to_digits(i)
        covered = [i]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_digits = digits.copy()
                    new_digits[pos] = new_val
                    covered.append(digits_to_index(new_digits))
        neighbor_list.append(covered)
    
    # Greedy construction: repeatedly pick the word that covers the most uncovered words
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # For speed, maintain a count of how many codewords cover each word
    cover_count = np.zeros(total, dtype=np.int32)
    
    # Precompute for each word, the list of words it covers (neighbor_list)
    # Greedy: at each step, pick the word with maximum number of uncovered neighbors
    
    while not covered.all() and time.time() < deadline - 10.0:
        # Find the word that covers the most uncovered words
        best_idx = -1
        best_count = -1
        
        # Vectorized computation: for each candidate word, count uncovered neighbors
        # This is expensive, so we do it in batches or use a smarter approach
        
        # For efficiency, compute gain for all words at once using precomputed neighbor lists
        # gain[i] = number of uncovered words in neighbor_list[i]
        
        # We can compute this by: for each word i, sum over its neighbors of (1 - covered[neighbor])
        # Using numpy: create a 2D array? No, that's 6561 x 17 which is fine.
        
        # Actually, let's use a different approach: 
        # For each uncovered word w, it contributes to the gain of all codeword candidates c such that w is in neighbor_list[c]
        # This is the same as: for each uncovered word w, add 1 to gain[c] for all c in "reverse_neighbor_list"[w]
        
        # Let's precompute reverse neighbors: for each word w, which codeword positions cover it?
        # reverse_neighbor_list[w] = list of c such that w in neighbor_list[c]
        # But this is the same as neighbor_list since the relation is symmetric (distance 1 is symmetric)
        
        # So: gain[c] = sum over w in neighbor_list[c] of (1 - covered[w])
        
        # Compute gain vectorized:
        # Create a matrix M where M[c, j] = 1 if word j is in neighbor_list[c]
        # Then gain = M @ (1 - covered)
        
        # But M is sparse. Let's just do it with loops but vectorized per codeword.
        
        # Alternative: since n=8 and q=3, each word has 1+8*2=17 neighbors.
        # We can compute gain for all 6561 words by iterating over uncovered words.
        
        # Let's use a counting approach:
        gain = np.zeros(total, dtype=np.int32)
        # For each uncovered word, increment gain for all its neighbors (which are potential codewords)
        for w in range(total):
            if not covered[w]:
                for c in neighbor_list[w]:
                    gain[c] += 1
        
        # Pick the word with maximum gain
        best_idx = int(np.argmax(gain))
        if gain[best_idx] == 0:
            break
        
        # Add this word as a codeword
        codewords.append(best_idx)
        
        # Update covered and cover_count
        for w in neighbor_list[best_idx]:
            cover_count[w] += 1
            if cover_count[w] == 1:
                covered[w] = True
        
        # Check if we've covered everything
        if covered.all():
            break
    
    # Local search: try to remove or swap codewords to reduce the count
    # While we still have time, try to improve
    
    if time.time() < deadline - 5.0:
        # Try removing codewords that are redundant
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Try each codeword for removal
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                # Check if removing codeword i still covers all words
                # A word is still covered if cover_count[w] > 1 after removal
                # i.e., cover_count[w] >= 2 for all w in neighbor_list[codewords[i]]
                
                cw = codewords[i]
                redundant = True
                for w in neighbor_list[cw]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove this codeword
                    for w in neighbor_list[cw]:
                        cover_count[w] -= 1
                    codewords.pop(i)
                    changed = True
                    break  # Restart the loop since indices changed
        
        # Try swapping: replace a codeword with another that covers more uncovered words
        # This is more complex; for now, let's try a simple improvement
        # Try adding a new codeword and then removing redundant ones
        for _ in range(50):  # Limited number of swap attempts
            if time.time() > deadline - 5.0:
                break
            
            # Find the word that would cover the most words if added
            # (This is the same as the greedy step, but we're looking for improvement)
            gain = np.zeros(total, dtype=np.int32)
            for w in range(total):
                if not covered[w]:
                    for c in neighbor_list[w]:
                        gain[c] += 1
            
            best_new = int(np.argmax(gain))
            if gain[best_new] == 0:
                break
            
            # Try adding best_new and see if we can remove some existing codewords
            # Tentatively add
            for w in neighbor_list[best_new]:
                cover_count[w] += 1
            
            # Now try to remove redundant codewords
            removed_any = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                cw = codewords[i]
                redundant = True
                for w in neighbor_list[cw]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in neighbor_list[cw]:
                        cover_count[w] -= 1
                    codewords.pop(i)
                    removed_any = True
                    break
            
            if removed_any:
                # We added one and removed one, net zero change in count but possibly better coverage
                # Actually, we want to reduce count, so this swap only helps if we removed more than we added
                # Let's undo this and try a different approach
                # Undo the addition
                for w in neighbor_list[best_new]:
                    cover_count[w] -= 1
                # Re-add the removed codeword? This is getting complex.
                # For simplicity, let's just keep the removal-based improvement
                break
            else:
                # No codeword could be removed, so undo the addition
                for w in neighbor_list[best_new]:
                    cover_count[w] -= 1
    
    # Convert codeword indices back to digit lists
    result = [index_to_digits(cw) for cw in codewords]
    
    # Ensure all codewords are distinct
    seen = set()
    unique_result = []
    for cw in result:
        key = tuple(cw)
        if key not in seen:
            seen.add(key)
            unique_result.append(cw)
    
    return unique_result

# EVOLVE-BLOCK-END
