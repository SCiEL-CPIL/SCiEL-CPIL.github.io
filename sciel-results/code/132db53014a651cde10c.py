"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import heapq

def construct(q, n, radius):
    # For the specific task: q=3, n=8, radius=1
    # Total words: 3^8 = 6561
    # Each codeword covers 1 + 8*2 = 17 words (itself + 8 positions * 2 alternatives)
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    total = q ** n
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for 120s limit
    
    # Map each word (tuple of 8 ints) to an index 0..6560
    word_to_idx = {}
    idx_to_word = []
    for word in itertools.product(range(q), repeat=n):
        idx = idx_to_word.__len__()
        word_to_idx[word] = idx
        idx_to_word.append(word)
    
    # Precompute for each codeword index, the list of word indices it covers
    # And for each word index, the list of codeword indices that cover it
    covers_word = [[] for _ in range(total)]  # covers_word[w] = list of codeword indices covering word w
    sphere = [[] for _ in range(total)]  # sphere[c] = list of word indices covered by codeword c
    
    for c_idx, c_word in enumerate(idx_to_word):
        c_list = list(c_word)
        # The word itself
        covers_word[c_idx].append(c_idx)
        sphere[c_idx].append(c_idx)
        # All words at distance 1
        for pos in range(n):
            orig_val = c_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                c_list[pos] = new_val
                w_idx = word_to_idx[tuple(c_list)]
                covers_word[w_idx].append(c_idx)
                sphere[c_idx].append(w_idx)
            c_list[pos] = orig_val
    
    # Greedy set cover with heap and lazy deletion
    coverage = [len(covers_word[i]) for i in range(total)]  # Initially each word is covered by all codewords that cover it
    # Wait, that's not right. coverage[c] should be the number of currently uncovered words that codeword c covers.
    # Initially, all words are uncovered, so coverage[c] = number of words in sphere[c] = 17 for all c.
    
    coverage = [17] * total  # Each codeword covers 17 words initially
    
    # Heap: (-coverage[c], c) for max-heap behavior (we want highest coverage first)
    heap = [(-coverage[i], i) for i in range(total)]
    heapq.heapify(heap)
    
    is_covered = [False] * total  # is_covered[w] = True if word w is covered
    selected = []
    
    # Count of uncovered words
    uncovered_count = total
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Pop the codeword with highest current coverage
        neg_cov, c_idx = heapq.heappop(heap)
        if -neg_cov != coverage[c_idx]:
            continue  # Stale entry
        if coverage[c_idx] == 0:
            break  # No codeword can cover any more words
        
        # Select this codeword
        selected.append(c_idx)
        
        # Update coverage for all words in this codeword's sphere
        for w_idx in sphere[c_idx]:
            if not is_covered[w_idx]:
                is_covered[w_idx] = True
                uncovered_count -= 1
                # Decrement coverage for all codewords that cover w_idx
                for j in covers_word[w_idx]:
                    coverage[j] -= 1
                    heapq.heappush(heap, (-coverage[j], j))
    
    # Convert selected indices to word lists
    result = [list(idx_to_word[i]) for i in selected]
    
    # Verify and ensure completeness if we timed out
    if uncovered_count > 0:
        # Add any remaining needed codewords
        # Simple fallback: add codewords until all covered
        for c_idx in range(total):
            if time.time() > deadline:
                break
            if c_idx not in selected:
                for w_idx in sphere[c_idx]:
                    if not is_covered[w_idx]:
                        is_covered[w_idx] = True
                        uncovered_count -= 1
                        if uncovered_count == 0:
                            break
                if uncovered_count == 0:
                    break
                selected.append(c_idx)
        result = [list(idx_to_word[i]) for i in selected]
    
    return result
# EVOLVE-BLOCK-END
