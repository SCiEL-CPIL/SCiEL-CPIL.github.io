import time
import numpy as np
from itertools import combinations

N = 18
MIN_DISTANCE = 6
TOTAL = 1 << N
DEADLINE_MARGIN = 5.0

def _seed_words():
    """
    Generate a strong initial code using the extended Reed-Muller RM(1,4) code.
    RM(1,4) has length 16, size 32, distance 8.
    We extend it to length 18 by adding 2 zeros.
    """
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _get_neighbors(center, radius):
    """Generate all words within Hamming distance <= radius of center."""
    results = [center]
    bits = list(range(N))
    for r in range(1, radius + 1):
        for comb in combinations(bits, r):
            w = center
            for b in comb:
                w ^= (1 << b)
            results.append(w)
    return results

def _build_forbidden_mask(words):
    """Build a boolean mask of all words forbidden by the current code."""
    forbidden = np.zeros(TOTAL, dtype=bool)
    for w in words:
        neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
        forbidden[neighbors] = True
    return forbidden

def _greedy_with_order(order, initial_words, deadline):
    """
    Greedy construction using a specific ordering of words.
    """
    words = list(initial_words)
    forbidden = np.zeros(TOTAL, dtype=bool)
    
    # Initialize forbidden mask with initial words
    for w in words:
        neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
        forbidden[neighbors] = True
    
    for w in order:
        if time.time() > deadline:
            break
        if not forbidden[w]:
            words.append(w)
            neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
            forbidden[neighbors] = True
            
    return words

def _local_search_swap(words, deadline):
    """
    Local search using swap moves: remove one word, try to add two.
    """
    words = list(words)
    n = len(words)
    
    # Build forbidden mask
    forbidden = _build_forbidden_mask(words)
    
    # Count how many times each word is forbidden by
    # We need a "support" count: for each word in the forbidden set,
    # how many codewords are causing it to be forbidden.
    # But actually, for a swap move, we remove word w_i.
    # This "un-forbids" some words that were ONLY forbidden by w_i.
    # We can find these by checking which neighbors of w_i are not forbidden by any other word.
    
    # This is complex. A simpler approach:
    # Just try random swaps.
    # Pick a random word to remove.
    # Pick 2 random candidates from the newly available space.
    # Check if they are valid against the remaining words.
    
    # To make this efficient, we precompute the "support" for each word in the forbidden set.
    # support[x] = number of codewords w such that dist(x, w) < MIN_DISTANCE.
    
    support = np.zeros(TOTAL, dtype=np.int32)
    for w in words:
        neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
        support[neighbors] += 1
    
    # Now, if we remove word w_i, a word x becomes available if support[x] == 1 and x is a neighbor of w_i.
    # Actually, x becomes available if support[x] == 1 (meaning it was only forbidden by w_i).
    # But we also need x to not be in the code itself (though if it's in the code, support[x] >= 1 due to itself, but wait, x is in the code, so it's forbidden by itself. If we remove w_i, and x != w_i, x is still forbidden by itself. So x must not be in the code).
    
    # Let's iterate through the code words.
    # For each word w_i in the code, find all x such that support[x] == 1 and x is a neighbor of w_i.
    # These are the words that become available if we remove w_i.
    # Then, from this set of available words, try to pick 2 that are mutually compatible (dist >= 6) and compatible with the rest of the code (which is guaranteed if they were only forbidden by w_i).
    
    # Wait, if x is available (support[x] == 1, caused by w_i), then dist(x, w_j) >= 6 for all j != i.
    # So we just need to find 2 such x, y such that dist(x, y) >= 6.
    
    # Let's implement this.
    
    improved = False
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        # Pick a random word to remove
        idx = np.random.randint(len(words))
        w_i = words[idx]
        
        # Find available words
        # available = [x for x in _get_neighbors(w_i, MIN_DISTANCE-1) if support[x] == 1 and x not in words_set]
        # But checking x not in words_set is O(1) with a set.
        words_set = set(words)
        
        available = []
        for x in _get_neighbors(w_i, MIN_DISTANCE - 1):
            if support[x] == 1 and x not in words_set:
                available.append(x)
        
        if len(available) < 2:
            continue
            
        # Try to find 2 compatible words from available
        found_pair = False
        for i in range(len(available)):
            if found_pair:
                break
            for j in range(i + 1, len(available)):
                if (available[i] ^ available[j]).bit_count() >= MIN_DISTANCE:
                    # We found a pair!
                    # Perform the swap
                    words.pop(idx)
                    words.append(available[i])
                    words.append(available[j])
                    improved = True
                    found_pair = True
                    break
        
        if improved:
            # Rebuild support for the new code
            support = np.zeros(TOTAL, dtype=np.int32)
            for w in words:
                neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
                support[neighbors] += 1
            # Continue searching
            pass
        else:
            # No improvement found in this iteration, maybe try a different word?
            # The loop structure above already picks a random word each time.
            pass

    return words

def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN
    
    best_words = None
    best_len = 0
    
    # Precompute orders
    orders = []
    
    # 1. Natural
    orders.append(range(TOTAL))
    
    # 2. Reverse
    orders.append(range(TOTAL-1, -1, -1))
    
    # 3. Weight sorted (low to high)
    try:
        all_indices = np.arange(TOTAL, dtype=np.int64)
        weights = np.array([w.bit_count() for w in all_indices])
        sorted_indices = all_indices[np.lexsort((all_indices, weights))]
        orders.append(sorted_indices)
        
        sorted_indices_desc = all_indices[np.lexsort((all_indices, -weights))]
        orders.append(sorted_indices_desc)
    except Exception:
        pass

    # Seed words
    seed_words = _seed_words()
    
    # Strategies
    strategies = [
        ([], orders[0]),
        ([], orders[1]),
    ]
    
    if len(orders) > 2:
        strategies.append(([], orders[2]))
        strategies.append(([], orders[3]))
    
    strategies.append((seed_words, orders[0]))
    strategies.append((seed_words, orders[1]))
    
    # Random order
    rng = np.random.default_rng(42)
    random_order = rng.permutation(TOTAL)
    strategies.append(([], random_order))
    
    # Run greedy strategies
    for init, order in strategies:
        if time.time() > deadline - 10.0:
            break
        try:
            w = _greedy_with_order(order, init, deadline - 10.0)
            if len(w) > best_len:
                best_len = len(w)
                best_words = w
        except Exception:
            pass

    # If we have time left, try local search on the best result
    if best_words is not None and time.time() < deadline - 5.0:
        current_words = _local_search_swap(best_words, deadline - 5.0)
        if len(current_words) > best_len:
            best_len = len(current_words)
            best_words = current_words

    if best_words is None:
        best_words = _seed_words()

    # Final validation and deduplication
    seen = set()
    out = []
    for w in best_words:
        if w not in seen and 0 <= w < TOTAL:
            seen.add(w)
            out.append(w)
    
    # Verify pairwise distance for safety
    if len(out) > 0:
        valid = True
        for i in range(len(out)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(out)):
                if (out[i] ^ out[j]).bit_count() < MIN_DISTANCE:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            out = _seed_words()

    return out

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Time: {time.time()-start:.2f}s, Size: {len(res)}")
