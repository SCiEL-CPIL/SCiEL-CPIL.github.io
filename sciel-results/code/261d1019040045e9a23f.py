import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 55  # Safety margin before 60s limit

    # Precompute all masks for distances 0 to MIN_DIST-1
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)

    # Total number of masks
    total_masks = sum(len(m) for m in masks_by_dist)

    # Precompute all masks as a flat list for vectorized operations
    all_masks = []
    for d in range(MIN_DIST):
        all_masks.extend(masks_by_dist[d])
    all_masks = np.array(all_masks, dtype=np.int64)

    def get_neighbors_vectorized(words):
        """Return all words within distance < MIN_DIST of any word in words."""
        # words is a list of ints
        # Use broadcasting: for each word, XOR with all masks
        # This creates a (len(words), total_masks) array
        words_arr = np.array(words, dtype=np.int64)
        # Reshape for broadcasting
        result = words_arr[:, np.newaxis] ^ all_masks[np.newaxis, :]
        # Flatten
        return result.flatten()

    # 1. Find a linear code [17, 8, 6]
    k = 8
    linear_code = None

    # Search for linear code with multiple attempts
    max_linear_search_time = 6.0
    np.random.seed(42)
    while time.time() - start_time < max_linear_search_time:
        G = np.random.randint(0, 2, size=(k, N))

        # Convert G rows to integers
        G_ints = []
        for i in range(k):
            val = 0
            for j in range(N):
                if G[i, j] == 1:
                    val |= (1 << j)
            G_ints.append(val)

        # Generate all 2^k codewords
        codewords = [0]
        for i in range(k):
            new_codewords = [c ^ G_ints[i] for c in codewords]
            codewords.extend(new_codewords)

        # Check min weight (for linear codes, min weight = min distance)
        min_d = N + 1
        for c in codewords:
            if c == 0:
                continue
            w = bin(c).count('1')
            if w < min_d:
                min_d = w
                if min_d < MIN_DIST:
                    break
        if min_d >= MIN_DIST:
            linear_code = codewords
            break

    if linear_code is None:
        # Fallback: Simple greedy from 0
        blocked = bytearray(1 << N)
        code = []
        def add_word(w):
            if blocked[w]:
                return
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1

        add_word(0)
        for w in range(1, 1 << N):
            if time.time() - start_time > time_limit:
                break
            if not blocked[w]:
                add_word(w)
        return code

    # 2. Expand the linear code using greedy + local search
    blocked = bytearray(1 << N)
    code = list(linear_code)  # 256 words

    # Block all neighbors of the linear code
    for w in code:
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1

    # Greedy scan for additional words - use a faster approach
    # Track unblocked words
    unblocked = []
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1

    # 3. Local search: try to improve by removing one word and re-greedy'ing
    best_code = list(code)
    best_size = len(code)

    # Build a set for fast membership testing
    code_set = set(code)

    # Try multiple passes of local search
    local_search_deadline = time_limit - 5  # Leave 5s safety margin

    # Track which words are "free" (not blocked)
    # Use a list of indices for faster iteration
    free_words = [w for w in range(1 << N) if not blocked[w]]

    # Multiple local search passes
    for pass_num in range(20):
        if time.time() - start_time > local_search_deadline:
            break

        improved = False
        # Shuffle for diversity
        indices = list(range(len(code)))
        random.shuffle(indices)

        for idx in indices:
            if time.time() - start_time > local_search_deadline:
                break

            # Remove word at idx
            removed_word = code[idx]
            code.pop(idx)
            code_set.remove(removed_word)

            # Unblock its neighbors
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[removed_word ^ mask] = 0

            # Re-greedy: try to add words that were blocked
            # Only scan words that might be free now
            new_additions = []
            for w in range(1 << N):
                if time.time() - start_time > local_search_deadline:
                    break
                if not blocked[w] and w not in code_set:
                    new_additions.append(w)
                    code_set.add(w)
                    for d in range(MIN_DIST):
                        for mask in masks_by_dist[d]:
                            blocked[w ^ mask] = 1

            # Check if we improved
            if len(code) + len(new_additions) > best_size:
                best_size = len(code) + len(new_additions)
                best_code = list(code) + new_additions
                improved = True
                # Update current state
                code = best_code
                code_set = set(code)
                # Rebuild blocked array
                blocked = bytearray(1 << N)
                for w in code:
                    for d in range(MIN_DIST):
                        for mask in masks_by_dist[d]:
                            blocked[w ^ mask] = 1
                break  # Restart the loop with improved code

            # Restore if no improvement
            code.insert(idx, removed_word)
            code_set.add(removed_word)
            # Re-block its neighbors
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[removed_word ^ mask] = 1

        if not improved:
            # Try a different strategy: swap two words
            if time.time() - start_time > local_search_deadline:
                break
            
            # Try swapping: remove word i, add word j
            # Find candidate swaps
            swap_improved = False
            for i in range(min(len(code), 100)):
                if time.time() - start_time > local_search_deadline:
                    break
                
                removed_word = code[i]
                code.pop(i)
                code_set.remove(removed_word)
                
                # Unblock its neighbors
                for d in range(MIN_DIST):
                    for mask in masks_by_dist[d]:
                        blocked[removed_word ^ mask] = 0
                
                # Try to find a word we can add
                best_add = None
                for w in range(1 << N):
                    if time.time() - start_time > local_search_deadline:
                        break
                    if not blocked[w] and w not in code_set:
                        best_add = w
                        break
                
                if best_add is not None:
                    code.append(best_add)
                    code_set.add(best_add)
                    for d in range(MIN_DIST):
                        for mask in masks_by_dist[d]:
                            blocked[best_add ^ mask] = 1
                    swap_improved = True
                else:
                    # Restore
                    code.insert(i, removed_word)
                    code_set.add(removed_word)
                    for d in range(MIN_DIST):
                        for mask in masks_by_dist[d]:
                            blocked[removed_word ^ mask] = 1

                if swap_improved:
                    break

            if not swap_improved:
                break

    return best_code
