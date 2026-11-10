# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import itertools
import random
import time


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet q with covering radius
    radius.  For the football-pool task this is (q, n, radius) = (3, 6, 1).

    Strategy:
      * Build an initial seed set using a product construction from a known
        perfect code on the first 4 coordinates (the ternary tetracode).
      * Use simulated annealing with smart move operations to minimize the
        number of codewords while maintaining coverage.
    """
    # --- helpers ---------------------------------------------------------------
    def encode(word, base=q):
        """Encode a tuple/list of symbols into a single integer."""
        v = 0
        for sym in word:
            v = v * base + sym
        return v

    def decode(v, base=q, length=n):
        """Decode an integer into a tuple of symbols (length n)."""
        out = []
        for _ in range(length):
            v, r = divmod(v, base)
            out.append(r)
        return tuple(reversed(out))

    # Precompute all words and their neighbours (within Hamming distance <= radius)
    total = q ** n
    neighbours = [None] * total
    for w in range(total):
        word = decode(w)
        neigh = set([w])
        if radius >= 1:
            for i in range(n):
                orig = word[i]
                for sym in range(q):
                    if sym != orig:
                        new_word = list(word)
                        new_word[i] = sym
                        neigh.add(encode(new_word))
        if radius >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    orig_i = word[i]
                    orig_j = word[j]
                    for sym_i in range(q):
                        for sym_j in range(q):
                            if sym_i == orig_i and sym_j == orig_j:
                                continue
                            new_word = list(word)
                            new_word[i] = sym_i
                            new_word[j] = sym_j
                            neigh.add(encode(new_word))
        neighbours[w] = list(neigh)

    # Coverage count array
    covered_by = [0] * total  # count of codewords covering each word
    uncovered_count = total  # number of words with coverage 0

    def add_codeword(cw):
        """Add a codeword (list of ints) and update coverage."""
        nonlocal uncovered_count
        cw_int = encode(cw)
        for nb in neighbours[cw_int]:
            if covered_by[nb] == 0:
                uncovered_count -= 1
            covered_by[nb] += 1

    def remove_codeword(cw):
        """Remove a codeword (list of ints) and update coverage."""
        nonlocal uncovered_count
        cw_int = encode(cw)
        for nb in neighbours[cw_int]:
            covered_by[nb] -= 1
            if covered_by[nb] == 0:
                uncovered_count += 1

    def is_covered():
        """Check if all words are covered."""
        return uncovered_count == 0

    # --- initial seed: tetracode product --------------------------------------
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]

    tails = list(itertools.product(range(q), repeat=n - 4)) if n > 4 else [[]]

    codewords = [h + list(tail) for h in tetracode for tail in tails]

    # Initialize coverage
    for cw in codewords:
        add_codeword(cw)

    # --- greedy removal of redundant codewords --------------------------------
    for idx in range(len(codewords)):
        cw = codewords[idx]
        remove_codeword(cw)
        if is_covered():
            codewords.pop(idx)
            idx -= 1
        else:
            add_codeword(cw)

    # --- Simulated Annealing ---------------------------------------------------
    random.seed(42)
    
    best_code = codewords[:]
    best_size = len(best_code)
    
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for safety
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    iteration = 0
    
    while T > T_end and (time.time() - start_time) < time_limit:
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random codeword
            if len(codewords) > 1:
                idx = random.randint(0, len(codewords) - 1)
                cw_removed = codewords[idx]
                remove_codeword(cw_removed)
                codewords.pop(idx)
                
                # Check if still covered
                if is_covered():
                    # Accept: size decreased
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_code = codewords[:]
                else:
                    # Try to add a codeword that covers uncovered words
                    uncovered = [w for w in range(total) if covered_by[w] == 0]
                    if uncovered:
                        # Find best codeword to add
                        best_gain = 0
                        best_cw = None
                        # Sample from uncovered words' neighbourhoods
                        sample_size = min(100, len(uncovered))
                        for _ in range(sample_size):
                            unc = random.choice(uncovered)
                            # Try adding a codeword near this uncovered word
                            cand_word = list(decode(unc))
                            # Perturb slightly
                            pos = random.randint(0, n - 1)
                            if random.random() < 0.5:
                                cand_word[pos] = random.randint(0, q - 1)
                            cand_int = encode(cand_word)
                            gain = sum(1 for nb in neighbours[cand_int] if covered_by[nb] == 0)
                            if gain > best_gain:
                                best_gain = gain
                                best_cw = cand_word
                        
                        if best_cw is not None and best_gain > 0:
                            add_codeword(best_cw)
                            codewords.append(best_cw)
                            if len(codewords) < best_size:
                                best_size = len(codewords)
                                best_code = codewords[:]
                        else:
                            # Revert
                            add_codeword(cw_removed)
                            codewords.insert(idx, cw_removed)
                    else:
                        # Shouldn't happen, but revert
                        add_codeword(cw_removed)
                        codewords.insert(idx, cw_removed)
        
        elif move_type < 0.6:
            # Replace a codeword with a random one
            if len(codewords) > 1:
                idx = random.randint(0, len(codewords) - 1)
                cw_old = codewords[idx]
                remove_codeword(cw_old)
                codewords.pop(idx)
                
                # Generate new codeword
                cand = [random.randint(0, q - 1) for _ in range(n)]
                add_codeword(cand)
                codewords.append(cand)
                
                if is_covered():
                    # Accept
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_code = codewords[:]
                else:
                    # Revert
                    remove_codeword(cand)
                    codewords.pop()
                    add_codeword(cw_old)
                    codewords.insert(idx, cw_old)
        
        else:
            # Swap two codewords' positions (no size change, but may help with coverage)
            if len(codewords) > 2:
                idx1 = random.randint(0, len(codewords) - 1)
                idx2 = random.randint(0, len(codewords) - 1)
                if idx1 != idx2:
                    cw1 = codewords[idx1]
                    cw2 = codewords[idx2]
                    # Remove both
                    remove_codeword(cw1)
                    remove_codeword(cw2)
                    codewords.pop(idx1)
                    if idx2 > idx1:
                        idx2 -= 1
                    codewords.pop(idx2)
                    
                    # Add back in swapped positions
                    add_codeword(cw2)
                    add_codeword(cw1)
                    codewords.insert(idx2, cw2)
                    codewords.insert(idx1, cw1)
                    
                    # If covered, keep; else revert
                    if not is_covered():
                        # Revert swap
                        remove_codeword(cw2)
                        remove_codeword(cw1)
                        codewords.pop(idx1)
                        if idx2 > idx1:
                            idx2 -= 1
                        codewords.pop(idx2)
                        add_codeword(cw1)
                        add_codeword(cw2)
                        codewords.insert(idx2, cw2)
                        codewords.insert(idx1, cw1)
        
        # Cool down
        T *= cooling_rate
        iteration += 1
    
    # Final greedy removal
    for idx in range(len(codewords)):
        cw = codewords[idx]
        remove_codeword(cw)
        if is_covered():
            codewords.pop(idx)
            idx -= 1
        else:
            add_codeword(cw)
    
    # Return best found
    return best_code
# EVOLVE-BLOCK-END
