"""Geometric Hyperplane Projection approach for K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    Uses geometric hyperplane projection to guide greedy selection.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    if n == 2:
        # For n=2, q=3, radius=1: greedy is fine
        pass

    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)

    def word_to_int(word):
        val = 0
        for v in word:
            val = val * q + v
        return val

    # Precompute neighbors for each word index
    neighbors = []
    for w in all_words:
        w_int = word_to_int(w)
        nbs = [w_int]
        w_list = list(w)
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val == old_val:
                    continue
                w_list[pos] = new_val
                nbs.append(word_to_int(tuple(w_list)))
            w_list[pos] = old_val
        neighbors.append(nbs)

    # Geometric hyperplane projection:
    # Project each word to a 2D lattice tile using two linear functionals over GF(q).
    # We use different weight vectors to create orthogonal-ish projections.
    # For n=7, q=3, we can use:
    #   proj1 = sum(w_i) mod q
    #   proj2 = sum(w_i * (i+1)) mod q
    # This gives a 2D tile of size q x q.

    if n >= 2:
        weights1 = [1] * n
        weights2 = [(i + 1) % q if q > 1 else 0 for i in range(n)]
    else:
        weights1 = [1]
        weights2 = [1]

    # Precompute tile index for each word
    tile_of_word = [0] * total_words
    for idx, w in enumerate(all_words):
        p1 = sum(w[i] * weights1[i] for i in range(n)) % q
        p2 = sum(w[i] * weights2[i] for i in range(n)) % q
        tile_of_word[idx] = p1 * q + p2

    # Count words per tile
    tile_size = q * q
    words_per_tile = [0] * tile_size
    for t in tile_of_word:
        words_per_tile[t] += 1

    # Greedy with geometric guidance:
    # At each step, score each candidate codeword by:
    #   primary: number of newly covered words
    #   secondary: how well it fills deficient tiles (tiles with low coverage density)
    
    covered = [False] * total_words
    remaining_uncovered = total_words
    tile_covered_count = [0] * tile_size
    codewords = []

    for step in range(300):
        if remaining_uncovered == 0:
            break

        best_codeword = -1
        best_score = (-1, -1)  # (newly_covered, tile_deficiency_score)

        for w_idx in range(total_words):
            # Count newly covered words
            newly = 0
            tile_newly = [0] * tile_size  # how many new words in each tile
            for nb in neighbors[w_idx]:
                if not covered[nb]:
                    newly += 1
                    t = tile_of_word[nb]
                    tile_newly[t] += 1

            if newly == 0:
                continue

            # Tile deficiency score: sum over tiles of (newly_in_tile * (1 - current_density))
            # where current_density = tile_covered_count[t] / words_per_tile[t]
            # Higher score means we're filling more under-covered tiles
            tile_score = 0.0
            for t in range(tile_size):
                if tile_newly[t] > 0:
                    density = tile_covered_count[t] / words_per_tile[t] if words_per_tile[t] > 0 else 0
                    tile_score += tile_newly[t] * (1.0 - density)

            score = (newly, tile_score)
            if score > best_score:
                best_score = score
                best_codeword = w_idx

        if best_codeword == -1:
            break

        # Add the best codeword
        codewords.append(all_words[best_codeword])
        for nb in neighbors[best_codeword]:
            if not covered[nb]:
                covered[nb] = True
                remaining_uncovered -= 1
                t = tile_of_word[nb]
                tile_covered_count[t] += 1

    return [list(w) for w in codewords]
# EVOLVE-BLOCK-END
