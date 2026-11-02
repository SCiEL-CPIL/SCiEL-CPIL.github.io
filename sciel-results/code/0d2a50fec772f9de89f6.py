"""Geometric Hyperplane Projection approach for K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    Uses geometric hyperplane projection to guide greedy selection with multiple
    projection directions for more robust tile-deficiency scoring.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    if n == 2:
        # For n=2, q=3, radius=1: simple greedy works
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

    # Multiple geometric hyperplane projections:
    # Use several pairs of linear functionals to create different 2D tile views.
    # This gives a more robust "deficiency" signal across multiple geometric perspectives.
    
    # Generate diverse weight vectors
    random.seed(42)
    num_projections = 4
    
    projections = []
    for proj_idx in range(num_projections):
        # Create two weight vectors that are somewhat orthogonal
        w1 = [random.randint(0, q-1) for _ in range(n)]
        w2 = [random.randint(0, q-1) for _ in range(n)]
        # Ensure they're not identical
        if w1 == w2:
            w2 = [v ^ 1 for v in w2]
        projections.append((w1, w2))
    
    # Also include the "standard" projection for stability
    weights1_std = [1] * n
    weights2_std = [(i + 1) % q if q > 1 else 0 for i in range(n)]
    projections.append((weights1_std, weights2_std))
    
    num_projections = len(projections)
    
    # Precompute tile index for each word under each projection
    tile_of_word = [[0] * total_words for _ in range(num_projections)]
    tile_size = q * q
    
    for proj_idx, (w1, w2) in enumerate(projections):
        for idx, w in enumerate(all_words):
            p1 = sum(w[i] * w1[i] for i in range(n)) % q
            p2 = sum(w[i] * w2[i] for i in range(n)) % q
            tile_of_word[proj_idx][idx] = p1 * q + p2

    # Count words per tile for each projection (should be uniform, but compute anyway)
    words_per_tile = [[0] * tile_size for _ in range(num_projections)]
    for proj_idx in range(num_projections):
        for t in tile_of_word[proj_idx]:
            words_per_tile[proj_idx][t] += 1

    # Greedy with multi-projection geometric guidance:
    # At each step, score each candidate codeword by:
    #   primary: number of newly covered words
    #   secondary: aggregate tile deficiency score across all projections
    
    covered = [False] * total_words
    remaining_uncovered = total_words
    tile_covered_count = [[0] * tile_size for _ in range(num_projections)]
    codewords = []

    for step in range(300):
        if remaining_uncovered == 0:
            break

        best_codeword = -1
        best_score = (-1, -1.0)  # (newly_covered, tile_deficiency_score)

        for w_idx in range(total_words):
            # Count newly covered words and per-tile new coverage
            newly = 0
            # Use a single tile_newly array for the standard projection (index -1)
            # but accumulate deficiency across all projections
            tile_newly_std = [0] * tile_size
            for nb in neighbors[w_idx]:
                if not covered[nb]:
                    newly += 1
                    tile_newly_std[tile_of_word[-1][nb]] += 1

            if newly == 0:
                continue

            # Aggregate tile deficiency score across all projections
            tile_score = 0.0
            for proj_idx in range(num_projections):
                # Build tile_newly for this projection
                tile_newly_proj = [0] * tile_size
                for nb in neighbors[w_idx]:
                    if not covered[nb]:
                        tile_newly_proj[tile_of_word[proj_idx][nb]] += 1
                
                for t in range(tile_size):
                    if tile_newly_proj[t] > 0:
                        density = tile_covered_count[proj_idx][t] / words_per_tile[proj_idx][t] if words_per_tile[proj_idx][t] > 0 else 0
                        tile_score += tile_newly_proj[t] * (1.0 - density)

            # Normalize by number of projections
            tile_score /= num_projections

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
                for proj_idx in range(num_projections):
                    tile_covered_count[proj_idx][tile_of_word[proj_idx][nb]] += 1

    return [list(w) for w in codewords]
# EVOLVE-BLOCK-END
