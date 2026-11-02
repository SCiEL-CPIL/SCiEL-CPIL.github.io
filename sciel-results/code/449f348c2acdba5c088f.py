"""Geometric Hyperplane Projection approach for K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
from collections import defaultdict
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    Uses Geometric Hyperplane Projection: map to 2D lattice via two linear functionals,
    then greedily select codewords that maximize coverage of sparsely populated tiles.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    
    # For the specific task: q=3, n=7, radius=1
    # We use a greedy algorithm guided by geometric projections.
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    
    # Create a mapping from word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Represent words as integers for efficient hashing
    def word_to_int(word):
        val = 0
        for i, v in enumerate(word):
            val = val * q + v
        return val
    
    def int_to_word(val, n):
        word = []
        for i in range(n):
            word.append(val % q)
            val //= q
        return word
    
    # Precompute neighbors for each word index
    # neighbors[i] = list of word indices within Hamming distance <= radius
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
    
    # Define two orthogonal linear projections over GF(q)
    # For q=3, n=7: use weights that are linearly independent
    # proj1: sum of coordinates mod q
    # proj2: weighted sum with different weights mod q
    weights1 = [1] * n
    weights2 = [1, 2, 1, 2, 1, 2, 1][:n]  # For n=7, this gives [1,2,1,2,1,2,1]
    
    # Compute 2D projection for each word
    # proj[i] = (a, b) where a = sum(w_j * weights1_j) mod q, b = sum(w_j * weights2_j) mod q
    proj = []
    for w in all_words:
        a = sum(w[j] * weights1[j] for j in range(n)) % q
        b = sum(w[j] * weights2[j] for j in range(n)) % q
        proj.append((a, b))
    
    # Group words by their 2D projection tile
    tile_to_words = defaultdict(list)
    for i, (a, b) in enumerate(proj):
        tile_to_words[(a, b)].append(i)
    
    # Initialize coverage tracking
    covered = [False] * total_words
    remaining_uncovered = total_words
    
    # Track tile coverage: how many words in each tile are covered
    tile_covered_count = defaultdict(int)
    for (a, b), words in tile_to_words.items():
        tile_covered_count[(a, b)] = 0
    
    codewords = []
    
    # Greedy algorithm with tile-based guidance
    # At each step, prioritize codewords that cover words in sparsely covered tiles
    for step in range(400):  # cap at 400 steps
        if remaining_uncovered == 0:
            break
        
        best_codeword = -1
        best_score = -1
        best_new_coverage = 0
        
        # Score function: maximize new coverage, with tie-breaking by tile sparsity
        # For each candidate, compute:
        # 1. Number of newly covered words
        # 2. Weighted sum of inverse tile density for newly covered words
        
        # To speed up, we can precompute which tiles each candidate covers
        # But for now, let's compute on the fly
        
        for w_idx in range(total_words):
            # Count newly covered words and their tile sparsity
            new_count = 0
            tile_sparsity_bonus = 0.0
            
            for nb in neighbors[w_idx]:
                if not covered[nb]:
                    new_count += 1
                    # Get tile of this word
                    tile = proj[nb]
                    total_in_tile = len(tile_to_words[tile])
                    covered_in_tile = tile_covered_count[tile]
                    # Sparsity bonus: higher when tile is less covered
                    if total_in_tile > 0:
                        uncovered_in_tile = total_in_tile - covered_in_tile
                        tile_sparsity_bonus += 1.0 / max(uncovered_in_tile, 1)
            
            # Combined score: new coverage + small bonus for tile sparsity
            score = new_count + 0.1 * tile_sparsity_bonus
            
            if score > best_score or (score == best_score and new_count > best_new_coverage):
                best_score = score
                best_codeword = w_idx
                best_new_coverage = new_count
        
        if best_new_coverage <= 0:
            break
        
        # Add the best codeword
        codewords.append(all_words[best_codeword])
        for nb in neighbors[best_codeword]:
            if not covered[nb]:
                covered[nb] = True
                remaining_uncovered -= 1
                # Update tile coverage
                tile = proj[nb]
                tile_covered_count[tile] += 1
    
    # Convert back to list of lists
    return [list(w) for w in codewords]
# EVOLVE-BLOCK-END
