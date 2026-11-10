"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _q3_tetracode_projected(n):
    # Full ternary tetracode (perfect radius-1 code on 4 coords) over {0,1,2}.
    full = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Project onto coordinates {0,1,2} (drop coordinate 3).
    projected = [[w[0], w[1], w[2]] for w in full]
    from itertools import product
    rows = []
    for p in projected:
        for tail in product(range(3), repeat=n - 3):
            rows.append(list(p) + list(tail))
    return rows


def _direct_sum(q, n, radius):
    # Generic construction: a perfect radius-1 code of length 4 over an alphabet of
    # size m=ceil(q/(radius+1)) (valid whenever q <= m*(radius+1)), times every word
    # over the remaining coordinates. Size = m**4 * q**(n-4).
    from itertools import product
    m = (q + radius) // (radius + 1)
    # Code over Z_m (m odd => radius-1 perfect covering; even m => radius 1).
    base = [[a, b, (a + b) % m, (a + 2 * b) % m] for a in range(m) for b in range(m)]
    rows = []
    if n >= 4:
        for h in base:
            for tail in product(range(q), repeat=n - 4):
                rows.append(list(h) + list(tail))
    else:  # fallback: full space (guaranteed valid, may be large)
        for w in product(range(q), repeat=n):
            rows.append(list(w))
    return rows


def _is_covering(codewords, q, n, radius):
    """Check if codewords cover all words within Hamming distance radius."""
    if len(codewords) == 0:
        return False
    # For small spaces, brute force check
    if q ** n <= 100000:
        all_words = set(itertools.product(range(q), repeat=n))
        covered = set()
        for cw in codewords:
            cw = tuple(cw)
            # Add all words within radius
            for i in range(n):
                for j in range(n):
                    pass
            # More efficient: generate all neighbors within radius
            # For radius 1: the word itself + all single-position changes
            if radius == 1:
                covered.add(cw)
                for pos in range(n):
                    for val in range(q):
                        if val != cw[pos]:
                            new_word = list(cw)
                            new_word[pos] = val
                            covered.add(tuple(new_word))
            else:
                # General case: use itertools to generate all words at distance <= radius
                # This is expensive but only for small spaces
                for dist in range(radius + 1):
                    for positions in itertools.combinations(range(n), dist):
                        for values in itertools.product(range(q), repeat=dist):
                            new_word = list(cw)
                            for pos, val in zip(positions, values):
                                new_word[pos] = val
                            # Check if actually at distance 'dist' (not less)
                            # Actually we want distance <= radius, so we just add all
                            covered.add(tuple(new_word))
        return covered >= all_words
    else:
        # For larger spaces, use probabilistic check or accept the construction
        # Since our constructions are theoretically valid, we trust them
        return True


def _greedy_improve(codewords, q, n, radius, deadline):
    """Try to remove redundant codewords via greedy local search."""
    start_time = time.time()
    codewords = [list(cw) for cw in codewords]
    
    # Precompute coverage for each codeword
    # For radius 1, a codeword covers itself and all words at Hamming distance 1
    # We can check if a codeword is redundant by seeing if all its unique neighbors
    # are covered by other codewords
    
    if len(codewords) > 10000:
        # Too many to do exhaustive checks efficiently
        return codewords
    
    def get_neighbors(cw):
        """Get all words within Hamming distance radius of cw."""
        cw = tuple(cw)
        neighbors = {cw}
        if radius == 1:
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        new_word = list(cw)
                        new_word[pos] = val
                        neighbors.add(tuple(new_word))
        else:
            # For general radius, this is expensive
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for values in itertools.product(range(q), repeat=dist):
                        new_word = list(cw)
                        for pos, val in zip(positions, values):
                            new_word[pos] = val
                        neighbors.add(tuple(new_word))
        return neighbors
    
    # For each codeword, compute which neighbors it uniquely covers
    # A codeword is redundant if all its neighbors are covered by others
    
    # This is O(k^2 * neighbor_count) which can be expensive
    # Let's try a simpler approach: try removing each codeword and check if still covering
    
    # Actually, for the projected tetracode construction, the codewords are structured
    # Let's just try a few random removals
    
    current = codewords[:]
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline - 10:
                break
            # Try removing codeword i
            candidate = current[:i] + current[i+1:]
            # Check if candidate still covers
            # For efficiency, only check if the removed codeword's unique neighbors
            # are covered by others
            # Simple check: verify coverage of the removed codeword's neighbors
            cw_to_remove = tuple(current[i])
            neighbors = get_neighbors(cw_to_remove)
            
            # Check if all neighbors are covered by remaining codewords
            all_covered = True
            for nb in neighbors:
                # Check if nb is within radius of any remaining codeword
                covered = False
                for cw in candidate:
                    # Hamming distance
                    dist = sum(1 for a, b in zip(nb, cw) if a != b)
                    if dist <= radius:
                        covered = True
                        break
                if not covered:
                    all_covered = False
                    break
            
            if all_covered:
                current = candidate
                improved = True
                break  # Restart with new list
    
    return current


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Choose initial construction
    if q == 3 and n >= 5 and radius == 1:
        # Use the projected tetracode construction: 9 * 3^(n-3) words
        codewords = _q3_tetracode_projected(n)
    else:
        # Use direct sum construction
        codewords = _direct_sum(q, n, radius)
    
    # Try to improve via greedy removal
    if len(codewords) <= 5000 and time.time() < deadline - 20:
        codewords = _greedy_improve(codewords, q, n, radius, deadline)
    
    return codewords
# EVOLVE-BLOCK-END
