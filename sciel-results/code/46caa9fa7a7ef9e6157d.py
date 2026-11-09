"""Football pool problem: ternary covering code K_3(7,1)
Uses bitmask-based greedy initialization + local search with cover-counts."""

import numpy as np
import time
import itertools

def _encode_word(w):
    """Encode a ternary word as an integer (base-3)."""
    val = 0
    for i, d in enumerate(w):
        val = val * 3 + d
    return val

def _decode_word(val, n):
    """Decode an integer (base-3) to a ternary word of length n."""
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = val % 3
        val //= 3
    return w

def _neighbors_within_radius(w, q, n, radius):
    """Generate all words within Hamming distance <= radius of w."""
    if radius == 0:
        yield w
        return
    if radius == 1:
        yield w
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    yield w[:pos] + [val] + w[pos+1:]
        return
    # General case for small radius
    def gen(w, remaining, pos):
        if remaining == 0:
            yield w
            return
        if pos == n:
            return
        # Don't change this position
        yield from gen(w, remaining, pos + 1)
        # Change this position
        for val in range(q):
            if val != w[pos]:
                yield from gen(w[:pos] + [val] + w[pos+1:], remaining - 1, pos + 1)
    yield from gen(w, radius, 0)

def _build_cover_array(codewords, q, n, radius):
    """Build a numpy array where cover[i] = number of codewords within radius of word i."""
    total = q ** n
    cover = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        cw_val = _encode_word(cw)
        for nb in _neighbors_within_radius(cw, q, n, radius):
            nb_val = _encode_word(nb)
            cover[nb_val] += 1
    return cover

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    total = q ** n
    all_words = [_decode_word(i, n) for i in range(total)]
    
    # Phase 1: Greedy initialization with bitmask
    # Sort words by a heuristic (e.g., prefer words that cover many uncovered points)
    codewords = []
    covered = np.zeros(total, dtype=np.bool_)
    
    # Precompute neighbor lists for efficiency
    neighbor_lists = {}
    for i in range(total):
        w = all_words[i]
        nbrs = []
        for nb in _neighbors_within_radius(w, q, n, radius):
            nbrs.append(_encode_word(nb))
        neighbor_lists[i] = nbrs
    
    # Greedy: repeatedly pick the word that covers the most uncovered points
    uncovered_count = np.ones(total, dtype=np.int32)
    
    while uncovered_count.sum() > 0:
        if time.time() > deadline:
            break
        
        # Find the word that covers the most uncovered points
        best_idx = -1
        best_count = -1
        for i in range(total):
            count = 0
            for nb in neighbor_lists[i]:
                if not covered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx < 0 or best_count <= 0:
            break
        
        # Add this codeword
        codewords.append(all_words[best_idx])
        for nb in neighbor_lists[best_idx]:
            if not covered[nb]:
                covered[nb] = True
                uncovered_count[nb] = 0
    
    if not codewords:
        # Fallback to tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Phase 2: Local search - try to remove codewords
    # Build cover counts
    cover = _build_cover_array(codewords, q, n, radius)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            # Check if removing codeword i still covers everything
            # A word is still covered if cover[word] > 1 after removing codeword i
            # (since cover[word] includes codeword i's contribution)
            cw = codewords[i]
            cw_val = _encode_word(cw)
            can_remove = True
            for nb in neighbor_lists[cw_val]:
                if cover[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove codeword i
                for nb in neighbor_lists[cw_val]:
                    cover[nb] -= 1
                codewords.pop(i)
                changed = True
                break  # Restart the loop after modification
    
    # Phase 3: Try swapping - replace one codeword with another to enable more removals
    # This is a simpler form: just try to find any codeword that can be removed
    # after some strategic moves
    
    # Final check: ensure validity
    final_cover = _build_cover_array(codewords, q, n, radius)
    if (final_cover > 0).all():
        return codewords
    else:
        # Invalid - fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
