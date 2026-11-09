import numpy as np
import time
import random

def _construct_qr_code_17():
    """
    Construct the 256-word [17, 8, 6] linear code by shortening the [18, 9, 6] QR code.
    """
    # Generator polynomial for [18, 9, 6] QR code
    # g(x) = x^9 + x^7 + x^6 + x^4 + x^3 + x^2 + 1
    g_mask = 0
    g_poly = [1, 0, 1, 1, 1, 0, 1, 1, 0, 1]
    for j in range(10):
        if g_poly[j]:
            g_mask |= (1 << j)
            
    codewords_ext = []
    for m in range(256):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_mask << i)
        # Check if the 18th bit (index 17) is 0
        if (c >> 17) & 1 == 0:
            codewords_ext.append(c & ((1 << 17) - 1))
            
    return codewords_ext

def _precompute_flip_masks(n, max_dist):
    """Precompute masks for flipping exactly d bits, for d in 0..max_dist."""
    masks = [[] for _ in range(max_dist + 1)]
    masks[0].append(0)
    for d in range(1, max_dist + 1):
        for m in masks[d - 1]:
            for i in range(n):
                if not (m & (1 << i)):
                    masks[d].append(m | (1 << i))
    return masks

def construct():
    """Construct a large binary code of length 17 with minimum distance 6."""
    deadline = time.time() + 55.0
    N = 17
    MIN_DIST = 6
    MAX_DIST_FORBID = MIN_DIST - 1  # 5
    
    # 1. Get the linear code baseline (256 words)
    linear_code = _construct_qr_code_17()
    linear_code = list(set(linear_code))
    
    # 2. Precompute flip masks
    flip_masks = _precompute_flip_masks(N, MAX_DIST_FORBID)
    
    # 3. Greedy nonlinear expansion
    total_words = 1 << N
    rng = random.Random(42)
    candidates = list(range(total_words))
    
    best_code = []
    best_size = 0
    
    # Try multiple random orderings
    for restart in range(10):
        if time.time() > deadline - 2:
            break
            
        forbidden = set(linear_code)
        # Add neighbors of linear code to forbidden
        for w in linear_code:
            for d in range(1, MAX_DIST_FORBID + 1):
                for m in flip_masks[d]:
                    forbidden.add(w ^ m)
                    
        current_code = list(linear_code)
        
        # Shuffle candidates
        rng.shuffle(candidates)
        
        for w in candidates:
            if w not in forbidden:
                current_code.append(w)
                # Update forbidden
                for d in range(1, MAX_DIST_FORBID + 1):
                    for m in flip_masks[d]:
                        forbidden.add(w ^ m)
                        
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code
            
    return best_code
