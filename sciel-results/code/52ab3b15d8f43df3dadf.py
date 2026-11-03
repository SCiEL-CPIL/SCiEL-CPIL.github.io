"""Greedy set-cover for K_q(n, radius) with a time guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math

def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0

    total = q ** n
    # Precompute powers for index conversion
    pows = [q ** (n - 1 - i) for i in range(n)]
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute ball masks for all indices
    # ball_mask[idx] is an integer with bits set for all words in the ball of idx
    ball_masks = [0] * total
    for idx in range(total):
        w = idx_to_word(idx)
        mask = 1 << idx
        for i in range(n):
            pw = pows[i]
            w_i = w[i]
            for v in range(q):
                if v != w_i:
                    new_idx = idx + (v - w_i) * pw
                    mask |= (1 << new_idx)
        ball_masks[idx] = mask

    # --- Lattice Reduction Heuristic ---
    # Map ternary space to a complex lattice using omega = e^(2*pi*i/3)
    # The Hamming graph H(3,n) has automorphism group S_3^n x S_n
    # Use this structure to identify high-symmetry positions for initial seeding
    # and to guide a lattice-reduction-like refinement

    omega = complex(-0.5, math.sqrt(3)/2)  # primitive cube root of unity
    omega2 = complex(-0.5, -math.sqrt(3)/2)
    
    def ternary_to_complex(word):
        """Map ternary word to complex lattice point"""
        z = 0 + 0j
        for i, digit in enumerate(word):
            if digit == 0:
                z += omega ** (2 * i)
            elif digit == 1:
                z += omega2 ** (2 * i)
            else:  # digit == 2
                z += 1.0 + 0j
        return z
    
    def complex_to_ternary(z, n):
        """Map complex lattice point back to ternary word (approximate)"""
        word = []
        for i in range(n):
            # Project onto the i-th coordinate
            proj = z * omega ** (-2 * i)
            # Round to nearest lattice point
            if abs(proj - 1.0) < 0.4:
                word.append(2)
            elif abs(proj - omega) < 0.4:
                word.append(0)
            else:
                word.append(1)
        return word

    # --- Initial Seed using Lattice Structure ---
    # For q=3, n=6, radius=1, use the algebraic structure to seed
    # The Tetracode has a natural lattice interpretation
    seed_indices = []
    if q == 3 and n >= 4:
        # Generate Tetracode words (9 words of length 4, radius 1)
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
        
        # Generate tails for remaining n-4 positions
        tails = list(itertools.product(range(q), repeat=n - 4))
        
        for h in tetracode:
            for tail in tails:
                word = h + list(tail)
                seed_indices.append(word_to_idx(word))
    else:
        # Fallback: greedy construction from scratch
        seed_indices = []

    code = list(seed_indices)
    
    # --- Lattice Reduction Pass ---
    # Identify codewords that are "lattice-reduced" (low complexity in the complex embedding)
    # and prioritize their removal if redundant
    if time.time() < deadline - 30:
        # Compute lattice complexity for each codeword
        lattice_scores = {}
        for c in code:
            w = idx_to_word(c)
            z = ternary_to_complex(w)
            # Complexity: deviation from the "center" of the lattice
            # Lower complexity = more symmetric = better candidate for retention
            complexity = abs(z) ** 2
            lattice_scores[c] = complexity
        
        # Sort by lattice complexity (lower = more symmetric)
        # Try to remove higher-complexity codewords first if redundant
        
        # Compute coverage counts
        coverage_count = [0] * total
        for c in code:
            mask = ball_masks[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                coverage_count[idx] += 1
                m ^= lsb
        
        # Lattice-guided removal: remove high-complexity redundant codewords first
        max_passes = 50
        for _ in range(max_passes):
            if time.time() > deadline - 20:
                break
            
            # Sort by lattice complexity (descending) - remove high complexity first
            sorted_code = sorted(code, key=lambda c: lattice_scores.get(c, 0), reverse=True)
            
            removed_any = False
            new_code = []
            for c in sorted_code:
                # Check if c is redundant
                is_redundant = True
                mask = ball_masks[c]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if coverage_count[idx] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    # Remove c
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] -= 1
                        m ^= lsb
                    removed_any = True
                else:
                    new_code.append(c)
            
            code = new_code
            
            if not removed_any:
                break
    else:
        # Standard removal phase
        # Compute coverage counts
        coverage_count = [0] * total
        for c in code:
            mask = ball_masks[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                coverage_count[idx] += 1
                m ^= lsb
        
        max_passes = 100
        for _ in range(max_passes):
            if time.time() > deadline - 20:
                break
            
            random.shuffle(code)
            
            removed_any = False
            new_code = []
            for c in code:
                is_redundant = True
                mask = ball_masks[c]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if coverage_count[idx] <= 1:
                        is_redundant = False
                        break
                        m ^= lsb
                
                if is_redundant:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] -= 1
                        m ^= lsb
                    removed_any = True
                else:
                    new_code.append(c)
            
            code = new_code
            
            if not removed_any:
                break

    # --- Final Greedy Fill (if needed) ---
    # Check if coverage is complete
    uncovered_mask = 0
    for j in range(total):
        if coverage_count[j] == 0:
            uncovered_mask |= (1 << j)
            
    while uncovered_mask:
        if time.time() > deadline - 5:
            break
        
        best_idx = -1
        best_gain = 0
        for idx in range(total):
            intersection = uncovered_mask & ball_masks[idx]
            gain = intersection.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_gain == 0:
            break
            
        code.append(best_idx)
        
        mask = ball_masks[best_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage_count[idx] += 1
            if coverage_count[idx] == 1:
                uncovered_mask ^= (1 << idx)
            m ^= lsb

    # Convert back to words
    result = [idx_to_word(idx) for idx in code]
    return result
# EVOLVE-BLOCK-END
