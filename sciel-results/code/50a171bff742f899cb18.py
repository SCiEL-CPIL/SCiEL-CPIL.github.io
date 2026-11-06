"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _validate(coloring):
    """Check that every K4 has at least 5 distinct colors. Returns True if valid."""
    n = 16
    edge_color = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_color[i][j] = coloring[idx]
            edge_color[j][i] = coloring[idx]
            idx += 1
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    colors = set([
                        edge_color[a][b], edge_color[a][c], edge_color[a][d],
                        edge_color[b][c], edge_color[b][d], edge_color[c][d]
                    ])
                    if len(colors) < 5:
                        return False
    return True

def _count_colors(coloring):
    return len(set(coloring))

def _gf16_construct():
    """Construct edge coloring using GF(16) structure.
    
    GF(16) = GF(2)[x]/(x^4 + x + 1)
    We map vertices 0..15 to field elements.
    Color of edge {i,j} = trace(a_i * a_j) interpreted as 4-bit integer.
    """
    # GF(16) with primitive polynomial x^4 + x + 1
    # Elements: 0, 1, alpha, alpha^2, ..., alpha^14, where alpha^4 = alpha + 1
    
    # Build multiplication table for GF(16)
    # Represent elements as 4-bit integers
    # alpha = 2 (binary 0010)
    # alpha^2 = 4 (binary 0100)
    # alpha^3 = 8 (binary 1000)
    # alpha^4 = alpha + 1 = 2 + 1 = 3 (binary 0011)
    # etc.
    
    def gf_mult(a, b):
        """Multiply two elements in GF(16)"""
        result = 0
        for i in range(4):
            if b & (1 << i):
                result ^= a
            # Multiply a by alpha: left shift, reduce if bit 4 is set
            a = (a << 1)
            if a & 16:  # x^4 term
                a ^= 0b10011  # x^4 + x + 1
        return result
    
    # Build lookup table
    mult_table = [[gf_mult(i, j) for j in range(16)] for i in range(16)]
    
    # Trace function: Tr(x) = x + x^2 + x^4 + x^8 (in GF(16))
    # But we need the trace to GF(2), which is just the sum of bits in the 4-bit representation
    # Actually, Tr_{GF(16)/GF(2)}(x) = x + x^2 + x^4 + x^8 mod 2
    # For a 4-bit element, this is the parity of the 4 bits
    
    # Wait, let me reconsider. The trace from GF(16) to GF(2) maps to GF(2), so it's 0 or 1.
    # That's not useful for coloring. Let me use a different approach.
    
    # Instead, let's use the 4-bit representation of a_i * a_j directly as the color.
    # This gives colors 0..15, and the algebraic structure should help.
    
    # Map vertices to field elements: vertex i -> element i (0..15)
    # Color of edge {i,j} = mult_table[i][j]
    
    coloring = []
    for i in range(16):
        for j in range(i+1, 16):
            coloring.append(mult_table[i][j])
    
    return coloring

def _additive_construct():
    """Simple additive coloring mod 17."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _try_merge_colors(current, deadline):
    """Try to merge color pairs to reduce total colors used."""
    current = current[:]
    current_count = _count_colors(current)
    best_count = current_count
    best_coloring = current[:]
    
    for iteration in range(10):
        if time.time() > deadline:
            break
        improved = False
        colors_in_use = sorted(set(current))
        for target_color in colors_in_use:
            if time.time() > deadline:
                break
            for source_color in colors_in_use:
                if source_color == target_color:
                    continue
                
                test = current[:]
                for idx in range(len(test)):
                    if test[idx] == source_color:
                        test[idx] = target_color
                
                new_count = _count_colors(test)
                if new_count >= current_count:
                    continue
                
                if _validate(test):
                    current = test
                    current_count = new_count
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_coloring = test[:]
        
        if not improved:
            break
    
    return best_coloring, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try multiple constructions and pick the best
    candidates = []
    
    # Candidate 1: GF(16) multiplication-based
    gf16_coloring = _gf16_construct()
    if _validate(gf16_coloring):
        candidates.append(gf16_coloring)
    
    # Candidate 2: Additive mod 17
    add_coloring = _additive_construct()
    if _validate(add_coloring):
        candidates.append(add_coloring)
    
    if not candidates:
        # Fallback: return additive (should be valid)
        return add_coloring
    
    # Try to improve each candidate by merging colors
    best_coloring = None
    best_count = float('inf')
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        improved, count = _try_merge_colors(candidate, deadline)
        if count < best_count:
            best_count = count
            best_coloring = improved
    
    if best_coloring is None:
        best_coloring = candidates[0]
    
    return best_coloring
# EVOLVE-BLOCK-END
