"""Improved program: construct a 5-colouring of K_n with no monochromatic C4."""
import time
import random

def _has_c4_in_color(colors, n, c):
    """Check if colour class c contains a C4.
    A C4 exists iff there are two vertices u,v that share two common neighbours in colour c.
    Equivalently, for some pair (u,v), |N_c(u) ∩ N_c(v)| >= 2.
    """
    # Build adjacency sets for colour c
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if colors[j - i * (2 * n - i - 1) // 2] == c:
                adj[i].add(j)
                adj[j].add(i)
    # Check for C4: for each pair (u,v), count common neighbours
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return True
    return False

def _is_valid(colors, n):
    """Check all 5 colour classes for C4."""
    for c in range(5):
        if _has_c4_in_color(colors, n, c):
            return False
    return True

def _index(i, j, n):
    """Lexicographic index of pair (i,j) with i<j in the flat list."""
    # Number of pairs before row i: i*(2*n - i - 1)//2
    # Within row i, position is (j - i - 1)
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _build_color_array(n, colors_list):
    """Convert list of colors (lex order) to a 2D-like lookup via flat array."""
    return colors_list

def _check_c4_fast(colors, n, c):
    """Faster C4 check using bitsets (Python integers as bitsets)."""
    adj = [0] * n  # bitset of neighbours in colour c
    for i in range(n):
        for j in range(i + 1, n):
            idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
            if colors[idx] == c:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            # Check if common has at least 2 bits set
            if common & (common - 1):  # has at least 2 set bits
                return True
    return False

def _is_valid_fast(colors, n):
    for c in range(5):
        if _check_c4_fast(colors, n, c):
            return False
    return True

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with the known 11-vertex construction (cyclic group Z_11, distances 1..5)
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to extend to larger n via local search / randomised construction
    # Strategy: try to build colourings for n = 12, 13, ... up to a limit
    # For each n, try multiple random initialisations + local repair
    
    max_n_try = 16  # Try up to 16; if we find valid, great
    
    for target_n in range(12, max_n_try + 1):
        if time.time() > deadline - 10:
            break
        
        # Multiple attempts for each target_n
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            # Random initial colouring
            colors = [random.randint(0, 4) for _ in range(target_n * (target_n - 1) // 2)]
            
            # Local search: iteratively fix C4 violations
            improved = True
            iters = 0
            max_iters = 5000
            while improved and iters < max_iters:
                if time.time() > deadline - 10:
                    break
                improved = False
                iters += 1
                
                # Find a C4 violation in some colour
                found = False
                for c in range(5):
                    if _check_c4_fast(colors, target_n, c):
                        # Find the specific C4 and recolor an edge
                        # Build adjacency
                        adj = [0] * target_n
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                idx = i * (2 * target_n - i - 1) // 2 + (j - i - 1)
                                if colors[idx] == c:
                                    adj[i] |= (1 << j)
                                    adj[j] |= (1 << i)
                        
                        # Find a pair with >= 2 common neighbours
                        for u in range(target_n):
                            for v in range(u + 1, target_n):
                                common = adj[u] & adj[v]
                                if common & (common - 1):
                                    # Find two common neighbours
                                    bits = common
                                    # Get first two set bits
                                    b1 = bits & (-bits)
                                    bits ^= b1
                                    b2 = bits & (-bits)
                                    n1 = b1.bit_length() - 1
                                    n2 = b2.bit_length() - 1
                                    
                                    # Recolour edge (u, n1) or (v, n1) or (u, n2) or (v, n2)
                                    # Try recoloring (u, n1)
                                    options = []
                                    for (a, b) in [(u, n1), (v, n1), (u, n2), (v, n2)]:
                                        if a < b:
                                            options.append((a, b))
                                        elif b < a:
                                            options.append((b, a))
                                    
                                    random.shuffle(options)
                                    for (a, b) in options:
                                        idx = a * (2 * target_n - a - 1) // 2 + (b - a - 1)
                                        old_c = colors[idx]
                                        # Try a new colour
                                        new_c = random.randint(0, 4)
                                        if new_c != old_c:
                                            colors[idx] = new_c
                                            # Quick check: does this create a new C4 in new_c?
                                            if not _check_c4_fast(colors, target_n, new_c):
                                                improved = True
                                                found = True
                                                break
                                            else:
                                                colors[idx] = old_c
                                    
                                    if found:
                                        break
                            if found:
                                break
                        if found:
                            break
                
                if not found:
                    # No C4 found in any colour — valid!
                    if _is_valid_fast(colors, target_n):
                        best_n = target_n
                        best_colors = colors[:]
                        # Try to go higher
                        break
                    else:
                        # Shouldn't happen if _check_c4_fast is correct
                        break
            
            if best_n >= target_n:
                break
    
    # If we haven't found anything better than 11, return the base
    if best_n == 11:
        return {'n': 11, 'colors': best_colors}
    
    return {'n': best_n, 'colors': best_colors}
