"""
Maximize n for a K2,2,2-free 2-colouring of K_n.
Uses Paley graph construction for n=29, the known maximum.
"""
import time

def _paley_coloring(n):
    """
    Generate the coloring for the Paley graph on n vertices.
    n must be a prime p = 1 mod 4.
    Edge (i, j) is color 0 if (j-i) is a quadratic residue mod n, else 1.
    """
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            colors.append(0 if d in squares else 1)
    return colors

def _has_k222(color_list, n, deadline=None):
    """
    Check if the coloring contains K2,2,2 in either color.
    Uses bitset adjacency for efficiency.
    Returns True if K2,2,2 found, False otherwise.
    """
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = color_list[idx]
            idx += 1
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
    
    for c in [0, 1]:
        adj = adj0 if c == 0 else adj1
        found = False
        for i in range(n):
            if deadline and time.time() > deadline:
                return True  # Conservative: assume found if timed out
            for j in range(i + 1, n):
                common_mask = adj[i] & adj[j]
                common_mask &= ~(1 << i) & ~(1 << j)
                
                S = []
                m = common_mask
                while m:
                    v = (m & -m).bit_length() - 1
                    S.append(v)
                    m &= m - 1
                
                s_len = len(S)
                if s_len < 4:
                    continue
                
                for bi in range(s_len):
                    b1 = S[bi]
                    for bj in range(bi + 1, s_len):
                        b2 = S[bj]
                        cand_mask = common_mask & adj[b1] & adj[b2] & ~(1 << b1) & ~(1 << b2)
                        if cand_mask == 0:
                            continue
                        v1 = (cand_mask & -cand_mask).bit_length() - 1
                        cand_mask &= ~(1 << v1)
                        if cand_mask == 0:
                            continue
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # The known record is n=29. Paley(29) is a valid K2,2,2-free coloring.
    n = 29
    colors = _paley_coloring(n)
    
    # Validate with deadline guard. Paley(29) is known valid, so this should pass.
    if _has_k222(colors, n, deadline):
        # Fallback to smaller known-good Paley graph if validation fails
        n = 17
        colors = _paley_coloring(n)
    
    return {'n': n, 'colors': colors}
