"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_k4_fast(colors):
    """Check every K4 has at least 5 distinct colours. Returns True if valid."""
    n = 16
    edge = [[0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edge[i][j] = c
            edge[j][i] = c
            idx += 1
    
    for a in range(n):
        ea = edge[a]
        for b in range(a + 1, n):
            eb = edge[b]
            ab = ea[b]
            for c in range(b + 1, n):
                ac = ea[c]
                bc = eb[c]
                ec = edge[c]
                for d in range(c + 1, n):
                    mask = (1 << ab) | (1 << ac) | (1 << ea[d]) | (1 << bc) | (1 << eb[d]) | (1 << ec[d])
                    if mask.bit_count() < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def _gen_linear(a, b, p, n=16):
    """Generate colors for (a*i + b*j) % p for all pairs i<j."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((a * i + b * j) % p)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    best_colors = None
    best_count = 999
    
    # Start with the known base solution
    base_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    if _check_k4_fast(base_colors):
        best_colors = base_colors
        best_count = _num_colours(base_colors)
    
    # Exhaustive search over (a*i + b*j) % p
    # Try primes from 13 up to reasonable limits
    # With the fast checker, we can afford many more trials
    primes = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    for p in primes:
        if time.time() > deadline - 5:
            break
        max_coeff = min(p, 16)  # coefficients up to 16 (or p if smaller)
        for a in range(1, max_coeff + 1):
            if time.time() > deadline - 5:
                break
            for b in range(a, max_coeff + 1):  # b >= a for symmetry reduction
                colors = _gen_linear(a, b, p)
                cnt = _num_colours(colors)
                if cnt < best_count:
                    if _check_k4_fast(colors):
                        best_colors = colors
                        best_count = cnt
                        if best_count <= 13:
                            return colors
    
    # Try quadratic forms: (a*i + b*j + c*i*j) % p
    for p in [13, 17, 19, 23, 29]:
        if time.time() > deadline - 5:
            break
        max_c = min(p, 8)
        for a in range(1, min(p, 6) + 1):
            if time.time() > deadline - 5:
                break
            for b in range(1, min(p, 6) + 1):
                for c in range(0, max_c):
                    colors = []
                    for i in range(16):
                        for j in range(i + 1, 16):
                            colors.append((a * i + b * j + c * i * j) % p)
                    cnt = _num_colours(colors)
                    if cnt < best_count:
                        if _check_k4_fast(colors):
                            best_colors = colors
                            best_count = cnt
                            if best_count <= 13:
                                return colors
    
    # Try (i*j) % p
    for p in [13, 17, 19, 23, 29, 31]:
        if time.time() > deadline - 5:
            break
        colors = [(i * j) % p for i in range(16) for j in range(i + 1, 16)]
        cnt = _num_colours(colors)
        if cnt < best_count and _check_k4_fast(colors):
            best_colors = colors
            best_count = cnt
    
    # Try (i^2 + j) % p and (i + j^2) % p
    for p in [13, 17, 19, 23, 29]:
        if time.time() > deadline - 5:
            break
        for form in ['i2j', 'ij2']:
            colors = []
            for i in range(16):
                for j in range(i + 1, 16):
                    if form == 'i2j':
                        colors.append((i * i + j) % p)
                    else:
                        colors.append((i + j * j) % p)
            cnt = _num_colours(colors)
            if cnt < best_count and _check_k4_fast(colors):
                best_colors = colors
                best_count = cnt
    
    # Try (a*i^2 + b*j^2 + c*i*j) % p
    for p in [13, 17, 19, 23]:
        if time.time() > deadline - 5:
            break
        for a in range(1, min(p, 5) + 1):
            if time.time() > deadline - 5:
                break
            for b in range(1, min(p, 5) + 1):
                for c in range(0, min(p, 5) + 1):
                    colors = []
                    for i in range(16):
                        for j in range(i + 1, 16):
                            colors.append((a * i * i + b * j * j + c * i * j) % p)
                    cnt = _num_colours(colors)
                    if cnt < best_count and _check_k4_fast(colors):
                        best_colors = colors
                        best_count = cnt
                        if best_count <= 13:
                            return colors
    
    # Local improvement: try to reduce colors by remapping
    if best_colors is not None and time.time() < deadline - 10:
        current = list(best_colors)
        current_count = best_count
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for idx in range(120):
                if time.time() > deadline - 3:
                    break
                orig = current[idx]
                for new_c in range(current_count):
                    if new_c == orig:
                        continue
                    current[idx] = new_c
                    if _check_k4_fast(current):
                        new_count = _num_colours(current)
                        if new_count < current_count:
                            current_count = new_count
                            best_count = current_count
                            best_colors = current[:]
                            improved = True
                            if best_count <= 13:
                                return current
                            break
                current[idx] = orig
    
    if best_colors is None:
        return base_colors
    
    return best_colors
# EVOLVE-BLOCK-END
