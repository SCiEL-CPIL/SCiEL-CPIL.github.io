"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 55  # safety margin
    
    max_target = 128
    
    # Generate smooth numbers (2^a * 3^b * 5^c) up to max_target
    smooth = []
    for x in range(1, max_target + 1):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    n_smooth = len(smooth)
    smooth_set = set(smooth)
    
    # Precompute: for each smooth number, which other smooth numbers it divides
    # divs[i] = set of indices j where smooth[i] divides smooth[j]
    # Actually we need: can we use smooth[i] and smooth[j] together? 
    # They're compatible if neither divides the other.
    
    # Precompute compatibility matrix as bitmasks for speed
    # compat[i] = bitmask of j such that smooth[i] and smooth[j] are compatible
    compat = [0] * n_smooth
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(n_smooth):
            sj = smooth[j]
            if i == j:
                continue
            # Check if si divides sj or sj divides si
            if sj % si == 0 or si % sj == 0:
                continue
            compat[i] |= (1 << j)
    
    answers = []
    
    for n in range(1, max_target + 1):
        if time.time() > deadline:
            # Return what we have so far, fill rest with a fallback
            # This shouldn't happen with our efficient approach
            break
        
        # Filter smooth numbers <= n
        candidates = [x for x in smooth if x <= n]
        cands_idx = [smooth.index(x) for x in candidates]
        
        found = None
        
        # Try 1 term
        if n in smooth_set:
            found = [n]
        
        # Try 2 terms
        if found is None:
            # For each pair of candidates, check if they sum to n and are compatible
            m = len(candidates)
            for i in range(m):
                a = candidates[i]
                b = n - a
                if b <= a:  # avoid duplicates, a < b
                    continue
                if b not in smooth_set or b > n:
                    continue
                # Find index of b
                # Check compatibility
                j = smooth.index(b) if b in smooth_set else -1
                if j < 0:
                    continue
                # Check if a and b are compatible (neither divides the other)
                if b % a == 0 or a % b == 0:
                    continue
                found = [a, b]
                break
        
        # Try 3 terms
        if found is None:
            m = len(candidates)
            for i in range(m):
                a = candidates[i]
                rem = n - a
                if rem <= 0:
                    continue
                # Find pairs in candidates that sum to rem
                for j in range(i + 1, m):
                    b = candidates[j]
                    if b >= rem:
                        break
                    c = rem - b
                    if c <= b:
                        continue
                    if c not in smooth_set or c > n:
                        continue
                    k = smooth.index(c) if c in smooth_set else -1
                    if k < 0:
                        continue
                    # Check all three are pairwise compatible
                    if b % a == 0 or a % b == 0:
                        continue
                    if c % a == 0 or a % c == 0:
                        continue
                    if c % b == 0 or b % c == 0:
                        continue
                    found = [a, b, c]
                    break
                if found:
                    break
        
        # Try 4 terms if needed
        if found is None:
            m = len(candidates)
            for i in range(m):
                a = candidates[i]
                rem1 = n - a
                if rem1 <= 0:
                    continue
                for j in range(i + 1, m):
                    b = candidates[j]
                    rem2 = rem1 - b
                    if rem2 <= 0:
                        break
                    for k in range(j + 1, m):
                        c = candidates[k]
                        if c >= rem2:
                            break
                        d = rem2 - c
                        if d <= c:
                            continue
                        if d not in smooth_set or d > n:
                            continue
                        l = smooth.index(d) if d in smooth_set else -1
                        if l < 0:
                            continue
                        # Check all four are pairwise compatible
                        valid = True
                        for p in range(4):
                            for q in range(p + 1, 4):
                                t1 = [a, b, c, d][p]
                                t2 = [a, b, c, d][q]
                                if t2 % t1 == 0 or t1 % t2 == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            found = [a, b, c, d]
                            break
                    if found:
                        break
                if found:
                    break
        
        if found is None:
            # Fallback: should not happen for targets 1..128
            # Use a simple valid representation
            found = [n] if n in smooth_set else [1, n - 1] if (n - 1) in smooth_set and (n - 1) % 1 != 0 or 1 % (n-1) != 0 else [2, 5, n - 7]
        
        answers.append(found)
    
    # If we hit the deadline, fill remaining with a safe fallback
    while len(answers) < max_target:
        n = len(answers) + 1
        if n in smooth_set:
            answers.append([n])
        else:
            # Find any valid 2-term or 3-term representation
            candidates = [x for x in smooth if x <= n]
            found = None
            m = len(candidates)
            for i in range(m):
                a = candidates[i]
                b = n - a
                if b <= a:
                    continue
                if b in smooth_set and b % a != 0 and a % b != 0:
                    found = [a, b]
                    break
            if found is None:
                for i in range(m):
                    a = candidates[i]
                    rem = n - a
                    for j in range(i + 1, m):
                        b = candidates[j]
                        if b >= rem:
                            break
                        c = rem - b
                        if c <= b or c not in smooth_set:
                            continue
                        if b % a != 0 and a % b != 0 and c % a != 0 and a % c != 0 and c % b != 0 and b % c != 0:
                            found = [a, b, c]
                            break
                    if found:
                        break
            if found is None:
                found = [n]  # last resort
            answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
