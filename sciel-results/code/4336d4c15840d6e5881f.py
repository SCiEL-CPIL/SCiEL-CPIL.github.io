"""Optimized divisibility-antichain representations by smooth numbers."""

def construct():
    # Precomputed optimal representations for targets 1..128
    # Each entry is a list of distinct 2,3,5-smooth numbers summing to the target,
    # forming a divisibility antichain, with minimum number of terms.
    
    # Smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # Precompute all valid antichain representations for each target
    # with minimum number of terms
    results = [None] * 129  # index 0 unused
    
    for n in range(1, 129):
        # Try 1 term
        if n in smooth:
            results[n] = [n]
            continue
        
        # Try 2 terms
        found = False
        for a in smooth:
            if a >= n:
                break
            b = n - a
            if b in smooth and a != b:
                # Check antichain: a does not divide b, b does not divide a
                if b % a != 0 and a % b != 0:
                    results[n] = [a, b]
                    found = True
                    break
        if found:
            continue
        
        # Try 3 terms
        found = False
        for a in smooth:
            if a >= n:
                break
            for b in smooth:
                if b <= a or a + b >= n:
                    continue
                c = n - a - b
                if c in smooth and c > b:
                    # Check all pairs for antichain
                    if (b % a != 0 and a % b != 0 and
                        c % a != 0 and a % c != 0 and
                        c % b != 0 and b % c != 0):
                        results[n] = [a, b, c]
                        found = True
                        break
            if found:
                break
        if found:
            continue
        
        # Try 4 terms as fallback (shouldn't be needed for 1..128)
        found = False
        for a in smooth:
            if a >= n:
                break
            for b in smooth:
                if b <= a or a + b >= n:
                    continue
                for c in smooth:
                    if c <= b or a + b + c >= n:
                        continue
                    d = n - a - b - c
                    if d in smooth and d > c:
                        # Check all pairs
                        terms = [a, b, c, d]
                        valid = True
                        for i in range(4):
                            for j in range(i+1, 4):
                                if terms[j] % terms[i] == 0 or terms[i] % terms[j] == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            results[n] = [a, b, c, d]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    
    # Return results for targets 1..128
    return [results[n] for n in range(1, 129)]
