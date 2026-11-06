<<<<<<< SEARCH
def _test_specific_circulants(n, time_limit=3.0):
    """Test specific algebraic circulant connection sets that are known to be triangle-free."""
    start = time.time()
    half = n // 2
    candidates = []
    
    # Test sets based on quadratic residues (Paley-like)
    if n % 4 == 1:
        # Quadratic residues mod n
        QR = set()
        for x in range(1, n):
            if (x * x) % n in range(1, n):
                QR.add((x * x) % n)
        # Symmetrize
        S = set()
        for x in QR:
            if 0 < x < n:
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Test small structured sets
    for k in [2, 3, 5, 7]:
        S = set()
        for x in range(1, n):
            if x % k in [1, k-1]:  # +/-1 mod k
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Test sets that avoid sum collisions
    for a in range(2, min(8, half)):
        for b in range(a + 1, min(12, half)):
            S = {a, b, (-a) % n, (-b) % n}
            # Check if triangle-free first (fast)
            if not _has_triangle_circulant(n, frozenset(S)):
                candidates.append(frozenset(S))
    
    # Test union of two small sets
    for a in range(2, min(6, half)):
        for b in range(a + 1, min(10, half)):
            for c in range(b + 1, min(14, half)):
                S = {a, b, c, (-a) % n, (-b) % n, (-c) % n}
                if not _has_triangle_circulant(n, frozenset(S)):
                    candidates.append(frozenset(S))
    
    # Test larger structured sets
    for base in range(2, 8):
        S = set()
        for k in range(1, n // base + 1):
            x = k * base
            if 0 < x < n and x % base in [1, base-1, 2, base-2]:
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Test specific known-good patterns
    patterns = [
        [1, 2, 4, 8],  # Powers of 2
        [1, 3, 5, 7],  # Odd numbers
        [2, 4, 6, 8],  # Even numbers
        [1, 4, 7, 10], # AP with step 3
        [2, 5, 8, 11], # AP with step 3, offset 2
    ]
    for pat in patterns:
        S = set()
        for x in pat:
            if 0 < x < n:
                S.add(x)
                S.add((-x) % n)
        if S and not _has_triangle_circulant(n, frozenset(S)):
            candidates.append(frozenset(S))
    
    for S in candidates:
        if time.time() - start > time_limit:
            break
        valid, alpha = _verify_circulant(n, S)
        if valid:
            return S
    return None
=======
def _test_specific_circulants(n, time_limit=3.0):
    """Test specific algebraic circulant connection sets that are known to be triangle-free."""
    start = time.time()
    half = n // 2
    candidates = []
    
    # Test sets based on quadratic residues (Paley-like)
    if n % 4 == 1:
        # Quadratic residues mod n
        QR = set()
        for x in range(1, n):
            if (x * x) % n in range(1, n):
                QR.add((x * x) % n)
        # Symmetrize
        S = set()
        for x in QR:
            if 0 < x < n:
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Test small structured sets
    for k in [2, 3, 5, 7]:
        S = set()
        for x in range(1, n):
            if x % k in [1, k-1]:  # +/-1 mod k
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Test sets that avoid sum collisions - expanded range
    for a in range(2, min(10, half)):
        for b in range(a + 1, min(15, half)):
            S = {a, b, (-a) % n, (-b) % n}
            # Check if triangle-free first (fast)
            if not _has_triangle_circulant(n, frozenset(S)):
                candidates.append(frozenset(S))
    
    # Test union of two small sets - expanded
    for a in range(2, min(8, half)):
        for b in range(a + 1, min(12, half)):
            for c in range(b + 1, min(16, half)):
                S = {a, b, c, (-a) % n, (-b) % n, (-c) % n}
                if not _has_triangle_circulant(n, frozenset(S)):
                    candidates.append(frozenset(S))
    
    # Test union of three small sets
    for a in range(2, min(6, half)):
        for b in range(a + 1, min(9, half)):
            for c in range(b + 1, min(12, half)):
                for d in range(c + 1, min(15, half)):
                    S = {a, b, c, d, (-a) % n, (-b) % n, (-c) % n, (-d) % n}
                    if not _has_triangle_circulant(n, frozenset(S)):
                        candidates.append(frozenset(S))
    
    # Test larger structured sets
    for base in range(2, 10):
        S = set()
        for k in range(1, n // base + 1):
            x = k * base
            if 0 < x < n and x % base in [1, base-1, 2, base-2]:
                S.add(x)
                S.add((-x) % n)
        if S:
            candidates.append(frozenset(S))
    
    # Test specific known-good patterns - expanded
    patterns = [
        [1, 2, 4, 8],  # Powers of 2
        [1, 3, 5, 7],  # Odd numbers
        [2, 4, 6, 8],  # Even numbers
        [1, 4, 7, 10], # AP with step 3
        [2, 5, 8, 11], # AP with step 3, offset 2
        [1, 2, 3, 5],  # Fibonacci-like
        [2, 3, 5, 8],  # Fibonacci-like
        [1, 3, 4, 6],  # Mixed
        [2, 4, 7, 9],  # Mixed
        [1, 2, 5, 7],  # Mixed
        [3, 5, 7, 11], # Primes
        [2, 5, 7, 11], # Primes
        [1, 4, 6, 9],  # Mixed
        [2, 3, 6, 9],  # Mixed
        [1, 5, 6, 10], # Mixed
        [3, 4, 8, 12], # Mixed
    ]
    for pat in patterns:
        S = set()
        for x in pat:
            if 0 < x < n:
                S.add(x)
                S.add((-x) % n)
        if S and not _has_triangle_circulant(n, frozenset(S)):
            candidates.append(frozenset(S))
    
    # Test near-cubic patterns
    for a in range(2, min(8, half)):
        for b in range(a + 1, min(12, half)):
            for c in range(b + 1, min(15, half)):
                for d in range(c + 1, min(18, half)):
                    S = {a, b, c, d, (-a) % n, (-b) % n, (-c) % n, (-d) % n}
                    if not _has_triangle_circulant(n, frozenset(S)):
                        candidates.append(frozenset(S))
    
    for S in candidates:
        if time.time() - start > time_limit:
            break
        valid, alpha = _verify_circulant(n, S)
        if valid:
            return S
    return None
>>>>>>> REPLACE

