import time
from math import isqrt

def _square_free_kernel(n):
    kernel = set()
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            n //= d
            count += 1
        if count % 2 == 1:
            kernel.add(d)
        d += 1 if d == 2 else 2
    if n > 1:
        kernel.add(n)
    return frozenset(kernel)

def construct():
    start = time.time()
    deadline = start + 80
    
    target = frozenset({2, 3, 5, 7})
    
    # basis: list of (frozenset, int) where frozenset is the reduced basis vector
    # pivot: dict prime -> index in basis
    basis = []
    pivot = {}
    
    # kernel_to_int: frozenset -> int (smallest int with that kernel)
    kernel_to_int = {}
    
    for M in range(211, 10211):
        if time.time() > deadline:
            break
        
        k = _square_free_kernel(M)
        
        if k not in kernel_to_int:
            kernel_to_int[k] = M
            
            # Try to add k to basis
            reduced = set(k)
            # Reduce against existing basis
            for i, (b, _) in enumerate(basis):
                if pivot[i] in reduced:
                    reduced = reduced ^ set(b)
            
            if reduced:
                p = min(reduced)
                if p not in pivot:
                    idx = len(basis)
                    basis.append((frozenset(reduced), M))
                    pivot[p] = idx
        
        # Check if target is in span
        remaining = set(target)
        for i, (b, _) in enumerate(basis):
            if pivot[i] in remaining:
                remaining = remaining ^ set(b)
        
        if not remaining:
            # Reconstruct
            selected = []
            rem = set(target)
            for i in range(len(basis) - 1, -1, -1):
                b, val = basis[i]
                if pivot[i] in rem:
                    selected.append(val)
                    rem = rem ^ set(b)
            
            if selected:
                product = 210
                for x in selected:
                    product *= x
                s = isqrt(product)
                if s * s == product and len(selected) == len(set(selected)):
                    return selected
    
    return [840]
