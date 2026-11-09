import math
import time

def _squarefree_kernel(n):
    """Compute the squarefree kernel of n."""
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            if cnt % 2 == 1:
                result *= d
        d += 1 if d == 2 else 2
    if n > 1:
        result *= n
    return result

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    max_val = 10210
    target_kernel = 210
    
    # Precompute kernels
    kernels = [0] * (max_val + 1)
    for i in range(1, max_val + 1):
        kernels[i] = _squarefree_kernel(i)
    
    # Map kernel to list of numbers having that kernel
    # We will update this as M increases
    # But for simplicity in the loop, we can just scan
    # Since M is expected to be small, scanning is fine
    
    best_solution = [840]
    best_max_val = 840
    
    # Check single numbers
    for m in range(2, 100):
        n = 210 * m * m
        if n > 210 and n <= max_val:
            if n < best_max_val:
                best_max_val = n
                best_solution = [n]
            break # First one is smallest
    
    # Iterate M from 211 upwards
    # We stop if M >= best_max_val
    for M in range(211, best_max_val):
        if time.time() > deadline:
            break
            
        # Check pairs
        found = False
        for a in range(211, M):
            ka = kernels[a]
            # Calculate required kernel for b
            # P_b = P_a Delta {2,3,5,7}
            # This is equivalent to sf(ka * 210 / gcd(ka, 210)^2 * ...)?
            # Simpler: sf(ka * kb) = 210
            # kb = 210 * ka * s^2 / (common factors)^2 ?
            # Let's just compute the required kernel using the symmetric difference logic
            # Or just: we need sf(ka * kb) == 210
            # This means ka * kb = 210 * square
            # So kb = 210 * square / ka
            # For kb to be integer, ka must divide 210 * square
            # Since ka is squarefree, this means any prime in ka not in 210 must be in square
            # Actually, let's just iterate b and check
            # But that's O(M^2)
            
            # Optimization: 
            # The condition sf(ka * kb) == 210 implies kb is uniquely determined up to squares?
            # No, kb is squarefree.
            # sf(ka * kb) = 210
            # Let X = ka * kb.
            # sf(X) = 210.
            # X = 210 * y^2.
            # ka * kb = 210 * y^2.
            # kb = 210 * y^2 / ka.
            # For a fixed ka, kb is determined by y.
            # But kb must be squarefree.
            # And we need to find if such a b exists in (a, M].
            
            # Let's just scan b for small M
            for b in range(a + 1, M + 1):
                kb = kernels[b]
                prod_kernel = _squarefree_kernel(ka * kb)
                if prod_kernel == target_kernel:
                    if M < best_max_val:
                        best_max_val = M
                        best_solution = [a, b]
                    found = True
                    break
            if found:
                break
        
        if found:
            break
            
    # Verify best solution
    if best_solution:
        prod = 210
        for x in best_solution:
            prod *= x
        if not _is_square(prod):
            best_solution = [840]
            best_max_val = 840
            
    result = [int(x) for x in best_solution]
    return result
