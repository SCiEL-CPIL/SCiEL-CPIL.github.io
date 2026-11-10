import time
import math

def construct():
    # Primes up to 1000
    primes = []
    for n in range(2, 1001):
        is_prime = True
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    
    n_primes = len(primes)
    
    # Initial residues based on seed logic: r_i = (i + 2) // 2 % p_i
    # But we can also try random or other initializations if seed is not enough.
    # Seed gives L=1000.
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Max range to check
    M = 5000
    
    # Time limit
    start_time = time.time()
    TIME_LIMIT = 115.0  # Leave margin for overhead
    
    def compute_counts(residues, M):
        counts = [0] * (M + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            # x = r, r+p, r+2p ... <= M
            # Note: x must be >= 1
            start = r
            if start < 1:
                start = 1 + p * ((1 - r + p - 1) // p) # Smallest x >= 1 such that x % p == r
                # Actually simpler:
                # x % p == r.
                # If r == 0, x can be p, 2p...
                # If r > 0, x can be r, r+p...
                # Let's just loop k from 0 while r + k*p <= M
                # But we need x >= 1.
                pass
            
            # Efficient loop:
            # First x >= 1 with x % p == r
            if r == 0:
                first_x = p
            else:
                first_x = r
            if first_x < 1:
                first_x = 1
            # Adjust first_x to be congruent to r mod p
            # If r == 0, first_x should be multiple of p.
            # If r > 0, first_x should be r + k*p.
            
            # Let's just use a simple loop for clarity first, optimize if needed.
            # But M=5000, 168 primes. 168*5000 = 840k ops. Very fast.
            for x in range(1, M + 1):
                if x % p == r:
                    counts[x] += 1
        return counts

    counts = compute_counts(residues, M)
    
    def get_L(counts):
        for x in range(1, len(counts)):
            if counts[x] < 2:
                return x - 1
        return len(counts) - 1
        
    L = get_L(counts)
    
    # Local search
    # We will try to extend L by fixing the first bad point.
    while time.time() - start_time < TIME_LIMIT:
        if L >= M - 1:
            break
            
        # Find first x > L with counts[x] < 2
        x_target = L + 1
        while x_target <= M and counts[x_target] >= 2:
            x_target += 1
            
        if x_target > M:
            break
            
        improved = False
        # Try changing each prime's residue to cover x_target
        for i, p in enumerate(primes):
            r_old = residues[i]
            r_new = x_target % p
            
            if r_new == r_old:
                continue
                
            # Check if changing r_old to r_new is safe for x <= L
            # We decrement counts at x where x % p == r_old
            # We increment counts at x where x % p == r_new
            # Safety: For all x <= L with x % p == r_old, counts[x] - 1 >= 2 => counts[x] >= 3
            safe = True
            # Iterate x <= L with x % p == r_old
            # First x >= 1 with x % p == r_old
            if r_old == 0:
                first_x = p
            else:
                first_x = r_old
            
            # Ensure first_x >= 1
            if first_x < 1:
                 # This shouldn't happen for r_old in [0, p-1] and x>=1
                 # If r_old=0, first_x=p. If r_old>0, first_x=r_old>=1.
                 pass
            
            x = first_x
            while x <= L:
                if counts[x] < 3:
                    safe = False
                    break
                x += p
            
            if safe:
                # Apply change
                # Decrement old
                x = first_x
                while x <= M:
                    counts[x] -= 1
                    x += p
                    
                # Increment new
                if r_new == 0:
                    first_x_new = p
                else:
                    first_x_new = r_new
                    
                x = first_x_new
                while x <= M:
                    counts[x] += 1
                    x += p
                    
                residues[i] = r_new
                improved = True
                # Update L
                new_L = get_L(counts)
                if new_L > L:
                    L = new_L
                break # Restart loop to find new bad point
        
        if not improved:
            # Stuck. Try a random perturbation or break?
            # To avoid infinite loop, break or try a different strategy.
            # Let's try a few random swaps to escape local minima.
            escaped = False
            for _ in range(10):
                i = int(time.time() * 1000) % n_primes # Pseudo random based on time
                p = primes[i]
                r_old = residues[i]
                r_new = (r_old + 1) % p
                
                if r_new == r_old:
                    continue
                    
                # Check safety
                safe = True
                if r_old == 0:
                    first_x = p
                else:
                    first_x = r_old
                    
                x = first_x
                while x <= L:
                    if counts[x] < 3:
                        safe = False
                        break
                    x += p
                    
                if safe:
                    # Apply
                    x = first_x
                    while x <= M:
                        counts[x] -= 1
                        x += p
                        
                    if r_new == 0:
                        first_x_new = p
                    else:
                        first_x_new = r_new
                        
                    x = first_x_new
                    while x <= M:
                        counts[x] += 1
                        x += p
                        
                    residues[i] = r_new
                    escaped = True
                    break
            if not escaped:
                break
                
    return residues
