"""Precomputed lookup for distinct three-term unit-fraction reps of 4/n."""
# EVOLVE-BLOCK-START
import time

def _build_lookup(max_denom=200000):
    """Precompute all triples x<y<z with z<=max_denom where 1/x+1/y+1/z=4/n for some n."""
    lookup = {}  # n -> [x, y, z]
    # 1/x + 1/y + 1/z = 4/n => n = 4xyz/(xy+yz+zx)
    # So n must be integer, meaning (xy+yz+zx) | 4xyz
    # For small x,y,z, n is small
    # WLOG x < y < z, x >= 2 (since 1/x < 4/n and n>=3, x > 3/4, so x>=1 but x must be < n/4*... actually x > n/4 >= 3/4 so x>=1)
    # But x,y,z must be distinct positive integers
    # 1/x + 1/y + 1/z <= 4/3 (max when x=1,y=2,z=3: 1+1/2+1/3=11/6>4/3)
    # We need 1/x+1/y+1/z = 4/n where n>=3, so 1/x+1/y+1/z <= 4/3
    # x >= 1: if x=1, 1/y+1/z = 4/n-1, need 4/n>1 => n<4, so n=3: 1/y+1/z=1/3
    #   y>1, 1/y+1/z=1/3 => (y-3)(z-3)=9, so (y,z)=(4,12),(6,6) but distinct, so (4,12)
    #   Check: 1+1/4+1/12=1+3/12+1/12=1+4/12=1+1/3=4/3=4/3 ✓
    # So n=3: [1,4,12]
    
    for x in range(1, max_denom + 1):
        if x > 3 * max_denom:  # x < 3n/4, n <= max_denom*4/3 roughly
            break
        for y in range(x + 1, max_denom + 1):
            # 1/z = 4/n - 1/x - 1/y, but we don't know n
            # n = 4xyz/(xy+yz+zx) must be integer
            # Let's compute numerator and denominator
            # n = 4xyz / (xy + yz + zx)
            # For n to be integer, (xy+yz+zx) | 4xyz
            # Also n >= 3
            # 4xyz/(xy+yz+zx) >= 3 => 4xyz >= 3xy + 3yz + 3zx
            # This is a constraint on x,y,z
            # Also n <= 256 (we only need n up to 256)
            # 4xyz/(xy+yz+zx) <= 256 => 4xyz <= 256(xy+yz+zx)
            
            # For efficiency, bound z: 1/z < 4/n - 1/x - 1/y
            # Since n >= 3: 1/z < 4/3 - 1/x - 1/y
            # If 4/3 - 1/x - 1/y <= 0, no valid z
            # z > 1/(4/3 - 1/x - 1/y)
            
            # Also from n <= 256: 1/z > 4/256 - 1/x - 1/y = 1/64 - 1/x - 1/y
            # If 1/64 - 1/x - 1/y >= 0, then z can be anything (but z > y)
            
            # Let's just iterate z and check
            # 1/x + 1/y + 1/z = 4/n => n = 4xyz/(xy+yz+zx)
            # For fixed x,y: n(z) = 4xyz/(xy + z(x+y))
            # As z increases, n(z) approaches 4xy/(x+y)
            
            # n must be integer in [3, 256]
            # 4xyz = n(xy + yz + zx)
            # 4xyz - nyz - nzx = nxy
            # z(4xy - ny - nx) = nxy
            # z = nxy / (4xy - ny - nx)
            # For fixed x,y,n: z = nxy / (4xy - n(x+y))
            # Denominator must be positive: 4xy > n(x+y)
            # And z must be integer > y
            
            # Iterate n from 3 to 256
            for n_val in range(3, 257):
                denom = 4 * x * y - n_val * (x + y)
                if denom <= 0:
                    break  # as n increases, denom decreases
                num = n_val * x * y
                if num % denom != 0:
                    continue
                z = num // denom
                if z <= y:
                    continue
                if z > max_denom:
                    continue
                # Verify
                if n_val * (y * z + x * z + x * y) == 4 * x * y * z:
                    if n_val not in lookup:
                        lookup[n_val] = [x, y, z]
    
    return lookup

def construct():
    deadline = time.time() + 85
    # Build lookup with a reasonable bound
    # The key insight: for n up to 256, the denominators x,y,z don't need to be huge
    # Let's try max_denom = 1000000 first, but that's 10^6 * 10^6 * 254 which is too slow
    # Need a smarter approach
    
    # Actually, let's just use the direct search but with better bounds
    BOUND = 10**12
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        found = False
        for x in range(x_min, x_max + 1):
            if found or time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            y_min = max(x + 1, B // A + 1)
            y_max = (2 * B) // A + 1
            if y_max < y_min:
                continue
            
            # Limit the range of y to keep it manageable
            # Try at most 10000 y values per x
            y_iter_max = min(y_max, y_min + 10000)
            
            for y in range(y_min, y_iter_max + 1):
                if found or time.time() > deadline:
                    break
                d = A * y - B
                if d <= 0 or d >= B:
                    continue
                if (B * B) % d != 0:
                    continue
                num_z = B * B // d + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                if z <= y or z > BOUND:
                    continue
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    results[idx] = [x, y, z]
                    found = True
                    break
    
    return results
# EVOLVE-BLOCK-END
