"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct distinct three-term unit-fraction representations of 4/n for n=3..256.
    Uses algebraic parameterization and systematic search with deadline guard.
    """
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    LIMIT = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        
        # Strategy 1: Algebraic identity approach
        # 1/x + 1/y + 1/z = 4/n
        # Let x = n + k, then 1/y + 1/z = 4/n - 1/(n+k) = (4(n+k) - n)/(n(n+k)) = (3n + 4k)/(n(n+k))
        # So 1/y + 1/z = (3n + 4k)/(n(n+k))
        # Let y = n(n+k)/a, z = n(n+k)/b where a + b = 3n + 4k and a*b divides (n(n+k))^2
        # More simply: 1/y + 1/z = A/B means y = B/a, z = B/b where a+b=A and a|B, b|B
        
        # Try small values of k (x = n + k, with x > n so k >= 1)
        # Actually x should be such that 1/x < 4/n, so x > n/4.
        # Let's use x = n + k for k >= 1, ensuring x > n > n/4.
        
        # Alternative: use the standard substitution
        # 1/x + 1/y + 1/z = 4/n
        # Multiply by xyz: yz + xz + xy = 4xyz/n
        # n(yz + xz + xy) = 4xyz
        # This is a Markov-type equation.
        
        # Let's try a more systematic approach:
        # For a given n, try x from floor(n/4)+1 upward.
        # 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let A = 4x - n, B = nx
        # 1/y + 1/z = A/B
        # y + z = Ayz/B => B(y+z) = Ayz => Ayz - By - Bz = 0
        # (Ay - B)(Az - B) = B^2
        # Let u = Ay - B, v = Az - B, then uv = B^2
        # y = (u + B)/A, z = (v + B)/A
        # We need u, v to be positive divisors of B^2 with u <= v (for y <= z)
        # and u + B ≡ 0 (mod A), v + B ≡ 0 (mod A)
        # Also y, z must be distinct from x and from each other.
        
        # For efficiency, iterate over possible x values
        # x must satisfy: 4/n - 1/x > 0 => x > n/4
        # And we need 1/y + 1/z = (4x-n)/(nx) to be representable with y, z > x (or at least distinct)
        
        # Let's bound x: since y, z >= x+1 (if we assume x < y < z),
        # 4/n - 1/x = 1/y + 1/z <= 2/(x+1)
        # So (4x - n)/(nx) <= 2/(x+1)
        # (4x - n)(x+1) <= 2nx
        # 4x^2 + 4x - nx - n <= 2nx
        # 4x^2 + 4x - 3nx - n <= 0
        # This gives an upper bound on x.
        
        # Let's try x from floor(n/4) + 1 up to some bound
        # For each x, compute A = 4x - n, B = nx
        # We need A > 0, so x > n/4
        
        # Find divisors of B^2 = n^2 * x^2
        # But B^2 can be huge. Instead, iterate over divisors of B and check.
        
        # Actually, let's use the factorization approach more carefully.
        # (Ay - B)(Az - B) = B^2
        # Let d be a divisor of B^2. Set u = d, v = B^2/d.
        # y = (d + B)/A, z = (B^2/d + B)/A
        # We need A | (d + B) and A | (B^2/d + B)
        
        # To avoid iterating over all divisors of B^2 (which can be enormous),
        # note that d must be such that d ≡ -B (mod A) and B^2/d ≡ -B (mod A).
        # From d ≡ -B (mod A), write d = A*k - B for some k.
        # Then B^2/d = B^2/(Ak - B)
        # And we need B^2/(Ak - B) ≡ -B (mod A)
        # B^2/(Ak - B) ≡ -B (mod A)
        # Multiply both sides by (Ak - B): B^2 ≡ -B(Ak - B) (mod A)
        # B^2 ≡ -ABk + B^2 (mod A)
        # 0 ≡ -ABk (mod A)
        # This is always true! So the condition is just that A | (d + B) and that d divides B^2.
        
        # So we need d | B^2 and d ≡ -B (mod A), i.e., d ≡ B(A-1) (mod A) if we want positive... 
        # Actually d ≡ -B (mod A). Let's write d = A*m - B for integer m >= 1.
        # We need d > 0 so m > B/A.
        # And d | B^2, i.e., (Am - B) | B^2.
        
        # Let's just iterate m and check divisibility.
        # d = Am - B, we need (Am - B) | B^2
        # i.e., B^2 % (Am - B) == 0
        
        # What range of m? d = Am - B must be a divisor of B^2, so d <= B^2.
        # Am - B <= B^2 => m <= (B^2 + B)/A
        # Also d > 0 => m > B/A => m >= floor(B/A) + 1
        # And we want y <= z, so d <= B^2/d => d^2 <= B^2 => d <= B => Am - B <= B => m <= 2B/A
        # So m ranges from floor(B/A) + 1 to floor(2B/A)
        
        # This is a bounded range! Let's use this.
        
        solved = False
        x_start = n // 4 + 1
        # Upper bound for x: from 4x^2 + 4x - 3nx - n <= 0
        # x_max approx: solve 4x^2 - 3nx = 0 => x = 3n/4
        # Let's use x up to 3*n//4 + 2 to be safe, but also check the quadratic
        # 4x^2 + (4 - 3n)x - n <= 0
        # Positive root: x = [-(4-3n) + sqrt((4-3n)^2 + 16n)] / 8
        # = [(3n-4) + sqrt(9n^2 - 24n + 16 + 16n)] / 8
        # = [(3n-4) + sqrt(9n^2 - 8n + 16)] / 8
        
        import math
        disc = 9*n*n - 8*n + 16
        if disc >= 0:
            x_max = int(math.ceil((3*n - 4 + math.isqrt(disc)) / 8)) + 2
        else:
            x_max = 3 * n // 4 + 2
        
        x_max = min(x_max, 3 * n // 4 + 5)  # Safety cap
        
        for x in range(x_start, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # m ranges from floor(B/A) + 1 to floor(2B/A)
            m_lo = B // A + 1
            m_hi = (2 * B) // A
            
            if m_hi < m_lo:
                continue
            
            # Iterate m in this range
            for m in range(m_lo, m_hi + 1):
                d = A * m - B
                if d <= 0:
                    continue
                if B * B % d != 0:
                    continue
                y = (d + B) // A
                # Check that d + B is divisible by A (should be by construction)
                if (d + B) % A != 0:
                    continue
                v = B * B // d
                z = (v + B) // A
                if (v + B) % A != 0:
                    continue
                
                # Check distinctness and bounds
                if x == y or x == z or y == z:
                    continue
                if y < 1 or z < 1:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                if x > LIMIT:
                    continue
                
                # Verify the equation
                # 1/x + 1/y + 1/z == 4/n
                # Check by cross multiplication: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    solved = True
                    break
            
            if solved:
                break
        
        answers[idx] = found
    
    return answers
# EVOLVE-BLOCK-END
