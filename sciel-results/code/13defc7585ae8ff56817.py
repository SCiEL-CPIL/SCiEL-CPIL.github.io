"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    answers = []
    BOUND = 10**12
    for n in range(3, 257):
        found = None
        # We want 1/x + 1/y + 1/z = 4/n with x < y < z, all distinct positive integers.
        # Rearranging: 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # So (y - nx/(4x-n)) ... let's use the standard Simon's Favorite Factoring Trick.
        # Let A = 4x - n. Then 1/y + 1/z = A/(nx).
        # => nx(1/y + 1/z) = A
        # => nx(z+y)/(yz) = A
        # => nx(z+y) = A y z
        # => A y z - nx y - nx z = 0
        # Multiply by A: A^2 y z - A nx y - A nx z = 0
        # Add (nx)^2: A^2 y z - A nx y - A nx z + (nx)^2 = (nx)^2
        # => (A y - nx)(A z - nx) = (nx)^2
        # Let u = A y - nx, v = A z - nx. Then u v = (nx)^2.
        # Also y = (u + nx)/A, z = (v + nx)/A.
        # We need y, z to be integers, so A must divide (u+nx) and (v+nx).
        # Since u v = (nx)^2, let u be a divisor of (nx)^2. Then v = (nx)^2 / u.
        # We need A | (u + nx) and A | (v + nx).
        # Note that u + nx ≡ nx (mod A) is not necessarily 0.
        # Actually, u v = (nx)^2.
        # u ≡ -nx (mod A) is required for y to be integer?
        # y = (u+nx)/A. So u+nx must be divisible by A.
        # Similarly v+nx must be divisible by A.
        #
        # Alternative approach: iterate x, then for each x, find y, z.
        # For a fixed x, we need (4x-n)(4y-n)(4z-n) ... no, let's stick to the simpler one.
        # For fixed x, let k = 4x - n. We need k > 0, so x > n/4.
        # We need 1/y + 1/z = k / (nx).
        # This is equivalent to (k y - nx)(k z - nx) = (nx)^2.
        # Let D = nx. We need (k y - D)(k z - D) = D^2.
        # Let u = k y - D, v = k z - D. Then u v = D^2.
        # y = (u+D)/k, z = (v+D)/k.
        # Conditions:
        # 1. k > 0
        # 2. u, v are divisors of D^2 such that u v = D^2.
        # 3. (u+D) % k == 0
        # 4. (v+D) % k == 0
        # 5. y > x (to ensure distinctness and ordering, we can enforce x < y < z)
        # 6. z > y
        # 7. y, z <= BOUND
        #
        # Since we want to maximize the fraction solved, and most n should be solvable,
        # we can iterate x from floor(n/4)+1 upwards, and for each x, iterate divisors u of D^2.
        # But D^2 can be huge (n up to 256, x up to ~10^12, so D up to ~2.5*10^14, D^2 up to 10^29).
        # Factoring D^2 is too slow if done naively for every x.
        #
        # However, note that k = 4x - n. As x increases, k increases.
        # We can try a smaller range of x first.
        # Actually, for most n, a small x works.
        # Let's iterate x from floor(n/4)+1 to some limit, say n+100 or so, and for each,
        # check if there exist y, z.
        #
        # To check for y, z efficiently:
        # We need u v = D^2, u ≡ -D (mod k), v ≡ -D (mod k).
        # Since u v = D^2, v = D^2 / u.
        # So u ≡ -D (mod k) and D^2 / u ≡ -D (mod k).
        # The second condition: D^2 / u ≡ -D (mod k) => D^2 ≡ -D u (mod k) => D^2 + D u ≡ 0 (mod k) => D(D+u) ≡ 0 (mod k).
        # Since u ≡ -D (mod k), D+u ≡ 0 (mod k), so D(D+u) is always 0 mod k.
        # So the condition reduces to: find a divisor u of D^2 such that u ≡ -D (mod k) and u < D^2/u (so y < z, i.e., u < v).
        # Also y = (u+D)/k > x => u+D > k x = (4x-n)x = 4x^2 - nx.
        # u > 4x^2 - nx - D = 4x^2 - nx - nx = 4x^2 - 2nx.
        # And z = (v+D)/k <= BOUND => v+D <= k BOUND => D^2/u + D <= k BOUND => D^2/u <= k BOUND - D => u >= D^2 / (k BOUND - D).
        #
        # So for each x, we need to find a divisor u of D^2 in the range [L, U] such that u ≡ -D (mod k), where:
        # L = max(1, D^2 // (k*BOUND - D) if k*BOUND > D else 1, 4*x*x - 2*n*x)
        # U = D // 2 (since u < v implies u < D^2/u => u^2 < D^2 => u < D)
        # Wait, u < v means u < D^2/u => u^2 < D^2 => u < D.
        # So u must be in [L, min(D-1, U)] where U is determined by y > x.
        # Actually y > x => (u+D)/k > x => u > kx - D = (4x-n)x - nx = 4x^2 - nx - nx = 4x^2 - 2nx.
        # So L = max(1, 4*x*x - 2*n*x, D^2 // (k*BOUND - D) if k*BOUND > D else 1)
        # And we need u < D.
        #
        # Finding such a divisor u is still hard because D^2 is huge.
        #
        # Let's go back to the original approach but optimize it.
        # The original approach iterates y for a fixed x.
        # y must satisfy: (4x-n)y - nx = denominator > 0, and (4x-n)y - nx divides nx*y.
        # Let k = 4x - n. Then (k y - nx) | nx y.
        # Let d = k y - nx. Then y = (d+nx)/k.
        # d | nx (d+nx)/k => k d | nx(d+nx) => k d | nx^2 (since d|nx(d+nx) implies d|nx^2 if gcd(d,nx)=... wait).
        # Actually, d | nx y and y = (d+nx)/k, so d | nx(d+nx)/k.
        # This means k d | nx(d+nx).
        # Since d = ky - nx, gcd(d, nx) = gcd(ky, nx) = gcd(ky, nx).
        # This is getting complicated.
        #
        # Let's try a different identity.
        # 1/x + 1/y + 1/z = 4/n
        # Assume x = n//4 + 1. This often works.
        # If not, try x = n//4 + 2, etc.
        # For each x, we can solve for y, z using:
        # 1/y + 1/z = 4/n - 1/x = (4x-n)/(nx)
        # Let a = 4x-n, b = nx. We need 1/y + 1/z = a/b.
        # => b(y+z) = a y z
        # => a y z - b y - b z = 0
        # => (a y - b)(a z - b) = b^2
        # Let u = a y - b, v = a z - b. Then u v = b^2.
        # y = (u+b)/a, z = (v+b)/a.
        # We need u, v to be divisors of b^2 such that u v = b^2, (u+b)%a==0, (v+b)%a==0, y>x, z>y, y,z<=BOUND.
        #
        # b = nx. For n=256, x~65, b~16640. b^2 ~ 2.7*10^8. This is small enough to factor!
        # Wait, x can be larger. But we start with small x.
        # If x is small, b is small, b^2 is small, we can factor b^2 and iterate divisors.
        #
        # Let's implement this.
        # For each n:
        #   for x in range(n//4 + 1, n//4 + 1000):  # try a few x values
        #       a = 4*x - n
        #       if a <= 0: continue
        #       b = n * x
        #       b2 = b * b
        #       # Find divisors u of b2 such that u <= b (since u < v => u < b)
        #       # and (u+b) % a == 0
        #       # and y = (u+b)//a > x
        #       # and z = (b2//u + b)//a > y
        #       # and z <= BOUND
        #
        # Factoring b2: b is at most 256 * (256//4 + 1000) ~ 256 * 1064 ~ 272384. b2 ~ 7.4*10^10.
        # Factoring 7.4*10^10 is feasible with trial division up to sqrt(7.4*10^10) ~ 2.7*10^5.
        # But doing this for 1000 x values per n, and 254 n values, might be too slow.
        #
        # Let's limit x search to a smaller range, say 100.
        # And only factor b2 if necessary.
        
        for x in range(n // 4 + 1, n // 4 + 200):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b2 = b * b
            # We need u | b2, u <= b, (u+b) % a == 0, y=(u+b)//a > x, z=(b2//u+b)//a > y, z<=BOUND
            # Let's find all divisors u of b2 up to b.
            # To do this efficiently, we can iterate i from 1 to sqrt(b2) and check if i divides b2.
            # sqrt(b2) = b. So we iterate i from 1 to b.
            # b is at most 256 * (256//4 + 200) ~ 256 * 264 ~ 67584.
            # Iterating 1 to 67584 is fast.
            
            # But we need (u+b) % a == 0 => u ≡ -b (mod a).
            # So we can iterate u in steps of a? No, u must divide b2.
            # Let's just iterate i from 1 to int(sqrt(b2))+1, and for each divisor i, check i and b2//i.
            
            limit = int(math.isqrt(b2))
            for i in range(1, limit + 1):
                if b2 % i != 0:
                    continue
                # i and b2//i are divisors
                for u in (i, b2 // i):
                    if u > b:
                        continue
                    if (u + b) % a != 0:
                        continue
                    y = (u + b) // a
                    if y <= x:
                        continue
                    v = b2 // u
                    if (v + b) % a != 0:
                        continue
                    z = (v + b) // a
                    if z <= y:
                        continue
                    if z > BOUND:
                        continue
                    if y > BOUND:
                        continue
                    # Check distinctness
                    if x == y or x == z or y == z:
                        continue
                    # Verify
                    if x * y * z != 4 * n * y * z + 4 * n * x * z + 4 * n * x * y:
                        continue
                    found = [x, y, z]
                    break
                if found:
                    break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
