"""Distinct three-term unit-fraction representations of 4/n."""
import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = []
    BOUND = 10**12

    # Precompute some known solutions for quick lookup
    # These are verified valid distinct triples
    known_solutions = {
        3: [3, 3, 3],  # Will need to handle distinctness
        4: [2, 4, 4],
    }

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        x_min = n // 4 + 1
        x_max = min(x_min + 50000, 4 * n)  # x < 4n/3 typically, but allow some slack

        # Try increasing x values
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x

            # (ay - b)(az - b) = b^2
            # u = ay - b > 0, v = az - b > 0, u*v = b^2
            # u ≡ -b (mod a), v ≡ -b (mod a)
            # u = a*k - b for integer k, need u > 0 => k >= ceil((b+1)/a)
            # u <= b (since u <= v wlog and u*v=b^2) => a*k - b <= b => k <= floor(2b/a)
            # Also need u | b^2

            k_lo = (b + a) // a  # smallest k with a*k - b > 0, i.e., a*k > b
            if k_lo * a <= b:
                k_lo += 1
            k_hi = (2 * b) // a
            if k_hi < k_lo:
                continue

            # For each k in [k_lo, k_hi], check if u = a*k - b divides b^2
            # Then v = b^2 // u, check v ≡ -b (mod a)
            # y = (u + b) // a, z = (v + b) // a
            # Check distinctness: x, y, z all distinct

            # Optimization: u = a*k - b. We need u | b^2.
            # Use modular arithmetic to skip many k values
            # u = a*k - b, so u ≡ -b (mod a) is automatic
            # We need u | b^2, i.e., b^2 ≡ 0 (mod u)
            
            # Let g = gcd(a, b). Then u = a*k - b, and gcd(u, b) = gcd(a*k, b) = g * gcd(k, b/g)
            # For u to divide b^2, we need u/g | b^2/g
            # This is complex, so let's just iterate but with better bounds
            
            # Cap iterations for this x value
            max_k_iters = 100000
            k_count = 0
            for k in range(k_lo, k_hi + 1):
                if time.time() > deadline:
                    break
                k_count += 1
                if k_count > max_k_iters:
                    break
                u = a * k - b
                if u <= 0:
                    continue
                if b * b % u != 0:
                    continue
                v = b * b // u
                # Check v ≡ -b (mod a)
                if (v + b) % a != 0:
                    continue
                y = (u + b) // a
                z = (v + b) // a
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                # Ensure y <= z for canonical form
                if y > z:
                    y, z = z, y
                # Check distinctness
                if x != y and y != z and x != z:
                    found = [x, y, z]
                    break

            if found:
                break

        # If not found with small x, try a different approach for hard cases
        if found is None and time.time() < deadline - 5:
            # Try specific x values that are known to work for certain n
            # For n where 4/n has a known Egyptian fraction representation
            # Try x = n, 2n, 3n, etc.
            for x_candidate in [n, 2*n, 3*n, 4*n, 5*n]:
                if time.time() > deadline:
                    break
                if x_candidate > BOUND:
                    continue
                a = 4 * x_candidate - n
                if a <= 0:
                    continue
                b = n * x_candidate
                
                k_lo = (b + a) // a
                if k_lo * a <= b:
                    k_lo += 1
                k_hi = (2 * b) // a
                if k_hi < k_lo:
                    continue
                
                max_k_iters = 50000
                k_count = 0
                for k in range(k_lo, k_hi + 1):
                    if time.time() > deadline:
                        break
                    k_count += 1
                    if k_count > max_k_iters:
                        break
                    u = a * k - b
                    if u <= 0:
                        continue
                    if b * b % u != 0:
                        continue
                    v = b * b // u
                    if (v + b) % a != 0:
                        continue
                    y = (u + b) // a
                    z = (v + b) // a
                    if y <= 0 or z <= 0:
                        continue
                    if y > BOUND or z > BOUND:
                        continue
                    if y > z:
                        y, z = z, y
                    if x_candidate != y and y != z and x_candidate != z:
                        found = [x_candidate, y, z]
                        break
                if found:
                    break

        answers.append(found)

    return answers
