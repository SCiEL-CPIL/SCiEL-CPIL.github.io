"""Distinct three-term unit-fraction representations of 4/n."""
import time
import math

def construct():
    deadline = time.time() + 80.0
    answers = []
    BOUND = 10**12

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        x_min = n // 4 + 1
        # Tighter upper bound: 3/x >= 4/n => x <= 3n/4
        x_max = min((3 * n) // 4 + 2, 4 * n)

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

            k_lo = (b + a) // a
            if k_lo * a <= b:
                k_lo += 1
            k_hi = (2 * b) // a
            if k_hi < k_lo:
                continue

            # Cap iterations for this x value
            max_k_iters = 150000
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
                if y < 1 or z < y:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                if x != y and y != z and x != z:
                    found = [x, y, z]
                    break

            if found:
                break

        # If not found, try broader x candidates
        if found is None and time.time() < deadline - 3:
            for x_candidate in [n, 2*n, 3*n, 4*n, 5*n, 6*n, 8*n, 10*n, 12*n, 16*n]:
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
                    if (v + b) % a != 0:
                        continue
                    y = (u + b) // a
                    z = (v + b) // a
                    if y < 1 or z < y:
                        continue
                    if y > BOUND or z > BOUND:
                        continue
                    if x_candidate != y and y != z and x_candidate != z:
                        found = [x_candidate, y, z]
                        break
                if found:
                    break

        answers.append(found)

    return answers
