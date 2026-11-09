import time
import math

def construct():
    answers = []
    deadline = time.time() + 85
    BOUND = 10**12

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break

            a = 4 * x - n
            b = n * x

            if a <= 0:
                continue

            # (ay - b)(az - b) = b^2
            # Let u = ay - b, v = az - b, then u*v = b^2
            # y = (u+b)/a, z = (v+b)/a
            # Need: u > 0, v > 0, u < v (for y < z), a | (u+b), a | (v+b)
            # y > x => (u+b)/a > x => u > a*x - b
            # z > y => v > u
            # Also need y, z distinct from x and each other

            b_sq = b * b
            sqrt_b_sq = b  # since sqrt(b^2) = b

            u_min = a * x - b + 1
            if u_min < 1:
                u_min = 1

            if u_min > sqrt_b_sq:
                continue

            # Iterate u from u_min to sqrt_b_sq, checking if u divides b_sq
            for u in range(u_min, sqrt_b_sq + 1):
                if time.time() > deadline:
                    break

                if b_sq % u != 0:
                    continue

                v = b_sq // u
                # v >= u since u <= sqrt(b_sq)
                if v == u:
                    continue  # would give y == z

                # Check divisibility: a | (u+b) and a | (v+b)
                if (u + b) % a != 0:
                    continue
                if (v + b) % a != 0:
                    continue

                y = (u + b) // a
                z = (v + b) // a

                # Ordering and distinctness checks
                if y <= x or z <= y:
                    continue
                if x == y or y == z or x == z:
                    continue
                if y > BOUND or z > BOUND:
                    continue

                # Verify via cross multiplication
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break

        answers.append(found)

    return answers
