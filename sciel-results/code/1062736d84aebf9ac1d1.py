import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def is_smooth(n):
        while n % 2 == 0:
            n //= 2
        while n % 3 == 0:
            n //= 3
        while n % 5 == 0:
            n //= 5
        return n == 1

    # Generate all 2-3-5 smooth numbers up to 128
    smooth_numbers = [i for i in range(1, 129) if is_smooth(i)]
    smooth_set = set(smooth_numbers)

    def is_antichain(terms):
        n = len(terms)
        for i in range(n):
            for j in range(i + 1, n):
                if terms[i] % terms[j] == 0 or terms[j] % terms[i] == 0:
                    return False
        return True

    answers = [None] * 128
    found_count = [0] * 129

    # Try 1-term: only valid if target itself is smooth
    for t in range(1, 129):
        if t in smooth_set:
            answers[t - 1] = [t]
            found_count[t] = 1

    if time.time() > deadline:
        return answers

    # Try 2-term representations
    n_smooth = len(smooth_numbers)
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s = a + b
            if s > 128:
                break
            if found_count[s] > 2:
                if a % b != 0 and b % a != 0:
                    answers[s - 1] = [a, b]
                    found_count[s] = 2

    if time.time() > deadline:
        return answers

    # Try 3-term representations
    for i in range(n_smooth):
        a = smooth_numbers[i]
        for j in range(i + 1, n_smooth):
            b = smooth_numbers[j]
            s2 = a + b
            if s2 > 127:
                break
            for k in range(j + 1, n_smooth):
                c = smooth_numbers[k]
                s = s2 + c
                if s > 128:
                    break
                if found_count[s] > 3:
                    if (a % b != 0 and b % a != 0 and
                        a % c != 0 and c % a != 0 and
                        b % c != 0 and c % b != 0):
                        answers[s - 1] = [a, b, c]
                        found_count[s] = 3

    if time.time() > deadline:
        return answers

    # Try 4-term representations for remaining
    missing = [t for t in range(1, 129) if found_count[t] == 0]
    if missing:
        for i in range(n_smooth):
            a = smooth_numbers[i]
            for j in range(i + 1, n_smooth):
                b = smooth_numbers[j]
                s2 = a + b
                if s2 > 126:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth_numbers[k]
                    s3 = s2 + c
                    if s3 > 125:
                        break
                    for l in range(k + 1, n_smooth):
                        d = smooth_numbers[l]
                        s = s3 + d
                        if s > 128:
                            break
                        if found_count[s] > 4:
                            terms = (a, b, c, d)
                            ac = True
                            for x in range(4):
                                for y in range(x + 1, 4):
                                    if terms[x] % terms[y] == 0 or terms[y] % terms[x] == 0:
                                        ac = False
                                        break
                                if not ac:
                                    break
                            if ac:
                                answers[s - 1] = list(terms)
                                found_count[s] = 4
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    # Fallback for any uncovered targets
    for t in range(1, 129):
        if found_count[t] == 0:
            # Try pairs systematically
            found = False
            for a in smooth_numbers:
                if a >= t:
                    break
                b = t - a
                if b > a and b in smooth_set and a % b != 0 and b % a != 0:
                    answers[t - 1] = [a, b]
                    found_count[t] = 2
                    found = True
                    break
            if not found:
                # Try triples
                for i in range(len(smooth_numbers)):
                    a = smooth_numbers[i]
                    if a >= t:
                        break
                    for j in range(i + 1, len(smooth_numbers)):
                        b = smooth_numbers[j]
                        if a + b >= t:
                            break
                        c = t - a - b
                        if c > b and c in smooth_set:
                            if (a % b != 0 and b % a != 0 and
                                a % c != 0 and c % a != 0 and
                                b % c != 0 and c % b != 0):
                                answers[t - 1] = [a, b, c]
                                found_count[t] = 3
                                found = True
                                break
                    if found:
                        break
            if not found:
                # Try 4 terms
                for i in range(len(smooth_numbers)):
                    a = smooth_numbers[i]
                    if a >= t:
                        break
                    for j in range(i + 1, len(smooth_numbers)):
                        b = smooth_numbers[j]
                        if a + b >= t:
                            break
                        for k in range(j + 1, len(smooth_numbers)):
                            c = smooth_numbers[k]
                            if a + b + c >= t:
                                break
                            d = t - a - b - c
                            if d > c and d in smooth_set:
                                terms = (a, b, c, d)
                                ac = True
                                for x in range(4):
                                    for y in range(x + 1, 4):
                                        if terms[x] % terms[y] == 0 or terms[y] % terms[x] == 0:
                                            ac = False
                                            break
                                    if not ac:
                                        break
                                if ac:
                                    answers[t - 1] = list(terms)
                                    found_count[t] = 4
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    # Last resort: try 5 terms
                    for i in range(len(smooth_numbers)):
                        a = smooth_numbers[i]
                        if a >= t:
                            break
                        for j in range(i + 1, len(smooth_numbers)):
                            b = smooth_numbers[j]
                            if a + b >= t:
                                break
                            for k in range(j + 1, len(smooth_numbers)):
                                c = smooth_numbers[k]
                                if a + b + c >= t:
                                    break
                                for l in range(k + 1, len(smooth_numbers)):
                                    d = smooth_numbers[l]
                                    if a + b + c + d >= t:
                                        break
                                    e = t - a - b - c - d
                                    if e > d and e in smooth_set:
                                        terms = (a, b, c, d, e)
                                        ac = True
                                        for x in range(5):
                                            for y in range(x + 1, 5):
                                                if terms[x] % terms[y] == 0 or terms[y] % terms[x] == 0:
                                                    ac = False
                                                    break
                                            if not ac:
                                                break
                                        if ac:
                                            answers[t - 1] = list(terms)
                                            found_count[t] = 5
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        # Absolute last resort
                        answers[t - 1] = [t]
                        found_count[t] = 1

    return answers
