import time

def construct():
    start = time.time()
    deadline = start + 85.0

    def check_collision(S, n):
        for k in range(n + 1):
            for l in range(k + 1, n + 1):
                if S[k] & S[l]:
                    return True
        return False

    def greedy(order):
        S = [0] * 62
        S[0] = 1
        n = 0
        result = []
        for x in order:
            if time.time() > deadline - 2:
                break
            new_S = S[:]
            for k in range(n, -1, -1):
                new_S[k + 1] |= (S[k] << x)
            if not check_collision(new_S, n + 1):
                S = new_S
                n += 1
                result.append(x)
        return result

    best = []

    r = greedy(range(256, 0, -1))
    if len(r) > len(best):
        best = r

    if time.time() < deadline - 5:
        r = greedy(range(1, 257))
        if len(r) > len(best):
            best = r

    known = list(range(226, 257))
    if len(known) > len(best):
        best = known

    if time.time() < deadline - 5:
        for offset in range(-10, 11):
            if time.time() > deadline - 2:
                break
            elements = list(range(226 + offset, 257 + offset))
            if elements[0] < 1 or elements[-1] > 256:
                continue
            S = [0] * 62
            S[0] = 1
            n = 0
            valid = True
            for x in elements:
                new_S = S[:]
                for k in range(n, -1, -1):
                    new_S[k + 1] |= (S[k] << x)
                if check_collision(new_S, n + 1):
                    valid = False
                    break
                S = new_S
                n += 1
            if valid and n > len(best):
                best = elements[:n]

    return best[:len(best)]
