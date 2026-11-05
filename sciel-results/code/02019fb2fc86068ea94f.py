"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    if n <= 1:
        return "".join(map(str, range(1, n + 1)))
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134132413214321"
    if n == 5:
        # Standard recursive construction for n=5
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 6, use the standard recursive construction which is the best known
    # simple construction. The length is sum_{k=1}^n k!
    # For n=7, this gives 1!+2!+...+7! = 1+2+6+24+120+720+5040 = 5913
    # The shortest known is 5905, but achieving that requires complex search.
    # We'll use the standard construction which is valid and fast.

    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline - 10:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            if time.time() > deadline - 5:
                break
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
