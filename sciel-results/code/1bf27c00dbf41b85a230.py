# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Generate all 2,3,5-smooth numbers <= 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)

    # Precompute divisibility: divs[a] = set of b in smooth such that a divides b
    # For antichain check on a set S: no element divides another.
    # For small sets (<=30), we can just check pairwise.

    def is_antichain(terms):
        # terms is a tuple/list of ints; check no element divides another
        m = len(terms)
        for i in range(m):
            a = terms[i]
            for j in range(i + 1, m):
                b = terms[j]
                if b % a == 0 or a % b == 0:
                    return False
        return True

    answers = [None] * 128

    # For each target, try sizes 1, 2, 3, ... up to 30
    # For size 1: trivial if n is smooth
    # For size 2: a + b = n, a,b smooth, a != b, a not divide b and b not divide a
    # For size 3: a+b+c = n, all distinct, antichain

    # Precompute for speed: for each n, find min-size antichain
    for n in range(1, 129):
        if time.time() > DEADLINE:
            break
        found = None

        # Size 1
        if n in smooth:
            found = [n]
        else:
            # Size 2: iterate over smooth a < n, b = n - a, check b smooth, a != b, antichain
            found = None
            for a in smooth:
                if a >= n:
                    break
                b = n - a
                if b <= 0:
                    continue
                if b in smooth and a != b:
                    # Check antichain: a not divide b and b not divide a
                    if b % a != 0 and a % b != 0:
                        found = [a, b]
                        break
            if found is None:
                # Size 3: iterate over pairs (a,b) from smooth with a < b < n
                # then c = n - a - b, check c smooth, distinct, antichain
                # To keep it fast, limit search
                s = smooth
                L = len(s)
                found = None
                for i in range(L):
                    a = s[i]
                    if a >= n:
                        break
                    for j in range(i + 1, L):
                        b = s[j]
                        if b >= n - a:
                            break
                        c = n - a - b
                        if c <= 0:
                            break
                        if c in smooth and c != a and c != b:
                            # Check antichain on {a,b,c}
                            if (b % a != 0 and a % b != 0 and
                                c % a != 0 and a % c != 0 and
                                c % b != 0 and b % c != 0):
                                found = [a, b, c]
                                break
                    if found is not None:
                        break
            if found is None:
                # Fallback: larger sizes — use a greedy/backtracking search
                # For now, try sizes 4..30 with a bounded search
                found = None
                for size in range(4, 31):
                    if time.time() > DEADLINE:
                        break
                    # Simple DFS with pruning
                    best = None
                    # Use combinations from smooth numbers <= n
                    candidates = [x for x in smooth if x < n]
                    # Try to find any antichain of given size summing to n
                    # This is expensive; use a heuristic: sort and try
                    # For safety, just do a limited search
                    # Given that sizes 1-3 should cover all 1..128, this is a fallback
                    break
            if found is None:
                # Emergency: allow any valid representation via full search
                # This should not happen for 1..128
                found = [n] if n in smooth else None

        answers[n - 1] = found

    # Fill any None with a fallback (should not happen)
    for i in range(128):
        if answers[i] is None:
            n = i + 1
            # Brute force with all sizes
            for size in range(1, 31):
                from itertools import combinations
                for terms in combinations([x for x in smooth if x <= n], size):
                    if sum(terms) == n and is_antichain(terms):
                        answers[i] = list(terms)
                        break
                if answers[i] is not None:
                    break

    return answers
# EVOLVE-BLOCK-END
