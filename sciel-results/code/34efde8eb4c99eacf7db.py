"""Residue-based conflict model with accelerated local search for max non-dividing triple-sum set."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin before 90s kill

    n = 192

    # Precompute conflict structure using residues.
    # For each a in 1..n, and for each pair (b, c) with b < c, b != a, c != a:
    #   a conflicts with the triple {b, c, d} iff a | (b + c + d), i.e., d ≡ -(b+c) (mod a)
    # So when checking if we can add x to set S:
    #   For each a in S: check if there exist b, c in S\{a} such that (b + c + x) % a == 0
    #   For x: check if there exist b, c, d in S such that (b + c + d) % x == 0

    # Optimization: precompute for each a, a mapping from residue r to the list of pairs (b,c) 
    # where b < c, b != a, c != a, and (b+c) % a == r.
    # Then when adding x, for each a in S, we need residue r = (-x) % a, and check if any pair 
    # from that residue list has both elements in S\{a}.

    # For the "x divides triple sum" check: for each x, precompute pairs (b,c) with b < c,
    # b != x, c != x, grouped by (b+c) % x. Then check if any triple from S sums to 0 mod x.
    # This is still O(|S|^3) in worst case, but we can optimize by checking pairs first.

    # Let's precompute pair residues for all a.
    # pair_res[a][r] = list of (b, c) with b < c, b != a, c != a, (b+c) % a == r
    pair_res = [dict() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if b == a:
                continue
            for c in range(b + 1, n + 1):
                if c == a:
                    continue
                r = (b + c) % a
                if r not in pair_res[a]:
                    pair_res[a][r] = []
                pair_res[a][r].append((b, c))

    # For the "x divides triple sum" check, we can precompute for each x:
    # triple_res[x][r] = list of triples (b,c,d) with b < c < d, none equal to x, (b+c+d) % x == r
    # But this is too large. Instead, we'll do a smarter check.
    # For a given x and set S, we need to check if any triple from S sums to 0 mod x.
    # We can do this by: for each pair (b,c) in S with b < c, compute needed = (-b-c) % x,
    # and check if there exists d in S, d > c, d != x, d != b, d != c, such that d % x == needed.
    # Actually d just needs to satisfy (b+c+d) % x == 0, so d ≡ -(b+c) (mod x).
    # We can precompute for each x and each residue r, which numbers in 1..n have that residue.
    # Then for a given set S, we can check efficiently.

    # Precompute: num_by_res[x][r] = sorted list of numbers v in 1..n, v != x, v % x == r
    num_by_res = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for r in range(x):
            nums = []
            for v in range(1, n + 1):
                if v != x and v % x == r:
                    nums.append(v)
            num_by_res[x][r] = nums

    def can_add(x, S_set, S_list):
        """Check if x can be added to S (represented as both a set and a list for speed)."""
        # Check 1: for each a in S, does a divide (b+c+x) for some b,c in S\{a}?
        for a in S_list:
            if a == x:
                continue
            r = (-x) % a
            if r in pair_res[a]:
                for (b, c) in pair_res[a][r]:
                    if b in S_set and c in S_set and b != a and c != a:
                        return False
        # Check 2: does x divide (b+c+d) for some triple b,c,d in S?
        if len(S_list) >= 3:
            # For each pair (b,c) in S, check if there's a d in S such that (b+c+d) % x == 0
            # d ≡ -(b+c) (mod x)
            for i in range(len(S_list)):
                b = S_list[i]
                for j in range(i + 1, len(S_list)):
                    c = S_list[j]
                    r = (-(b + c)) % x
                    if r in num_by_res[x]:
                        for d in num_by_res[x][r]:
                            if d in S_set and d != b and d != c and d != x:
                                return False
        return True

    # Greedy: large first
    def greedy(order):
        S = []
        S_set = set()
        for x in order:
            if can_add(x, S_set, S):
                S.append(x)
                S_set.add(x)
        return S

    best_solution = []
    best_size = 0

    # Multiple greedy passes with different orderings
    rng = random.Random(12345)

    # Pass 1: descending
    sol = greedy(range(n, 0, -1))
    best_size = len(sol)
    best_solution = sol[:]

    # Pass 2: ascending
    sol = greedy(range(1, n + 1))
    if len(sol) > best_size:
        best_size = len(sol)
        best_solution = sol[:]

    # Pass 3: biased random (favor larger)
    for trial in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        order.sort(key=lambda x: (-x, rng.random()))
        sol = greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol[:]

    # Local search: try to improve by removing one element and adding one or more others
    S = best_solution[:]
    S_set = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element and see if we can add back a different one
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            a = S[idx]
            # Temporarily remove a
            S_set.remove(a)
            S_temp = S[:idx] + S[idx + 1:]
            # Try adding each candidate not in S
            added = False
            for x in range(n, 0, -1):
                if x in S_set:
                    continue
                if can_add(x, S_set, S_temp):
                    S_temp.append(x)
                    S_set.add(x)
                    # Now try to add more
                    more_added = True
                    while more_added and time.time() < deadline:
                        more_added = False
                        for y in range(n, 0, -1):
                            if y in S_set:
                                continue
                            if can_add(y, S_set, S_temp):
                                S_temp.append(y)
                                S_set.add(y)
                                more_added = True
                    if len(S_temp) > len(S):
                        S = S_temp
                        S_set = set(S)
                        best_size = len(S)
                        best_solution = S[:]
                        improved = True
                    break
            if not added:
                S_set.add(a)  # restore

    return best_solution

# EVOLVE-BLOCK-END
