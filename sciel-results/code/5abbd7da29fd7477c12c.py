"""Structural change: use residue-based validity checking for faster incremental updates."""

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s safety margin

    n = 192
    max_size = 48

    # Precompute for each x in 1..n, a mapping from residue r to list of elements d in 1..n
    # such that d % x == r.
    elem_by_residue = {}
    for x in range(1, n + 1):
        elem_by_residue[x] = {}
        for d in range(1, n + 1):
            r = d % x
            if r not in elem_by_residue[x]:
                elem_by_residue[x][r] = []
            elem_by_residue[x][r].append(d)

    def is_valid_addition(S, pair_residues, x):
        """Check if adding x to S (with pair_residues) maintains validity."""
        S_set = set(S)
        
        # Check (A): for each a in S, does a divide (b+c+x) for some pair b,c in S\{a}?
        # i.e., is there a pair (b,c) in S\{a} with (b+c) ≡ -x (mod a)?
        for a in S:
            target = (-x) % a
            if pair_residues[a].get(target, 0) > 0:
                return False
        
        # Check (B): does x divide (b+c+d) for some triple b,c,d in S?
        # For each pair (b,c) in S, check if there's d in S, d!=b, d!=c, with d ≡ -(b+c) (mod x)
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            for j in range(i + 1, m):
                b = S_list[i]
                c = S_list[j]
                target = (-(b + c)) % x
                candidates = elem_by_residue[x].get(target, [])
                for d in candidates:
                    if d != b and d != c and d in S_set:
                        return False
        
        return True

    def build_pair_residues(S):
        """Build pair_residues from scratch for set S."""
        pair_residues = {}
        S_set = set(S)
        for a in S:
            pair_residues[a] = {}
            others = [v for v in S if v != a]
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    b = others[i]
                    c = others[j]
                    r = (b + c) % a
                    if r not in pair_residues[a]:
                        pair_residues[a][r] = 0
                    pair_residues[a][r] += 1
        return pair_residues

    def greedy(order):
        """Greedy: add elements in the given order if valid."""
        S = []
        pair_residues = {}
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(S, pair_residues, x):
                S.append(x)
                # Update pair_residues incrementally
                # For each existing a in S[:-1]: add pairs involving x
                for a in S[:-1]:
                    for b in S[:-1]:
                        if b == a:
                            continue
                        p, q = (b, x) if b < x else (x, b)
                        r = (p + q) % a
                        if r not in pair_residues[a]:
                            pair_residues[a][r] = 0
                        pair_residues[a][r] += 1
                # Add pair_residues for x
                pair_residues[x] = {}
                S_without_x = S[:-1]
                for i in range(len(S_without_x)):
                    for j in range(i + 1, len(S_without_x)):
                        b = S_without_x[i]
                        c = S_without_x[j]
                        r = (b + c) % x
                        if r not in pair_residues[x]:
                            pair_residues[x][r] = 0
                        pair_residues[x][r] += 1
        return S

    # Try multiple orderings
    best = []
    
    # Greedy from large to small
    S1 = greedy(list(range(n, 0, -1)))
    if len(S1) > len(best):
        best = S1[:]
    
    # Greedy from small to large
    S2 = greedy(list(range(1, n + 1)))
    if len(S2) > len(best):
        best = S2[:]
    
    # Randomized greedy with bias towards large numbers
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        # Sort with bias towards larger numbers
        order.sort(key=lambda x: (-x * 0.8 - rng.random() * 0.2))
        S = greedy(order)
        if len(S) > len(best):
            best = S[:]
    
    # Local search: try to swap elements to improve
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(best):
            if time.time() > deadline:
                break
            # Try removing a
            S_temp = best[:i] + best[i+1:]
            S_temp_set = set(S_temp)
            pair_residues_temp = build_pair_residues(S_temp)
            
            # Try adding each candidate not in S_temp
            for x in range(1, n + 1):
                if x in S_temp_set:
                    continue
                if is_valid_addition(S_temp, pair_residues_temp, x):
                    # Found a valid swap
                    S_temp.append(x)
                    best = S_temp
                    improved = True
                    break
            if improved:
                break
    
    return best
