"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _verify_girth(n, edges, k, g):
    """Verify the graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False
        edge_set.add((i, j))
        adj[i].append(j)
        adj[j].append(i)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    for v in range(n):
        visited = set()
        queue = deque()
        for w in adj[v]:
            visited.add((w, v))
            queue.append((w, v, 1))
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= g:
                break
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == v:
                    if depth + 1 < g:
                        return False
                    continue
                state = (neighbor, node)
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, node, depth + 1))
    return True

def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction - the baseline."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _cayley_girth_check(S, g, group_size):
    """
    Check if a Cayley graph with connection set S on Z_group_size has girth >= g.
    S is a list of generators (symmetric: if s in S then -s in S).
    We check all non-backtracking walks of length < g.
    """
    # For a Cayley graph, non-backtracking walk of length L starting from 0
    # is a sequence s1, s2, ..., sL where si != -s(i-1) for i >= 2
    # and s1 + s2 + ... + sL != 0 for L < g
    
    # We need to check that no such walk sums to 0 for L < g
    # But we also need to check from every vertex, but by translation invariance,
    # checking from 0 is sufficient for Cayley graphs.
    
    # Actually, we need to be more careful. A cycle in a Cayley graph corresponds
    # to a closed walk. A non-backtracking closed walk of length L < g would
    # violate the girth condition.
    
    # For efficiency, let's enumerate all non-backtracking sequences of length L
    # for L = 3, 4, ..., g-1 and check if any sum to 0.
    
    # But the number of sequences grows as k*(k-1)^(L-1) which is manageable for small L.
    
    S_set = set(S)
    
    for L in range(3, g):
        # Enumerate all non-backtracking sequences of length L
        # Start with any s1 in S
        for s1 in S:
            # Build sequences recursively
            sequences = [(s1,)]
            for step in range(1, L):
                new_sequences = []
                for seq in sequences:
                    last = seq[-1]
                    for s in S:
                        if s != -last:
                            new_sequences.append(seq + (s,))
                sequences = new_sequences
            
            for seq in sequences:
                total = sum(seq) % group_size
                if total == 0:
                    return False
    return True

def _construct_cayley_4_10(deadline):
    """
    Try to construct a 4-regular graph of girth 10 on 320 vertices
    using a Cayley graph on Z_32 x Z_10.
    
    Connection set: {(a1, b1), (-a1, -b1), (a2, b2), (-a2, -b2)}
    where (a1, b1) != (a2, b2) and neither is identity.
    """
    G1, G2 = 32, 10
    group_size = G1 * G2
    
    # We need to find a1, b1, a2, b2 such that the Cayley graph has girth >= 10.
    # By symmetry, we can fix a1 = 1 (WLOG, since we can multiply the first coordinate by a unit).
    # Actually, let's be more systematic.
    
    # The girth condition: no non-trivial word of length < 10 in {s1, s1^{-1}, s2, s2^{-1}}
    # with no immediate backtracking equals the identity.
    
    # Let s1 = (a1, b1), s2 = (a2, b2).
    # A word of length L is a sequence of choices from {+1, -1} for each generator.
    # Non-backtracking means we don't choose the same generator with opposite sign consecutively.
    
    # For L < 10, the number of non-backtracking words is:
    # L=1: 4 (but these are single generators, never 0)
    # L=2: 4*3 = 12 (choose first from 4, second from 3 not backtracking)
    # L=3: 4*3*3 = 36
    # ...
    # L=9: 4*3^8 = 78732
    
    # Total: 4 + 12 + 36 + 108 + ... + 4*3^8 = 4*(1 + 3 + 9 + ... + 3^8) = 4*(3^9-1)/2 = 4*9841 = 39364
    
    # For each candidate (a1, b1, a2, b2), we need to check ~40k words.
    # Number of candidates: a1 in [0,31], b1 in [0,9], a2 in [0,31], b2 in [0,9]
    # = 32*10*32*10 = 102400 candidates.
    # Total: 102400 * 39364 = ~4 billion operations. Too slow.
    
    # Better approach: use the structure. The sum of a word is:
    # (sum of ±a_i for each step, sum of ±b_i for each step)
    # where at each step we choose either generator 1 or 2, and a sign.
    
    # Let's precompute for each possible word structure (which generator at each step),
    # the set of possible (a_sum, b_sum) values.
    
    # Actually, let's use a different approach: enumerate words by their generator pattern.
    # A word is determined by:
    # - A sequence of length L of generator indices (1 or 2)
    # - A sequence of signs (+1 or -1)
    # Subject to: no immediate backtracking, i.e., if step i and i+1 use the same generator,
    #   the signs must be the same (otherwise it's backtracking).
    #   If step i and i+1 use different generators, any signs are OK.
    
    # Let's enumerate all valid (generator_sequence, sign_sequence) pairs for L = 3 to 9.
    
    def generate_words(L):
        """Generate all non-backtracking words of length L."""
        if L == 1:
            for gen in [1, 2]:
                for sign in [1, -1]:
                    yield (gen, sign)
            return
        
        # Recursive generation
        def rec(prefix_gen, prefix_sign):
            if len(prefix_gen) == L:
                yield tuple(prefix_gen), tuple(prefix_sign)
                return
            last_gen = prefix_gen[-1]
            last_sign = prefix_sign[-1]
            for gen in [1, 2]:
                for sign in [1, -1]:
                    if gen == last_gen and sign == -last_sign:
                        continue  # backtracking
                    prefix_gen.append(gen)
                    prefix_sign.append(sign)
                    yield from rec(prefix_gen, prefix_sign)
                    prefix_gen.pop()
                    prefix_sign.pop()
        
        for gen in [1, 2]:
            for sign in [1, -1]:
                yield from rec([gen], [sign])
    
    # Precompute all words for L = 3 to 9
    all_words = {}
    for L in range(3, 10):
        words = list(generate_words(L))
        all_words[L] = words
    
    # For each word, the total displacement is:
    # a_total = sum(sign_i * a_{gen_i})
    # b_total = sum(sign_i * b_{gen_i})
    # We need a_total ≡ 0 (mod 32) and b_total ≡ 0 (mod 10).
    
    # Let's try random candidates with early termination.
    rng = random.Random(12345)
    
    # Generate candidate pairs
    candidates = []
    for a1 in range(G1):
        for b1 in range(G2):
            if a1 == 0 and b1 == 0:
                continue
            for a2 in range(G1):
                for b2 in range(G2):
                    if a2 == 0 and b2 == 0:
                        continue
                    # Ensure s1 != s2 and s1 != -s2
                    if a1 == a2 and b1 == b2:
                        continue
                    if a1 == (-a2) % G1 and b1 == (-b2) % G2:
                        continue
                    candidates.append((a1, b1, a2, b2))
    
    # Shuffle and try
    rng.shuffle(candidates)
    
    start_time = time.time()
    for a1, b1, a2, b2 in candidates:
        if time.time() > deadline - 20:
            break
        
        # Check girth
        valid = True
        for L in range(3, 10):
            for gen_seq, sign_seq in all_words[L]:
                a_total = 0
                b_total = 0
                for i in range(L):
                    gen = gen_seq[i]
                    sign = sign_seq[i]
                    if gen == 1:
                        a_total += sign * a1
                        b_total += sign * b1
                    else:
                        a_total += sign * a2
                        b_total += sign * b2
                if a_total % G1 == 0 and b_total % G2 == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Found a valid connection set!
            # Construct the Cayley graph
            edges = []
            s1 = (a1, b1)
            s2 = (a2, b2)
            S = [s1, (-a1) % G1, s2, (-a2) % G2]
            S_b = [b1, (-b1) % G2, b2, (-b2) % G2]
            
            for v in range(group_size):
                v1 = v % G1
                v2 = v // G1
                for i in range(4):
                    nv1 = (v1 + S[i]) % G1
                    nv2 = (v2 + S_b[i]) % G2
                    nv = nv1 + nv2 * G1
                    if v < nv:
                        edges.append((v, nv))
            
            # Verify
            if _verify_girth(group_size, edges, 4, 10):
                return group_size, edges
            
            # If verification failed, something went wrong; continue
            continue
    
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 140  # Safety margin
    
    # Try the Cayley graph construction for (4,10)
    if k == 4 and g == 10:
        result = _construct_cayley_4_10(deadline)
        if result is not None:
            return result
    
    # Fallback to LUW construction
    n, edges = _construct_luw(k, g)
    
    return n, edges

# EVOLVE-BLOCK-END
