"""Structural rewrite: branch-and-bound maximum independent set on the
conflict graph (Tomita-style), with bitsets and a greedy coloring bound.

A pair {a,b} is a "conflict" if there exists c in [1,128] such that
lcm(a,b)=lcm(a,c)=lcm(b,c).  A valid set is an independent set in this graph.
We maximize |independent set| via exact branch-and-bound (fast for n=128).
"""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0

    def _lcm(a, b):
        return a // gcd(a, b) * b

    n = 128
    full_mask = (1 << n) - 1

    # ---- Build conflict graph (pair is conflict iff a common bad third exists)
    adj = [0] * (n + 1)          # bit i (0-based) set if (i+1) conflicts with x
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    adj[a] |= 1 << (b - 1)
                    adj[b] |= 1 << (a - 1)
                    break

    # Precompute non-adjacent mask (compatible vertices) for each vertex
    # compat[x] has bits set for all vertices NOT adjacent to x (and not x itself)
    compat = [0] * (n + 1)
    for x in range(1, n + 1):
        compat[x] = full_mask & ~(adj[x] | (1 << (x - 1)))

    # ---- Fast greedy coloring bound (Tomita-style with bit operations)
    # Processes vertices in a fixed order, uses bit-level color tracking
    def color_bound(cands):
        """Greedy coloring bound: returns upper bound on independent set size in cands."""
        if cands == 0:
            return 0
        # We'll use a simple greedy coloring: repeatedly find maximal independent sets
        # in the induced subgraph, each giving one color class.
        # But for speed, use the standard Tomita approach:
        # Order vertices by degree (precomputed), then greedily assign colors.
        
        # Simpler fast approach: count via iterative removal
        # Use a list-based greedy coloring with precomputed ordering
        # For n=128, even O(n^2) is fine if we do it smartly
        
        remaining = cands
        num_colors = 0
        # Precompute vertex order by degree (once, outside)
        # Use static ordering: vertices sorted by degree descending
        # Actually, for the bound to be valid, any ordering works.
        # Let's use a bit-parallel approach:
        
        # Fast method: repeatedly find a maximal clique in the complement
        # (i.e., maximal independent set in original) to form color classes.
        # Each color class is an independent set, so the number of color classes
        # is an upper bound on the size of any independent set? No, that's wrong.
        # The number of colors in a proper coloring is an UPPER bound on 
        # the independence number? No! The independence number is at most the
        # number of colors only if... actually no. 
        # 
        # Wait: In a proper coloring with k colors, each color class is an
        # independent set. The largest independent set could be as large as
        # the largest color class. So the number of colors is NOT an upper
        # bound on the independence number.
        #
        # Actually, I had it backwards. The standard Tomita bound is:
        # color the vertices greedily, and the color number gives an upper
        # bound on the clique number (in the complement graph). For maximum
        # independent set in G = maximum clique in complement of G.
        #
        # So: color_bound on the COMPLEMENT graph gives upper bound on
        # maximum clique in complement = maximum independent set in G.
        #
        # Let me redo: we want max independent set in G. This equals max
        # clique in complement(G). We color complement(G) greedily; the
        # number of colors is an upper bound on the clique number of
        # complement(G) = upper bound on independence number of G.
        #
        # So we need to color the complement graph. Two vertices are
        # adjacent in complement if they are NOT adjacent in G (and distinct).
        
        # For vertex v, its neighbors in complement are compat[v] (non-adjacent in G)
        # We greedily color the complement graph induced by cands.
        
        remaining = cands
        num_colors = 0
        while remaining:
            num_colors += 1
            # Find a maximal clique in complement[remaining] = maximal independent set in G[remaining]
            # Greedy: pick vertices one by one that are compatible with all chosen
            color_class = 0
            avail = remaining
            while avail:
                # Pick the vertex with smallest index (or highest, doesn't matter for bound)
                lb = avail & (-avail)
                v = lb.bit_length()
                # Add v to color class
                color_class |= lb
                # Remove v and its non-neighbors in complement (i.e., neighbors in G) from avail
                # In complement, v is adjacent to compat[v]. So non-neighbors in complement = adj[v]
                # We can only keep vertices that are adjacent to v in complement, i.e., in compat[v]
                avail = avail & compat[v] & ~color_class
            remaining &= ~color_class
        
        return num_colors

    # Precompute degree for ordering
    degrees = [bin(adj[x]).count('1') for x in range(n + 1)]
    
    # ---- Initial greedy independent set (lower bound)
    def greedy_seed(order):
        chosen = set()
        chosen_mask = 0
        for x in order:
            if (adj[x] & chosen_mask) == 0:
                chosen.add(x)
                chosen_mask |= (1 << (x - 1))
        return chosen

    best = greedy_seed(range(1, n + 1))
    for order in (range(n, 0, -1),
                  sorted(range(1, n + 1), key=lambda x: degrees[x])):
        s = greedy_seed(order)
        if len(s) > len(best):
            best = s
    best_size = len(best)

    # ---- Branch and bound (Tomita-style) with fast coloring
    # Use recursion with bitset candidates
    def expand(cands, size):
        nonlocal best, best_size
        if time.time() > deadline - 3.0:
            return
        if cands == 0:
            return
        cb = color_bound(cands)
        if size + cb <= best_size:
            return
        while cands:
            lb = cands & (-cands)
            x = lb.bit_length()
            cands ^= lb
            # If we include x, candidates become cands & compat[x]
            new_cands = cands & compat[x]
            if size + 1 + color_bound(new_cands) > best_size:
                expand(new_cands, size + 1)
            if size + color_bound(cands) <= best_size:
                return

    if time.time() < deadline - 5.0:
        expand(full_mask, 0)

    # ---- Local search with bit-parallel operations
    # Convert best to a mask for fast operations
    def set_to_mask(S):
        m = 0
        for x in S:
            m |= (1 << (x - 1))
        return m

    def mask_to_set(m):
        s = set()
        while m:
            lb = m & (-m)
            s.add(lb.bit_length())
            m ^= lb
        return s

    # Phase 1: Add any vertex that can be added (using bit operations)
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        S_mask = set_to_mask(best)
        # Find all vertices not in S that are compatible with all of S
        # A vertex x is addable if adj[x] & S_mask == 0
        # Equivalent: x is in compat of every element in S
        # Faster: iterate over candidates not in S
        candidates = full_mask & ~S_mask
        while candidates:
            lb = candidates & (-candidates)
            x = lb.bit_length()
            if (adj[x] & S_mask) == 0:
                best.add(x)
                best_size += 1
                S_mask |= lb
                candidates &= ~lb
                improved = True
            else:
                candidates ^= lb

    # Phase 2: 1-for-2 exchanges using bit operations
    # For each x in best, remove x, then try to add vertices compatible with remaining
    for iteration in range(2000):
        if time.time() > deadline - 3.0:
            break
        found = False
        S_mask = set_to_mask(best)
        best_list = list(best)
        for x in best_list:
            x_bit = 1 << (x - 1)
            trial_mask = S_mask & ~x_bit
            # Find vertices addable to trial
            # A vertex y is addable if y not in trial, y != x, and adj[y] & trial_mask == 0
            candidates = full_mask & ~trial_mask & ~x_bit
            # We need adj[y] & trial_mask == 0 for y
            # Iterate and collect
            addable = 0
            c = candidates
            while c:
                lb = c & (-c)
                y = lb.bit_length()
                if (adj[y] & trial_mask) == 0:
                    addable |= lb
                c ^= lb
            # Greedily add from addable
            trial_mask2 = trial_mask
            added = 0
            a = addable
            while a:
                lb = a & (-a)
                y = lb.bit_length()
                if (adj[y] & trial_mask2) == 0:
                    trial_mask2 |= lb
                    added += 1
                a ^= lb
            if added >= 2:
                # We can improve: remove x, add the greedily selected vertices
                best = mask_to_set(trial_mask2)
                best_size = len(best)
                found = True
                break
        if not found:
            break

    # Phase 3: Random restarts with local search
    if time.time() < deadline - 5.0:
        for _ in range(50):
            if time.time() > deadline - 3.0:
                break
            # Random subset of current best
            random.shuffle(best_list)
            trial = set(best_list[:len(best_list) // 2])
            trial_mask = set_to_mask(trial)
            # Greedily add
            candidates = full_mask & ~trial_mask
            while candidates:
                lb = candidates & (-candidates)
                x = lb.bit_length()
                if (adj[x] & trial_mask) == 0:
                    trial.add(x)
                    trial_mask |= lb
                    candidates &= ~lb
                else:
                    candidates ^= lb
            if len(trial) > best_size:
                best = trial
                best_size = len(best)

    return list(best)
# EVOLVE-BLOCK-END
