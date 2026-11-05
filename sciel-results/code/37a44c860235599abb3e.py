"""Maximum Independent Set in 3-uniform hypergraph via branch-and-bound."""
import time
import sys
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    N = 180

    # 1. Precompute all forbidden triples {a,b,c} with 1/a = 1/b + 1/c, distinct, in [1,N]
    # (b-a)(c-a) = a^2, with a < b, a < c, b != c
    triples = []
    for a in range(1, N + 1):
        a_sq = a * a
        for d in range(1, a):  # d < a ensures d < e so b < c
            if a_sq % d == 0:
                e = a_sq // d
                b = a + d
                c = a + e
                if b <= N and c <= N:
                    triples.append((a, b, c))

    triples = list(set(triples))

    # 2. Build the 3-uniform hypergraph
    # For each vertex, store the set of edges (as frozensets) it participates in
    # For B&B, we need: for each pair (u,v), the set of w such that {u,v,w} is an edge
    pair_conflicts = defaultdict(set)  # pair_conflicts[(u,v)] = set of w
    vertex_edges = defaultdict(set)   # vertex_edges[v] = set of edges containing v

    for (a, b, c) in triples:
        for pair in [(a, b), (a, c), (b, c)]:
            u, v = pair
            other = a + b + c - u - v
            pair_conflicts[(u, v)].add(other)
            pair_conflicts[(v, u)].add(other)
        for v in (a, b, c):
            vertex_edges[v].add(frozenset((a, b, c)))

    # 3. Build the "conflict graph" for coloring:
    # Two vertices u, v are adjacent in the conflict graph if there exists w
    # such that {u,v,w} is a forbidden triple.
    # Actually, for the independent set in a 3-uniform hypergraph, the standard
    # approach is different. Let me reconsider.
    #
    # A set S is independent (valid) iff no edge {a,b,c} is fully contained in S.
    # This is the 3-uniform hypergraph independent set problem.
    #
    # Branch-and-bound approach:
    # State: (P, X) where P = candidates that could be added, X = excluded.
    # At each step, pick a vertex v from P.
    # Branch 1: Include v -> remove v from P, and for every edge {v, w, z} with w,z in P,
    #            we must exclude at least one of w,z. This is complex.
    #
    # Alternative: Use the "pair" structure.
    # If we include v, then for each pair (w,z) in forbidden_pairs[v], we cannot include both w and z.
    # This is like a 2-SAT constraint on the remaining variables.
    #
    # Better approach: Reduce to a graph problem.
    # Key insight: For a fixed vertex v that we decide to include, the remaining problem
    # is: choose a subset of P\{v} such that no forbidden pair (w,z) for v is both chosen,
    # AND the subset itself is valid (no triple fully within it).
    #
    # Let me use a different B&B formulation:
    # We process vertices in some order. For each vertex, decide include/exclude.
    # If we include vertex v, we must exclude all vertices w such that there exists
    # a triple {v, w, z} where z is already included. This is the "forward checking" approach.
    #
    # Actually, the cleanest B&B for 3-uniform hypergraph independent set:
    # State: (P, included_count)
    # P = set of vertices still under consideration.
    # We maintain that no forbidden triple is fully in the chosen set.
    #
    # Simpler: Since N=180 is small, let me use a different strategy.
    #
    # STRATEGY: Use the structure that if we sort by value, small numbers create more constraints.
    # Process in order 1, 2, ..., 180. For each number, decide include or not.
    # When we include a number a, it creates pairwise constraints on future numbers.
    # Specifically, if a is included, then for each (b,c) in forbidden_pairs[a],
    # we cannot include both b and c.
    #
    # This is equivalent to: after fixing which small numbers are included,
    # the remaining problem on large numbers is a maximum independent set in a graph
    # (where edges represent "cannot both be chosen" pairs).
    #
    # So: Branch on small numbers (1..K), then solve the resulting graph MIS for large numbers.
    #
    # Let me try K=40 or so, branching on 2^40 is too much.
    #
    # Alternative cleaner approach: Direct B&B on the 3-uniform hypergraph.
    #
    # Let me implement a proper B&B:
    # - Maintain P (candidate set) and a set of "active constraints" (pairs that can't both be in P)
    # - Actually, let me just use the direct approach:
    #
    # The valid set S must satisfy: for every triple {a,b,c}, not all three in S.
    #
    # B&B:
    # - P = set of vertices not yet decided
    # - C = set of vertices already chosen (committed)
    # - For each pair (u,v) in C, we know which w are forbidden (can't be in S if both u,v in C)
    #
    # When we add v to C, new forbidden pairs are created: for each (w,z) in forbidden_pairs[v],
    # if w is in C, then z is forbidden; if z is in C, then w is forbidden;
    # if both w,z in P, then (w,z) becomes a "mutually exclusive" pair.
    #
    # This is getting complex. Let me use a simpler but effective approach:
    #
    # APPROACH: Greedy coloring + branch and bound on the "conflict graph"
    # where two vertices u,v are connected if there's a triple {u,v,w} for some w.
    # Wait, that's not quite right either.
    #
    # Let me just implement a clean B&B for 3-uniform hypergraph independent set:

    # Precompute for fast lookup
    # forbidden_pairs[x] = list of (y, z) such that {x,y,z} is a triple
    forbidden_pairs = [set() for _ in range(N + 1)]
    for (a, b, c) in triples:
        forbidden_pairs[a].add((b, c))
        forbidden_pairs[b].add((a, c))
        forbidden_pairs[c].add((a, b))

    # Convert to lists for faster iteration
    fp = [list(s) for s in forbidden_pairs]

    # For B&B, we need to check: given a set S, can we add vertex v?
    # v can be added iff for no (y,z) in fp[v] do we have y in S AND z in S.

    # Upper bound: |P| (trivial) or use coloring-based bound.
    # For a 3-uniform hypergraph, a good upper bound on the independent set size
    # in P is: |P| - (number of forced exclusions).
    #
    # Let me use a simpler but effective B&B:
    # Process vertices in a fixed order. At each step, branch: include or exclude.
    # Prune if current_size + |remaining| <= best_found.
    # Also prune using the constraint that including v forces certain exclusions.

    # Order vertices by degree (number of constraints) descending - most constrained first
    degrees = [len(fp[i]) for i in range(1, N + 1)]
    order = sorted(range(1, N + 1), key=lambda x: -degrees[x])

    # B&B state:
    # - chosen: set of vertices chosen so far
    # - excluded: set of vertices that must be excluded (due to constraints from chosen)
    # - pos: index in order (next vertex to consider)
    # - current_size: len(chosen)
    #
    # When we choose vertex v at position pos:
    #   For each (y, z) in fp[v]:
    #     If y is chosen, then z must be excluded.
    #     If z is chosen, then y must be excluded.
    #     If both y and z are not yet decided (in future positions), we add a "mutual exclusion" constraint.
    #
    # This mutual exclusion tracking is the complex part. Let me simplify:
    # Instead of tracking all mutual exclusions, I'll just check validity at the end
    # and use a tighter upper bound.
    #
    # Actually, for N=180 with ~few thousand triples, let me try a different approach:
    #
    # APPROACH: Use the fact that the problem decomposes nicely.
    # If we include a small number a, it creates O(a) pairwise constraints.
    # Numbers > 90 have few constraints (since b,c must be > a and <= 180).
    #
    # Let me count triples per vertex:
    # Small numbers (1-30) have many triples, large numbers (150-180) have few.
    #
    # KEY INSIGHT: Numbers 91-180 form a nearly independent set!
    # Check: if a >= 91, then b >= a+1 >= 92, c >= a+1 >= 92.
    # 1/a = 1/b + 1/c <= 2/(a+1).
    # For a=91: 1/91 vs 2/92 = 1/46. 1/91 < 1/46, so possible.
    # But we need (b-a)(c-a) = a^2 with b,c <= 180.
    # b-a <= 180-a, c-a <= 180-a.
    # So (b-a)(c-a) <= (180-a)^2.
    # We need a^2 <= (180-a)^2, i.e., a <= 180-a, i.e., a <= 90.
    # So for a >= 91, NO triples exist where a is the smallest element!
    #
    # This means: if all three elements are >= 91, there's no valid triple!
    # So {91, 92, ..., 180} is a valid set of size 90!
    #
    # Can we do better? We can add some numbers from 1-90 if they don't create
    # triples with elements in {91,...,180} or with other added small numbers.
    #
    # If we add a small number a (1 <= a <= 90), we need:
    # - No triple {a, b, c} with b,c in {91,...,180}
    # - No triple {a, b, c} with b,c in our chosen small numbers
    #
    # For the first condition: triple {a, b, c} with a < b, a < c, (b-a)(c-a)=a^2.
    # If b >= 91 and c >= 91, then b-a >= 91-a and c-a >= 91-a.
    # So (b-a)(c-a) >= (91-a)^2.
    # We need a^2 >= (91-a)^2, i.e., a >= 91-a, i.e., a >= 45.5, so a >= 46.
    # For a <= 45: (91-a)^2 > a^2, so no triple with both b,c >= 91.
    # For a >= 46: there may be triples with b,c >= 91.
    #
    # So numbers 1-45 can be freely added (they don't form triples with two large numbers).
    # But they might form triples with one large and one small, or two smalls.
    #
    # Let me refine: Start with S0 = {91, ..., 180} (size 90).
    # Try to add numbers from 1 to 90 one by one (greedily or via optimization).
    #
    # When adding a to S:
    # - Check no triple {a, b, c} with b,c already in S.
    # - This means: for each (b,c) in fp[a], not both b and c in S.
    #
    # Since S0 = {91,...,180}, for a in 1..45:
    #   fp[a] contains pairs (b,c) where b,c > a and (b-a)(c-a)=a^2.
    #   Since a <= 45, b-a <= a-1 < a, so b < 2a <= 90. Wait no.
    #   b = a + d, c = a + a^2/d where d divides a^2, 1 <= d < a.
    #   Max b = a + (a^2/1) = a + a^2 (when d=1). For a=45: b up to 45+2025 > 180.
    #   Min c = a + a (when d=a-1, e=a^2/(a-1) ~ a+1). c ~ 2a+1.
    #   For a <= 45: max element in triple is a + a^2 (d=1) which can be > 180.
    #   But we need b <= 180 and c <= 180.
    #
    # Let me just compute which small numbers can be added to S0:

    S0 = set(range(91, N + 1))
    best_set = set(S0)
    best_size = len(best_set)

    # For each small number a, check if it can be added to current best_set
    # We'll do a greedy pass, then a local search to improve.

    # Greedy: try adding each a in 1..90 in various orders
    def can_add(a, S):
        """Check if a can be added to S without creating a triple."""
        for (b, c) in fp[a]:
            if b in S and c in S:
                return False
        return True

    # Multiple greedy passes with different orders
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        if trial == 0:
            order_small = list(range(1, 91))
        elif trial == 1:
            order_small = list(range(90, 0, -1))
        elif trial == 2:
            # Add by increasing "constraint count" (fewer constraints first)
            order_small = sorted(range(1, 91), key=lambda x: len(fp[x]))
        elif trial == 3:
            order_small = sorted(range(1, 91), key=lambda x: -len(fp[x]))
        else:
            order_small = list(range(1, 91))
            random.seed(trial)
            random.shuffle(order_small)

        S = set(S0)
        for a in order_small:
            if can_add(a, S):
                S.add(a)

        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Now do local search: try to swap elements to improve
    def local_search_improve(S, time_limit):
        """Try to improve S by removing one element and adding one or more."""
        current = set(S)
        best = set(current)
        best_sz = len(current)
        start = time.time()

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Try: remove x, add y (net 0 change in size, but may enable future adds)
            # Better: remove x, then greedily add as many as possible
            for x in list(current):
                if time.time() > time_limit:
                    break
                trial = current - {x}
                # Greedily add
                for y in range(1, N + 1):
                    if y in trial:
                        continue
                    if can_add(y, trial):
                        trial.add(y)
                if len(trial) > best_sz:
                    best_sz = len(trial)
                    best = set(trial)
                    current = trial
                    improved = True
                    break

            # Try: remove two elements x1, x2, then greedily add
            if not improved and len(current) >= 2 and time.time() < time_limit:
                # Random pairs to remove
                for _ in range(50):
                    if time.time() > time_limit:
                        break
                    if len(current) < 2:
                        break
                    x1, x2 = random.sample(list(current), 2)
                    trial = current - {x1, x2}
                    for y in range(1, N + 1):
                        if y in trial:
                            continue
                        if can_add(y, trial):
                            trial.add(y)
                    if len(trial) > best_sz:
                        best_sz = len(trial)
                        best = set(trial)
                        current = trial
                        improved = True
                        break

            # Try: add a single element if possible (shouldn't happen if greedy was thorough)
            if not improved:
                for y in range(1, N + 1):
                    if y not in current and can_add(y, current):
                        current.add(y)
                        best_sz = len(current)
                        best = set(current)
                        improved = True
                        break

        return best

    # Run local search
    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        improved = local_search_improve(best_set, remaining)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)

    # Second round of local search with more aggressive kicks
    remaining = deadline - time.time() - 2.0
    if remaining > 5.0:
        # Random restarts from perturbed best
        for kick in range(10):
            if time.time() > deadline - 2:
                break
            S = set(best_set)
            # Remove 3-5 random elements
            k = random.randint(3, 5)
            if len(S) >= k:
                to_remove = random.sample(list(S), k)
                S -= set(to_remove)
            # Greedily refill
            for y in range(1, N + 1):
                if y not in S and can_add(y, S):
                    S.add(y)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            # Local search from here
            rem = deadline - time.time() - 2.0
            if rem > 3.0:
                S2 = local_search_improve(S, rem)
                if len(S2) > best_size:
                    best_size = len(S2)
                    best_set = set(S2)

    return sorted(best_set)

# EVOLVE-BLOCK-END
