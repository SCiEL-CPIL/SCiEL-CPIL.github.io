import time
import random
from itertools import combinations
from math import gcd

def construct():
    """
    Finds a large subset of [1, 128] with no three distinct elements 
    whose three pairwise GCDs are equal.
    Uses a hypergraph MIS approach with greedy initialization and local search.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    N = 128
    # 1. Precompute GCDs
    # gcd[i][j] for 1 <= i,j <= N
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Identify all bad triples (hyperedges)
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We store for each vertex the set of hyperedges it belongs to, 
    # or more efficiently, for local search, we can check validity on the fly 
    # or maintain conflict counts.
    
    # Given N=128, total triples is ~350k. This is small enough to precompute 
    # a conflict structure if needed, but on-the-fly checking with bitsets or sets 
    # is also fast.
    
    # Let's use a set of bad triples for quick lookup if we decide to use 
    # a specific data structure, but for local search, we need to check 
    # if adding x creates a bad triple with any pair in the current set.
    
    # Optimization: For each pair (a,b), precompute the list of c such that 
    # (a,b,c) is a bad triple.
    # If we have a set S, adding x is valid if for all a,b in S, 
    # (a,b,x) is NOT a bad triple.
    
    # Let's build a dictionary: pair_conflicts[(a,b)] = set of c
    pair_conflicts = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            conflicts = set()
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    conflicts.add(c)
            if conflicts:
                pair_conflicts[(a, b)] = conflicts
                pair_conflicts[(b, a)] = conflicts # Symmetric
    
    def is_valid_addition(S, x):
        """Check if x can be added to set S."""
        # x must not be in S
        # For every pair (a, b) in S, x must not be in pair_conflicts[(a,b)]
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                key = (a, b)
                if key in pair_conflicts:
                    if x in pair_conflicts[key]:
                        return False
        return True

    def greedy_init():
        """Greedy min-degree initialization."""
        S = set()
        remaining = set(range(1, N + 1))
        
        # Calculate initial "degree" (number of conflicts if added)
        # Degree of x is number of pairs (a,b) in S such that x conflicts with (a,b)
        # Since S starts empty, degree is 0 for all.
        # We pick randomly or by some heuristic.
        # Let's try picking elements that have fewest conflicts with each other?
        # Actually, standard greedy for MIS: pick vertex with min degree in remaining graph.
        # Here "degree" is dynamic.
        
        while remaining:
            if time.time() > deadline:
                break
                
            # Calculate potential conflicts for each candidate in remaining
            # If we add x, it conflicts with pairs in S.
            # We want to minimize future conflicts?
            # Heuristic: Pick x that minimizes the number of pairs (a,b) in S 
            # such that x is in conflict with (a,b).
            # If S is empty, all are 0.
            
            best_x = None
            min_conflicts = float('inf')
            
            # To speed up, we can just pick randomly if S is small, 
            # or use a heuristic.
            # Let's try a heuristic: pick x that is "least connected" in the 
            # induced subgraph of remaining? No, that's for pairwise.
            
            # Simple heuristic: Pick x that minimizes the size of 
            # union of conflicts it creates with existing pairs?
            
            # For simplicity and speed, let's just iterate through remaining 
            # and check validity. If valid, add it.
            # To make it "min-degree", we can count how many future elements 
            # would be blocked? That's expensive.
            
            # Let's stick to a simple greedy: 
            # Sort remaining by some static property? 
            # Or just pick the first valid one?
            # The knowledge says "greedy min-degree".
            
            # Let's approximate min-degree by counting how many elements in `remaining`
            # would become invalid if we added x? No, that's not it.
            
            # Let's try: Pick x that has the fewest "bad triples" involving x 
            # and any two other numbers in `remaining`?
            # This is a static measure of how "crowded" x is.
            
            # Precomputed static "badness" for each x:
            # Count of pairs (a,b) in [1..N] such that (x,a,b) is a bad triple.
            # We can compute this once.
            
            pass # Placeholder, will implement static score below

        return S

    # Precompute static "conflict count" for each x
    # score[x] = number of pairs (a,b) such that (x,a,b) is bad
    static_conflict_count = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a, b) in pair_conflicts:
                for c in pair_conflicts[(a, b)]:
                    static_conflict_count[c] += 1
                    static_conflict_count[a] += 1 # a is part of bad triple
                    static_conflict_count[b] += 1

    def greedy_init_v2():
        S = set()
        remaining = list(range(1, N + 1))
        # Sort by static conflict count (ascending) - pick "least bad" first
        remaining.sort(key=lambda x: static_conflict_count[x])
        
        for x in remaining:
            if time.time() > deadline:
                break
            if is_valid_addition(S, x):
                S.add(x)
        return S

    def local_search(S, max_iter=1000):
        """
        Try to improve S by swapping elements.
        1. Remove a random element r from S.
        2. Try to add any element from \S that is now valid.
        3. If we can add more than we removed (or just add one), keep it.
        """
        S = set(S)
        no_improve_count = 0
        current_size = len(S)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if not S:
                break
                
            # Pick a random element to remove
            r = random.choice(list(S))
            S.remove(r)
            
            # Try to add any element from complement
            # We want to add an element that restores or increases size
            added_any = False
            # To be efficient, we can just try to add one element.
            # If we can add one, size is same. We need to add >1 to improve?
            # Or just find a local optimum where no single swap improves size.
            
            # Let's try to find ANY element that can be added.
            # If we find one, we add it. Size remains same.
            # This is a "walk" in the solution space.
            # To improve size, we need to remove 1 and add 2? 
            # Or remove 1, add 1 (neutral), then later remove 1, add 2?
            
            # Better strategy:
            # 1. Remove r.
            # 2. Check all x in [1..N] \ S.
            # 3. If x is valid, add it.
            # 4. If we added x, we have a new set of same size.
            # 5. Repeat.
            
            # To actually increase size, we might need to remove 1 and add 2.
            # Let's try: Remove r. Then try to add x1. If valid, add x1.
            # Then try to add x2. If valid, add x2.
            # If we added 2, size increased by 1.
            
            # Let's implement a "greedy fill" after removal.
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            
            added_count = 0
            for x in candidates:
                if is_valid_addition(S, x):
                    S.add(x)
                    added_count += 1
                    # We can stop if we've added enough to beat the removal (1)
                    if added_count > 1:
                        break
                        # Actually, adding just 1 keeps size same. 
                        # Adding 2 increases size.
                        # Adding 3 increases size by 2.
                        
            if added_count > 1:
                # We improved size
                current_size += added_count - 1
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    break

        return S

    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Randomize the greedy order slightly? 
        # Or just use the static sort.
        # To diversify, we can add some randomness to the static score.
        
        # Let's just run greedy_init_v2 and then local search.
        S = greedy_init_v2()
        
        # Local search to polish
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # If we've reached a very high size, we might be near optimal.
        # Max possible is 128. 
        # If best_size > 80, we can be more aggressive with time.
        
        # Add some randomness to the next restart's initialization
        # by permuting the remaining list in greedy_init_v2?
        # The current greedy_init_v2 uses a fixed sort.
        # Let's modify greedy_init_v2 to accept a random seed or shuffle.

    return sorted(list(best_set))

# Refining the greedy init to allow randomness for diversification
def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 128
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    pair_conflicts = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            conflicts = set()
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    conflicts.add(c)
            if conflicts:
                pair_conflicts[(a, b)] = conflicts
                pair_conflicts[(b, a)] = conflicts

    def is_valid_addition(S, x):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                key = (a, b)
                if key in pair_conflicts:
                    if x in pair_conflicts[key]:
                        return False
        return True

    # Precompute static conflict count
    static_conflict_count = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a, b) in pair_conflicts:
                for c in pair_conflicts[(a, b)]:
                    static_conflict_count[c] += 1
                    static_conflict_count[a] += 1
                    static_conflict_count[b] += 1

    def greedy_init(randomize=False):
        S = set()
        remaining = list(range(1, N + 1))
        if randomize:
            # Add noise to static score to diversify
            scored = [(x, static_conflict_count[x] + random.uniform(0, 10)) for x in remaining]
            scored.sort(key=lambda item: item[1])
            remaining = [item[0] for item in scored]
        else:
            remaining.sort(key=lambda x: static_conflict_count[x])
            
        for x in remaining:
            if time.time() > deadline:
                break
            if is_valid_addition(S, x):
                S.add(x)
        return S

    def local_search(S):
        S = set(S)
        no_improve_count = 0
        current_size = len(S)
        
        while time.time() < deadline and no_improve_count < 100:
            if not S:
                break
                
            r = random.choice(list(S))
            S.remove(r)
            
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            
            added_count = 0
            for x in candidates:
                if is_valid_addition(S, x):
                    S.add(x)
                    added_count += 1
                    if added_count > 1:
                        break
                        
            if added_count > 1:
                current_size += added_count - 1
                no_improve_count = 0
            else:
                no_improve_count += 1

        return S

    best_set = set()
    best_size = 0
    
    # Run restarts
    while time.time() < deadline:
        # Alternate between deterministic and randomized greedy
        use_random = random.random() > 0.5
        S = greedy_init(use_random)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # If we have a very large set, we can tighten the deadline check
        if best_size > 90:
             # Maybe do a final polish on the best set found
             pass

    return sorted(list(best_set))

if __name__ == "__main__":
    print(construct())
