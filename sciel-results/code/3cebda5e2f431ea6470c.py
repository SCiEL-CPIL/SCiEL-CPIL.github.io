"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s out of 90s budget
    N = 180
    
    # 1. Precompute all reciprocal triples (a, b, c) with 1/a = 1/b + 1/c and a,b,c distinct in [1, N]
    # Note: 1/a = 1/b + 1/c  <=>  (b-a)(c-a) = a^2
    # Since a,b,c are distinct and positive, and 1/a = 1/b + 1/c > 1/c, we have a < c.
    # Similarly a < b. So a is the smallest element in any such triple.
    
    triples = []
    for a in range(1, N + 1):
        a_sq = a * a
        # Find divisors d of a_sq
        # Let d = b - a, e = c - a. Then d * e = a^2.
        # b = a + d, c = a + e.
        # We need b <= N and c <= N, and b != c (which implies d != e, unless we handle order).
        # Actually, the pair (b,c) is unordered for the equation, but the set {a,b,c} is what matters.
        # We just need to find all valid (b,c) pairs.
        
        # Iterate d from 1 to sqrt(a^2) = a
        # If d divides a^2, then e = a^2 // d is also a divisor.
        # b = a + d, c = a + e
        # We must ensure b, c <= N and b != c.
        
        for d in range(1, a + 1):
            if a_sq % d == 0:
                e = a_sq // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    triples.append((a, b, c))
                # Note: if d == e, then b == c, which is invalid.
                # If d != e, then (a, b, c) and (a, c, b) are the same set {a, b, c}.
                # We only need to store the set once.
                # Since d and e are just divisors, iterating d up to sqrt(a^2) covers all pairs {d, e}.
                # Wait, if d < e, then b < c. If d > e, then b > c.
                # But we only iterate d up to a. Since d*e = a^2, if d < a, then e > a.
                # So d < e always when d < a. If d = a, e = a, b=c=a+a=2a, invalid.
                # So for d in 1..a-1, we have d < e, so b < c.
                # This generates each unique pair {b,c} exactly once for a fixed a.
    
    # Remove duplicates just in case (though logic above should prevent it)
    triples = list(set(triples))
    
    # 2. Build conflict lists
    # For each element x, we need to know which pairs (y, z) cannot both be in the set if x is in the set.
    # Specifically, if x is in S, then for any triple (x, y, z) or (y, x, z) or (y, z, x),
    # we cannot have both y and z in S.
    # Actually, the constraint is: if {x, y, z} is a triple, we cannot have all three in S.
    # This is a 3-uniform hypergraph independent set problem.
    
    # Let's define `forbidden_pairs[x]` as a set of frozensets or sorted tuples (y, z)
    # such that {x, y, z} is a triple.
    # If x is in S, then for each (y, z) in forbidden_pairs[x], we cannot have both y and z in S.
    
    forbidden_pairs = [set() for _ in range(N + 1)]
    for (a, b, c) in triples:
        # Triple is {a, b, c}
        # If a is in S, (b, c) is forbidden
        forbidden_pairs[a].add((b, c))
        # If b is in S, (a, c) is forbidden
        forbidden_pairs[b].add((a, c))
        # If c is in S, (a, b) is forbidden
        forbidden_pairs[c].add((a, b))
    
    # To speed up checking "can I add x to S?", we need to check if any forbidden pair for x is fully contained in S.
    # This is still O(deg(x)) per check.
    
    # 3. Strategy: Local Search with "Repair" and "Kick" moves.
    # Instead of strict validity during search, we allow invalid states but penalize them.
    # Or, we stick to valid states but use a more powerful neighborhood.
    # Let's try a "Max-SAT" style approach:
    # Objective: Maximize |S| - lambda * (number of violations)
    # But the task requires a valid set. So we must return a valid set.
    # We can search in the space of all subsets, scoring by size, but only accept valid ones as "best".
    # If we allow invalid states, we need a repair step.
    
    # Alternative: Use the structure that `a` is the smallest.
    # If we sort elements, the constraints are "forward-looking" in a sense? No.
    
    # Let's try a different structural change: **Partitioning by magnitude**.
    # Elements 1..90 are "small", 91..180 are "large".
    # Triples involving only large elements?
    # If a, b, c > 90, then 1/a < 1/90.
    # 1/b + 1/c < 2/91 ~ 0.022. 1/90 ~ 0.011.
    # It is possible to have triples in the large range.
    # Example: a=91, b=182 (out of range).
    # Max a in triple: 1/a = 1/b + 1/c. Min b,c = a+1.
    # 1/a < 2/(a+1) => a+1 < 2a => a > 1.
    # So triples exist for all a > 1.
    
    # Let's implement a robust **Simulated Annealing** that allows invalid states
    # but uses a penalty function. This allows "kicking" out of local optima
    # by temporarily accepting violations, then repairing.
    
    def count_violations(S):
        """Count the number of forbidden triples fully contained in S."""
        # This is expensive. We need a faster way.
        # We can maintain a count of violations incrementally.
        # But for SA, we need to evaluate neighbors quickly.
        # Let's stick to valid-state search but with a better move set.
        pass
    
    # Let's go back to valid-state search but improve the move generation.
    # Current moves: Add, Remove, Swap.
    # Problem: "Add" is often blocked. "Swap" keeps size same.
    # To increase size, we need to find a sequence of swaps that enables an add.
    # This is a "2-opt" or "k-opt" move.
    
    # Let's implement **2-opt moves**: Remove x1, x2 and add y1, y2.
    # This changes size by 0, but can change the structure to allow future adds.
    # Or, **1-add, 1-remove** (Swap) is already there.
    
    # A more powerful approach: **Greedy with Randomized Backtracking**.
    # Or **Genetic Algorithm**.
    
    # Let's try **Genetic Algorithm**.
    # Representation: Bitmask of length 180.
    # Fitness: Size of set if valid, else 0 (or negative penalty).
    # Crossover: Uniform crossover.
    # Mutation: Flip random bits.
    # Selection: Tournament selection.
    
    # But GA can be slow.
    
    # Let's try **Multi-Start Local Search** with **2-opt** moves.
    
    def is_valid(S):
        # Check if S is valid.
        # For each a in S, check if any (b,c) in forbidden_pairs[a] has both b,c in S.
        # This is O(|S| * avg_deg).
        # To speed up, we can precompute for each pair (b,c) which 'a's it conflicts with.
        # Actually, the check is: for each a in S, for each (b,c) in forbidden_pairs[a], if b in S and c in S, return False.
        # This is sufficient.
        for a in S:
            for (b, c) in forbidden_pairs[a]:
                if b in S and c in S:
                    return False
        return True
    
    def get_violations(S):
        """Return list of violating triples."""
        viol = []
        for a in S:
            for (b, c) in forbidden_pairs[a]:
                if b in S and c in S:
                    viol.append((a, b, c))
        return viol
    
    def greedy_construct(order):
        S = set()
        for x in order:
            # Check if adding x is valid
            valid = True
            for (y, z) in forbidden_pairs[x]:
                if y in S and z in S:
                    valid = False
                    break
            if valid:
                S.add(x)
        return S
    
    def local_search(S, time_limit):
        """
        Local search with 1-swap and 2-swap moves.
        Returns improved set.
        """
        current = set(S)
        best = set(current)
        best_size = len(current)
        start = time.time()
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try 1-swap
            for x in list(current):
                if time.time() > time_limit:
                    return best
                trial = current - {x}
                # Try to add any y
                for y in range(1, N + 1):
                    if y in trial or y == x:
                        continue
                    # Check if y can be added to trial
                    valid = True
                    for (u, v) in forbidden_pairs[y]:
                        if u in trial and v in trial:
                            valid = False
                            break
                    if valid:
                        new_set = trial | {y}
                        # Try to add more to new_set greedily
                        # This is a "kick" to escape
                        for z in range(1, N + 1):
                            if z in new_set:
                                continue
                            valid_z = True
                            for (u, v) in forbidden_pairs[z]:
                                if u in new_set and v in new_set:
                                    valid_z = False
                                    break
                            if valid_z:
                                new_set.add(z)
                        if len(new_set) > best_size:
                            best = new_set
                            best_size = len(new_set)
                            current = new_set
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            
            # Try 2-swap: remove x1, x2, add y1, y2
            # This is expensive. O(N^4).
            # We can limit this to random 2-swaps.
            if time.time() > time_limit - 5:
                break
                
            # Random 2-swap
            if len(current) >= 2:
                x1, x2 = random.sample(list(current), 2)
                trial = current - {x1, x2}
                # Pick y1, y2 randomly from outside
                outside = [x for x in range(1, N + 1) if x not in trial]
                if len(outside) >= 2:
                    y1, y2 = random.sample(outside, 2)
                    new_set = trial | {y1, y2}
                    # Check validity
                    if is_valid(new_set):
                        # Try to add more
                        for z in range(1, N + 1):
                            if z in new_set:
                                continue
                            valid_z = True
                            for (u, v) in forbidden_pairs[z]:
                                if u in new_set and v in new_set:
                                    valid_z = False
                                    break
                            if valid_z:
                                new_set.add(z)
                        if len(new_set) > best_size:
                            best = new_set
                            best_size = len(new_set)
                            current = new_set
                            improved = True
    
        return best
    
    # Multi-start
    best_overall = set(range(91, 181))
    best_size_overall = len(best_overall)
    
    # Start 1: Greedy from random order
    random.seed(42)
    for i in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_size_overall:
            best_overall = S
            best_size_overall = len(S)
    
    # Start 2: Local search from best found
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        improved_set = local_search(best_overall, remaining)
        if len(improved_set) > best_size_overall:
            best_overall = improved_set
            best_size_overall = len(improved_set)
    
    # Start 3: More local search if time remains
    remaining = deadline - time.time() - 2.0
    if remaining > 5.0:
        improved_set = local_search(best_overall, remaining)
        if len(improved_set) > best_size_overall:
            best_overall = improved_set
            best_size_overall = len(improved_set)
            
    return sorted(best_overall)

# EVOLVE-BLOCK-END
