"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s out of 90s budget
    N = 180
    
    # 1. Precompute the conflict graph
    # A conflict exists between x and y if there exists a third element z in [1, N]
    # such that {x, y, z} is a reciprocal triple (1/a = 1/b + 1/c with distinct a,b,c).
    # The graph is undirected: if x and y conflict, we cannot have both in the set.
    
    # To build the graph efficiently:
    # For each a in 1..N, find all pairs (b, c) such that (b-a)(c-a) = a^2.
    # For each such pair, add an edge between b and c.
    # Note: The triple is (a, b, c). The conflicts are:
    #   - b and c conflict (because of a)
    #   - a and b conflict? No, the constraint is on the set {a,b,c}. 
    #     Wait, the problem says "no THREE DISTINCT elements a,b,c satisfying...".
    #     This means if we pick a set S, there should be no distinct x,y,z in S 
    #     such that 1/x = 1/y + 1/z.
    #     The equation is symmetric in y and z. x is the "sum" side.
    #     So for a specific triple (a, b, c) where 1/a = 1/b + 1/c:
    #     If we include all three {a,b,c}, it's invalid.
    #     This is a hypergraph independent set problem (forbidden hyperedges of size 3).
    #     However, we can convert this to a graph problem?
    #     Not directly, because forbidding {a,b,c} doesn't mean forbidding {a,b} or {a,c}.
    #     BUT, looking at the current program's logic:
    #     `is_valid_addition` checks if adding x creates a triple.
    #     This implies the problem is equivalent to finding a set with no forbidden triple.
    #     This is the "Maximum 3-Hypergraph Independent Set" problem.
    #     Standard Max Independent Set solvers work on graphs (2-uniform).
    #     Can we reduce 3-uniform to 2-uniform?
    #     Generally no, but maybe the structure of reciprocal triples allows it?
    #     Let's check: If 1/a = 1/b + 1/c, then a < b and a < c.
    #     So a is always the smallest element in the triple.
    #     This structure might allow a different approach.
    
    # Let's stick to the heuristic but make it much more powerful:
    # The current program uses greedy + local search (1-swap).
    # A better structural change: **Simulated Annealing** or **Genetic Algorithm** 
    # with a more complex neighborhood (2-swaps, 3-swaps) and a better fitness function.
    
    # However, the directive asks for a "different design or modelling family".
    # Let's try **Integer Linear Programming (ILP)** using a simple branch-and-bound 
    # or even just a much more aggressive **Local Search with 2-opt moves** and 
    # **Conflict-Driven Clause Learning (CDCL)** style restarts.
    
    # Actually, a very effective method for these "subset with forbidden configurations" 
    # is **Backtracking with Pruning** guided by the most constrained variable.
    
    # Let's implement a **Depth-First Search (DFS) with Pruning** that tries to include 
    # elements in a specific order, pruning branches that cannot possibly beat the current best.
    
    # Precompute constraints
    # For each element x, store the list of pairs (y, z) such that {x, y, z} is a triple.
    # If x is the 'a' (smallest), pairs are (b, c).
    # If x is 'b' or 'c', we need to know which 'a' and other element form the triple.
    
    # Let's precompute `conflicts[x]` as a set of (y, z) such that {x,y,z} is a triple.
    # When adding x to a set S, we must ensure that for no (y, z) in conflicts[x] 
    # are both y and z already in S.
    
    # This is still a 3-uniform hypergraph.
    # To use a graph-based MIS solver, we can't directly.
    # But we can use a **Constraint Satisfaction Problem (CSP)** solver approach.
    
    # Let's try a different structural change: **Partitioning the problem**.
    # The reciprocal relation 1/a = 1/b + 1/c implies a = bc/(b+c).
    # Since a < b and a < c, the "a" values are small.
    # Maybe we can iterate over possible "a" values? No, a is determined by b,c.
    
    # Let's go with **Simulated Annealing** with a very robust neighborhood.
    # Current: Greedy + 1-swap.
    # New: Simulated Annealing with 1-swap, 2-swap, and "kick" moves (random removals).
    
    # Precompute all triples
    triples = []
    # a is the smallest.
    # (b-a)(c-a) = a^2
    for a in range(1, N + 1):
        a_sq = a * a
        for d in range(1, int(a_sq**0.5) + 1):
            if a_sq % d == 0:
                e = a_sq // d
                b1 = a + d
                c1 = a + e
                b2 = a + e
                c2 = a + d
                if b1 <= N and c1 <= N and b1 != c1:
                    triples.append((a, min(b1, c1), max(b1, c1)))
                if d != e:
                    if b2 <= N and c2 <= N and b2 != c2:
                        triples.append((a, min(b2, c2), max(b2, c2)))
    
    # Remove duplicates
    triples = list(set(triples))
    
    # Build conflict lists for fast checking
    # For each x, list of (y, z) such that {x,y,z} is a triple.
    # We need to check if y and z are both in the set.
    # To make this faster, we can store for each x, a list of "forbidden pairs".
    # But checking "are y and z in S" is O(1) if S is a set.
    # The bottleneck is the number of pairs.
    
    conflicts = [[] for _ in range(N + 1)]
    for (a, b, c) in triples:
        # If we add a, we conflict with pair (b,c)
        conflicts[a].append((b, c))
        # If we add b, we conflict with pair (a,c)
        conflicts[b].append((a, c))
        # If we add c, we conflict with pair (a,b)
        conflicts[c].append((a, b))
    
    # Sort conflicts to help with pruning? Maybe not needed.
    
    def is_valid_addition(selected, x):
        """Check if adding x to selected keeps the set valid."""
        for (y, z) in conflicts[x]:
            if y in selected and z in selected:
                return False
        return True
    
    def greedy(order):
        selected = set()
        for x in order:
            if is_valid_addition(selected, x):
                selected.add(x)
        return selected
    
    # Initial best solution
    best_set = set(range(91, 181))
    best_size = len(best_set)
    
    # Simulated Annealing
    def simulated_annealing(initial_set, time_budget):
        current = set(initial_set)
        current_size = len(current)
        best_found = set(current)
        best_size_found = current_size
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        start = time.time()
        
        while time.time() - start < time_budget and T > T_min:
            # Generate neighbor
            # Moves:
            # 1. Add a random element not in current
            # 2. Remove a random element from current
            # 3. Swap: remove one, add one
            
            r = random.random()
            if r < 0.3 and len(current) < N:
                # Add
                candidates = [x for x in range(1, N + 1) if x not in current]
                if candidates:
                    x = random.choice(candidates)
                    if is_valid_addition(current, x):
                        new_set = current | {x}
                        new_size = current_size + 1
                        # Accept
                        current = new_set
                        current_size = new_size
                        if current_size > best_size_found:
                            best_found = set(current)
                            best_size_found = current_size
                    else:
                        # Reject or accept with probability exp(-delta/T)
                        # Delta is "cost" of invalidity? 
                        # In SA for combinatorial optimization, we usually define a cost function.
                        # Here, validity is binary. We can't add invalid states easily.
                        # So we only propose valid moves? That's just random walk on valid states.
                        # Let's stick to valid moves only (Random Walk on Valid States) 
                        # with occasional "kicks" (remove random elements to escape local optima).
                        pass
            elif r < 0.6 and len(current) > 0:
                # Remove
                x = random.choice(list(current))
                new_set = current - {x}
                new_size = current_size - 1
                # Always accept removal? Or with probability?
                # To escape local optima, we should accept removals sometimes.
                # Let's accept with probability 0.5 or based on T.
                if random.random() < 0.5:
                    current = new_set
                    current_size = new_size
            else:
                # Swap: remove x, add y
                if len(current) > 0:
                    x = random.choice(list(current))
                    trial = current - {x}
                    candidates = [y for y in range(1, N + 1) if y not in trial]
                    if candidates:
                        # Try to find a valid y
                        # Just pick random
                        y = random.choice(candidates)
                        if is_valid_addition(trial, y):
                            new_set = trial | {y}
                            # Size remains same
                            current = new_set
                            # current_size unchanged
                            if current_size > best_size_found:
                                best_found = set(current)
                                best_size_found = current_size
                            
            T *= cooling
            
        return best_found, best_size_found
    
    # Run SA for a while
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        sa_result, sa_size = simulated_annealing(best_set, remaining)
        if sa_size > best_size:
            best_set = sa_result
            best_size = sa_size
    
    # Final local improvement
    remaining = deadline - time.time() - 3.0
    if remaining > 2.0:
        # 1-swap local search
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for x in list(best_set):
                if time.time() > deadline - 3.0:
                    break
                trial = best_set - {x}
                for y in range(1, N + 1):
                    if y in trial or y == x:
                        continue
                    if is_valid_addition(trial, y):
                        # If y > x, it might be better? No, size is same.
                        # But maybe it allows future additions?
                        # For now, just accept any valid swap to explore.
                        # Actually, we want to maximize size. A swap keeps size same.
                        # So we should only swap if it leads to a state where we can add more?
                        # That's a 2-step lookahead.
                        # Let's just do the swap if it doesn't hurt, and then try to add.
                        new_set = trial | {y}
                        # Try to add more to new_set
                        added = False
                        for z in range(1, N + 1):
                            if z in new_set:
                                continue
                            if is_valid_addition(new_set, z):
                                new_set.add(z)
                                added = True
                                break
                        if added:
                            best_set = new_set
                            best_size = len(best_set)
                            improved = True
                            break
                if improved:
                    break
    
    if not best_set:
        return list(range(91, 181))
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
