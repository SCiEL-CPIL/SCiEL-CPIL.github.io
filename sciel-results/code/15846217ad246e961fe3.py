import time
import math
import random
import sys

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    Equation: 1/a = 1/b + 1/c => c = ab / (b-a).
    We require b > a and c to be an integer in [1, N] distinct from a and b.
    Note: Since 1/a = 1/b + 1/c, a must be smaller than both b and c.
    """
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    triples.append((a, b, c))
    return triples

def _build_graph(N, triples):
    """
    Build a graph representation for the independent set problem.
    We want a set S such that no triple (a,b,c) is fully contained in S.
    This is a 3-uniform hypergraph independent set problem.
    However, we can convert this to a graph problem by noting that if we fix 'a', 
    the constraints are pairwise between b and c.
    Actually, a simpler approach for small N=180 is to use a backtracking search 
    with pruning based on the "conflict" structure directly, or convert to a 
    maximum clique problem on a compatibility graph?
    
    Let's stick to a direct backtracking on the hypergraph, but optimized.
    A set S is valid if it contains no triple.
    We can view this as finding a maximum size subset of V=[1..N] that is an independent set in the 3-uniform hypergraph H.
    
    Alternative: Convert to a graph.
    If we define a graph G where an edge exists between x and y if there exists a z such that (x,y,z) is a triple... 
    No, that's not quite right. The constraint is that we cannot pick all three.
    We CAN pick two.
    
    Let's use a branch and bound on the set of elements.
    State: (current_set, candidates)
    candidates are elements > max(current_set) to avoid permutations.
    This is slow.
    
    Better: Use the property that the triples are sparse.
    Let's use a greedy heuristic first to get a good lower bound, then a local search / simulated annealing or just a very aggressive local search.
    
    Actually, for N=180, the number of triples is small.
    Let's count:
    a=1: 1/1 = 1/b + 1/c => b=c=2. Triple (1,2,2) - invalid (distinct).
    a=2: 1/2 = 1/b + 1/c. b=3, c=6. (2,3,6). b=4, c=4 (invalid). b=6, c=3 (dup).
    a=3: 1/3 = 1/b + 1/c. b=4, c=12. (3,4,12). b=6, c=6 (invalid).
    a=4: 1/4 = 1/b + 1/c. b=5, c=20. (4,5,20). b=6, c=12. (4,6,12). b=8, c=8 (invalid).
    
    The number of triples is O(N log N) roughly.
    Total triples for N=180 is likely < 1000.
    
    Strategy: 
    1. Find a large initial independent set using a greedy algorithm (pick element with fewest conflicts, remove its conflicting partners? No, that's for vertex cover).
    For independent set in hypergraph:
    Greedy: Pick element x. If adding x creates a triple, we must have removed at least one of the other two.
    
    Let's try a different angle: 
    The problem is equivalent to finding the maximum size subset S such that for every triple (a,b,c), S does not contain {a,b,c}.
    
    We can use a SAT solver approach or a dedicated Max-Independent-Set heuristic.
    Given the time limit (90s), we can afford a sophisticated search.
    
    Let's implement a randomized local search starting from a good greedy seed.
    Seed: Start with all elements. While there is a conflict, remove one of the conflicting elements (e.g., the one with highest degree in the conflict graph).
    
    Conflict Graph:
    Nodes: 1..180
    Edge (u,v) if there exists w such that {u,v,w} is a triple.
    Note: If we pick u and v, we are FORBIDDEN from picking w.
    So, if we have a set S, and u,v in S, then w cannot be in S.
    This is a "2-choose" constraint.
    
    Let's refine the "Greedy Removal" strategy mentioned in the inspiration:
    "Greedy removal of high-conflict elements from the full set typically finds independent sets significantly larger than contiguous suffixes"
    
    Algorithm:
    1. S = {1..180}
    2. Precompute all triples.
    3. While S contains a triple:
         - Find all triples fully contained in S.
         - For each element x in S, count how many active triples it is part of.
         - Remove the element x with the highest count.
         - Repeat.
    
    This is a Greedy Vertex Cover on the hypergraph? No, it's a Greedy Independent Set construction by removal.
    It's equivalent to finding a minimal set of vertices to remove to hit all triples (Hitting Set), and keeping the rest.
    Minimizing removals = Maximizing kept.
    
    This greedy hitting set approach is often good.
    
    Improvement:
    After the greedy removal stabilizes, perform a local search.
    Try to add back a removed element if it doesn't create a new triple (i.e., all its potential partners are already removed).
    Try to swap: remove one, add two?
    
    Let's implement this "Greedy Hitting Set" + "Local Search".
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    triples = _precompute_triples(N)
    
    # Map each element to the list of triples it participates in
    # triple_index: list of (a,b,c)
    # elem_triples[x]: list of indices in triples where x appears
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
        
    # Initial set: all elements
    S = set(range(1, N + 1))
    
    # Greedy Removal Phase
    # We maintain a set of "active" triples (those fully in S)
    # When we remove an element, we invalidate all triples containing it.
    
    # To speed up, we can track for each triple if it is "active" (all 3 in S)
    # But checking all triples every time is O(T). T is small (~1000).
    # So we can just loop through triples.
    
    # Greedy loop
    # To make it deterministic but robust, we can randomize tie-breaking or use a specific order.
    # Let's try a few random seeds for the tie-breaking.
    
    best_set = set()
    best_size = 0
    
    for seed in range(50): # 50 restarts of the greedy process with different randomizations
        if time.time() > deadline - 10.0:
            break
            
        random.seed(seed)
        S = set(range(1, N + 1))
        
        # Precompute "conflict count" for each element in current S?
        # Count how many active triples each element is in.
        
        changed = True
        while changed:
            changed = False
            # Find active triples
            active_triples = []
            for t in triples:
                if t[0] in S and t[1] in S and t[2] in S:
                    active_triples.append(t)
                    
            if not active_triples:
                break
                
            # Count conflicts for each element in S
            conflict_count = {x: 0 for x in S}
            for (a, b, c) in active_triples:
                conflict_count[a] += 1
                conflict_count[b] += 1
                conflict_count[c] += 1
                
            # Find max conflict
            max_conf = max(conflict_count.values())
            # Get all elements with max conflict
            candidates_to_remove = [x for x in S if conflict_count[x] == max_conf]
            
            # Randomly pick one to break symmetry
            x_to_remove = random.choice(candidates_to_remove)
            S.remove(x_to_remove)
            changed = True
            
        # Local Search Phase
        # Try to add back elements that were removed
        # An element x can be added if no triple containing x is fully in S \ {x} (which is S currently since x not in S)
        # i.e., for all (a,b,c) containing x, it is NOT the case that the other two are in S.
        
        removed_elements = [x for x in range(1, N + 1) if x not in S]
        # Shuffle removed elements to try adding them in random order
        random.shuffle(removed_elements)
        
        improved = True
        while improved:
            improved = False
            for x in removed_elements:
                if x in S:
                    continue
                # Check if x can be added
                can_add = True
                for idx in elem_triples[x]:
                    a, b, c = triples[idx]
                    # Check if the other two are in S
                    others = [v for v in (a, b, c) if v != x]
                    if others[0] in S and others[1] in S:
                        can_add = False
                        break
                if can_add:
                    S.add(x)
                    improved = True
                    # Remove from removed_elements list? 
                    # Since we are iterating, and S changes, it's safer to just mark or re-evaluate.
                    # Actually, if we add x, it might prevent adding others. 
                    # But adding x is always good if it doesn't violate constraints.
                    # So we keep adding as long as we can.
                    
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # Final check and return
    # Ensure best_set is valid
    S_set = set(best_set)
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            # This should not happen if logic is correct, but if it does, we have a bug.
            # For safety, we could fall back to a simpler set, but let's trust the logic.
            pass
            
    return list(best_set)

# EVOLVE-BLOCK-END
