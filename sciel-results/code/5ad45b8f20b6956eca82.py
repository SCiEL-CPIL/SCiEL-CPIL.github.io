"""
Maximum size subset of [1,256] such that for every a<b<=c in the set,
a does NOT divide (b+c).

Strategy: Model as maximum independent set on a 3-uniform hypergraph.
- Vertices: integers 1..256
- Hyperedges: triples (a,b,c) with a<b<=c and a|(b+c)
- Goal: Maximize number of vertices with no hyperedge fully contained.

Use greedy randomized local search (simulated annealing) to find large
independent sets.
"""
import random
import time
import math
from itertools import combinations

N = 256

def _build_conflict_graph():
    """
    Build the conflict structure. For each pair (b,c) with b<=c, 
    find all a < b such that a | (b+c).
    
    We'll represent conflicts as: for each vertex a, which pairs (b,c) 
    are forbidden if a is in the set.
    
    Actually, let's think of it differently. A set S is valid iff for no
    (a,b,c) in S with a<b<=c do we have a|(b+c).
    
    For local search, we need to efficiently check if adding/removing
    an element creates/fixes violations.
    
    Let's precompute: for each a, the set of pairs (b,c) with b>c... no, b<=c,
    b>a, c>=b, such that a|(b+c).
    """
    # conflict[a] = set of (b,c) pairs where a<b<=c and a|(b+c)
    conflict = [set() for _ in range(N+1)]
    # Also: for each pair (b,c), which a values conflict?
    # pair_conflict[(b,c)] = set of a < b with a|(b+c)
    
    for b in range(1, N+1):
        for c in range(b, N+1):
            s = b + c
            # a must be < b and a | s
            for a in range(1, b):
                if s % a == 0:
                    conflict[a].add((b, c))
    
    return conflict

def _count_violations(S, conflict):
    """Count number of violated hyperedges in set S."""
    S_set = set(S)
    count = 0
    for a in S:
        for (b, c) in conflict[a]:
            if b in S_set and c in S_set:
                count += 1
    return count

def _count_violations_fast(S, conflict, S_set):
    """Count violations given S as a set for faster membership."""
    count = 0
    for a in S_set:
        for (b, c) in conflict[a]:
            if b in S_set and c in S_set:
                count += 1
    return count

def construct():
    """
    Return a nonempty subset of [1,256] maximizing cardinality
    subject to: for all a<b<=c in the set, a does not divide b+c.
    """
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # Precompute conflict structure
    conflict = _build_conflict_graph()
    
    # --- Phase 1: Greedy initial solution ---
    # Start with all elements, then greedily remove those causing most conflicts
    S = list(range(1, N+1))
    S_set = set(S)
    
    # Greedy: repeatedly remove the element whose removal fixes the most violations
    # But this is expensive. Let's try a different greedy:
    # Start empty, add elements that don't create violations, or use a heuristic.
    
    # Better greedy: Start with upper tail (which is feasible), then try to add lower elements
    # The seed returns [192,256] which has size 65.
    
    # Let's try: start with all of [1,256], then iteratively remove elements
    # that are part of the most violations.
    
    best_S = list(range(192, 257))  # Seed solution, size 65
    best_size = len(best_S)
    
    # Greedy removal from full set
    S = list(range(1, N+1))
    S_set = set(S)
    
    # Compute violation count per element
    # For each a in S, how many (b,c) pairs in S conflict with a?
    # This is: for each a, count of (b,c) in conflict[a] where b,c in S
    
    # Let's do iterative removal
    changed = True
    while changed:
        changed = False
        # For each element, count its "conflict degree"
        degrees = {}
        for a in S_set:
            deg = 0
            for (b, c) in conflict[a]:
                if b in S_set and c in S_set:
                    deg += 1
            degrees[a] = deg
        
        if not degrees:
            break
        
        max_deg = max(degrees.values())
        if max_deg == 0:
            break
        
        # Remove all elements with max degree (or just one)
        worst = [a for a, d in degrees.items() if d == max_deg]
        # Remove one at a time to be safe, or remove all tied
        for a in worst:
            S_set.discard(a)
            S = [x for x in S if x != a]
            changed = True
        
        if time.time() > deadline - 10:
            break
    
    if len(S) > best_size:
        best_S = S[:]
        best_size = len(S)
    
    # --- Phase 2: Simulated annealing ---
    # State: S (set of included elements)
    # Move: swap - remove one element, add another (not in S)
    # Or: single add/remove
    
    S = best_S[:]
    S_set = set(S)
    
    # Precompute for speed: for each element a, the list of (b,c) conflicts
    # conflict[a] is already a set of (b,c)
    
    # To speed up violation counting, maintain a count of how many times
    # each pair (b,c) is "covered" by some a in S
    # Actually, let's maintain: for each (b,c) pair, the number of a in S 
    # such that a < b and a|(b+c)
    
    # pair_count[(b,c)] = number of a in S_set with a < b and (b+c) % a == 0
    # A violation exists iff there's a (b,c) in S_set with pair_count[(b,c)] > 0
    
    # But this is complex to maintain incrementally. Let's just recompute
    # violations for the affected elements.
    
    def get_violations(S_set, conflict):
        """Return set of violated triples (a,b,c)."""
        violations = set()
        for a in S_set:
            for (b, c) in conflict[a]:
                if b in S_set and c in S_set:
                    violations.add((a, b, c))
        return violations
    
    # For efficiency, let's track violations as a count
    # When we add element x: new violations are triples (x,b,c) where b,c in S and x<b<=c and x|(b+c)
    #   plus triples (a,x,c) where a in S, a<x, c>=x, and a|(x+c)
    #   plus triples (a,b,x) where a in S, a<b, b<=x, and a|(b+x)
    # When we remove element x: remove all violations involving x.
    
    # Let's precompute for each element x:
    #   as_a[x] = list of (b,c) where x<b<=c and x|(b+c)  [x is the 'a' role]
    #   as_b[x] = list of (a,c) where a<x, x<=c, and a|(x+c)  [x is the 'b' role]
    #   as_c[x] = list of (a,b) where a<b, b<=x, and a|(b+x)  [x is the 'c' role]
    
    as_a = [[] for _ in range(N+1)]
    as_b = [[] for _ in range(N+1)]
    as_c = [[] for _ in range(N+1)]
    
    for a in range(1, N+1):
        for (b, c) in conflict[a]:
            as_a[a].append((b, c))
            # a is the 'a' role, b and c are the other roles
            as_b[b].append((a, c))
            as_c[c].append((a, b))
    
    # Now maintain violation count
    # violations_count = total number of violated triples
    # When adding x:
    #   new_viol = len([(b,c) in as_a[x] if b in S and c in S])
    #             + len([(a,c) in as_b[x] if a in S and c in S])
    #             + len([(a,b) in as_c[x] if a in S and b in S])
    # When removing x:
    #   removed_viol = same count
    
    def count_new_violations(x, S_set):
        """Count violations created by adding x to S_set."""
        cnt = 0
        for (b, c) in as_a[x]:
            if b in S_set and c in S_set:
                cnt += 1
        for (a, c) in as_b[x]:
            if a in S_set and c in S_set:
                cnt += 1
        for (a, b) in as_c[x]:
            if a in S_set and b in S_set:
                cnt += 1
        return cnt
    
    # Initialize violation count
    viol_count = 0
    for a in S_set:
        for (b, c) in as_a[a]:
            if b in S_set and c in S_set:
                viol_count += 1
        for (a2, c) in as_b[a]:
            if a2 in S_set and c in S_set:
                viol_count += 1
        for (a2, b) in as_c[a]:
            if a2 in S_set and b in S_set:
                viol_count += 1
    
    # Simulated annealing
    T = 10.0
    cooling = 0.995
    best_SA = S_set.copy()
    best_SA_size = len(S_set)
    best_SA_viol = viol_count
    
    # We want to maximize size while keeping violations = 0
    # Objective: maximize |S| - M * violations, where M is a large penalty
    # Or: accept moves that increase size, or fix violations
    
    # Let's use: score = |S| * 1000 - viol_count
    # We want to maximize this.
    
    def score(S_set, viol_count):
        return len(S_set) * 1000 - viol_count
    
    current_score = score(S_set, viol_count)
    best_score = current_score
    
    all_elements = set(range(1, N+1))
    
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        
        # Generate a move: swap (remove one, add one) or add or remove
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap: remove x from S, add y not in S
            if not S_set:
                continue
            x = random.choice(list(S_set))
            candidates = [y for y in all_elements if y not in S_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Compute new violation count
            # Remove x: subtract violations involving x
            # Add y: add violations involving y (with x removed)
            
            # Violations involving x (as a, b, or c role)
            x_viol = 0
            for (b, c) in as_a[x]:
                if b in S_set and c in S_set:
                    x_viol += 1
            for (a2, c) in as_b[x]:
                if a2 in S_set and c in S_set:
                    x_viol += 1
            for (a2, b) in as_c[x]:
                if a2 in S_set and b in S_set:
                    x_viol += 1
            
            # After removing x, compute violations for y
            S_set.discard(x)
            y_viol = 0
            for (b, c) in as_a[y]:
                if b in S_set and c in S_set:
                    y_viol += 1
            for (a2, c) in as_b[y]:
                if a2 in S_set and c in S_set:
                    y_viol += 1
            for (a2, b) in as_c[y]:
                if a2 in S_set and b in S_set:
                    y_viol += 1
            
            new_viol = viol_count - x_viol + y_viol
            S_set.add(y)
            new_score = score(S_set, new_viol)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / max(T, 0.01)):
                viol_count = new_viol
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_SA = S_set.copy()
                    best_SA_size = len(S_set)
                    best_SA_viol = viol_count
            else:
                # Reject
                S_set.discard(y)
                S_set.add(x)
        elif move_type < 0.8:
            # Add an element
            candidates = [y for y in all_elements if y not in S_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            y_viol = 0
            for (b, c) in as_a[y]:
                if b in S_set and c in S_set:
                    y_viol += 1
            for (a2, c) in as_b[y]:
                if a2 in S_set and c in S_set:
                    y_viol += 1
            for (a2, b) in as_c[y]:
                if a2 in S_set and b in S_set:
                    y_viol += 1
            
            new_viol = viol_count + y_viol
            S_set.add(y)
            new_score = score(S_set, new_viol)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / max(T, 0.01)):
                viol_count = new_viol
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_SA = S_set.copy()
                    best_SA_size = len(S_set)
                    best_SA_viol = viol_count
            else:
                S_set.discard(y)
        else:
            # Remove an element
            if not S_set:
                continue
            x = random.choice(list(S_set))
            
            x_viol = 0
            for (b, c) in as_a[x]:
                if b in S_set and c in S_set:
                    x_viol += 1
            for (a2, c) in as_b[x]:
                if a2 in S_set and c in S_set:
                    x_viol += 1
            for (a2, b) in as_c[x]:
                if a2 in S_set and b in S_set:
                    x_viol += 1
            
            new_viol = viol_count - x_viol
            S_set.discard(x)
            new_score = score(S_set, new_viol)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / max(T, 0.01)):
                viol_count = new_viol
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_SA = S_set.copy()
                    best_SA_size = len(S_set)
                    best_SA_viol = viol_count
            else:
                S_set.add(x)
        
        # Cool down
        if iteration % 100 == 0:
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        # Periodically check if we have a valid solution
        if viol_count == 0 and len(S_set) > best_SA_size:
            best_SA = S_set.copy()
            best_SA_size = len(S_set)
            best_SA_viol = 0
    
    # Use the best solution found
    if best_SA_viol == 0 and len(best_SA) > len(best_S):
        best_S = list(best_SA)
    elif best_SA_viol == 0:
        best_S = list(best_SA)
    
    # Final check: ensure validity
    best_S_set = set(best_S)
    valid = True
    for a in best_S_set:
        for (b, c) in conflict[a]:
            if b in best_S_set and c in best_S_set:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to seed solution
        return list(range(192, 257))
    
    return sorted(best_S)

