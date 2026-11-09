"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import deque

def _build_triple_index(n):
    """Build mapping from (a,b,c) to index in lex-ordered triple list."""
    triple_to_idx = {}
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triple_to_idx[(a, b, c)] = idx
                idx += 1
    return triple_to_idx

def _build_constraint_graphs(n, triple_to_idx):
    """
    Build constraint propagation graphs.
    
    For red K4^(3) constraint:
    - Each triple (a,b,c) participates in K4s with each d not in {a,b,c}
    - K4 = {a,b,c,d} has triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
    - If 3 of these 4 triples are red, the 4th must be blue (forced)
    
    For blue K5^(3) constraint:
    - Each triple (a,b,c) participates in K5s with each pair {d,e} not in {a,b,c}
    - K5 = {a,b,c,d,e} has 10 triples
    - If 9 of these 10 triples are blue, the 10th must be red (forced)
    
    Returns:
    - red_forced: dict mapping triple_idx -> list of (other_triple_idx, condition)
      where condition indicates what the other triple must be
    - blue_forced: similar for blue
    """
    # For each triple, precompute which K4s it's part of
    # red_k4_partners[t_idx] = list of (d, [t1, t2, t3]) where t1,t2,t3 are the other 3 triples in the K4
    red_k4_partners = [[] for _ in range(len(triple_to_idx))]
    # For blue K5, we need to track partial K5s
    # blue_k5_partners[t_idx] = list of ((d,e), [t1,...,t9]) where t1..t9 are other 9 triples
    
    # Build red K4 structure
    for (a, b, c), t_idx in triple_to_idx.items():
        for d in range(n):
            if d == a or d == b or d == c:
                continue
            # K4 = {a,b,c,d}, sorted
            verts = sorted([a, b, c, d])
            v0, v1, v2, v3 = verts
            # The 4 triples in this K4
            k4_triples = [
                triple_to_idx[(v0, v1, v2)],
                triple_to_idx[(v0, v1, v3)],
                triple_to_idx[(v0, v2, v3)],
                triple_to_idx[(v1, v2, v3)]
            ]
            # Find which index in k4_triples corresponds to (a,b,c)
            our_idx = k4_triples.index(t_idx)
            others = [k4_triples[i] for i in range(4) if i != our_idx]
            red_k4_partners[t_idx].append((d, others))
    
    # For blue K5, we'll handle differently - check incrementally
    # Precompute for each triple, the list of vertex pairs that could form a K5 with it
    # But this is expensive. Instead, we'll use a different approach:
    # Track for each 5-set how many blue triples it has
    
    return red_k4_partners

def _count_violations_fast(n, colors, triple_to_idx, red_k4_partners):
    """
    Count violations efficiently.
    Returns (red_violations, blue_violations)
    """
    red_violations = 0
    blue_violations = 0
    
    # Count red K4 violations
    # For each K4, check if all 4 triples are red
    # We can iterate over all 4-sets
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        t1 = triple_to_idx[(a, b, c)]
        t2 = triple_to_idx[(a, b, d)]
        t3 = triple_to_idx[(a, c, d)]
        t4 = triple_to_idx[(b, c, d)]
        if colors[t1] == 0 and colors[t2] == 0 and colors[t3] == 0 and colors[t4] == 0:
            red_violations += 1
    
    # Count blue K5 violations
    # For each 5-set, check if all 10 triples are blue
    for quint in combinations(range(n), 5):
        all_blue = True
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    t = triple_to_idx[(quint[i], quint[j], quint[k])]
                    if colors[t] != 1:
                        all_blue = False
                        break
                if not all_blue:
                    break
            if not all_blue:
                break
        if all_blue:
            blue_violations += 1
    
    return red_violations, blue_violations

def _incremental_violation_check(n, colors, triple_to_idx, red_k4_partners, 
                                  changed_idx, old_color, new_color):
    """
    Check if changing colors[changed_idx] from old_color to new_color
    creates or removes violations.
    Returns (delta_red, delta_blue) - change in violation counts.
    """
    delta_red = 0
    delta_blue = 0
    
    a, b, c = None, None, None
    # Find the triple corresponding to changed_idx
    # We need reverse mapping
    # For efficiency, let's precompute idx_to_triple
    # But for now, iterate (this is called frequently, so we need it fast)
    
    # Actually, let's pass the triple directly
    # Reconstruct: we need to know which triples are affected
    
    # For red K4: changing one triple affects all K4s containing that triple
    # For each such K4, check if it was a violation before and after
    
    # This is complex. Let's use a simpler approach for now:
    # Just do full recount for the affected structures
    
    # For red K4 violations involving the changed triple:
    # The changed triple is part of (n-3) K4s
    # For each, check the other 3 triples
    
    # Find which vertices form the changed triple
    # We need idx_to_triple mapping
    # Let's compute it once and pass it around
    
    return delta_red, delta_blue

def _is_valid_fast(n, colors, triple_to_idx):
    """Fast validity check."""
    # Check red K4
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[triple_to_idx[(a,b,c)]] == 0 and
            colors[triple_to_idx[(a,b,d)]] == 0 and
            colors[triple_to_idx[(a,c,d)]] == 0 and
            colors[triple_to_idx[(b,c,d)]] == 0):
            return False
    
    # Check blue K5
    for quint in combinations(range(n), 5):
        all_blue = True
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    if colors[triple_to_idx[(quint[i], quint[j], quint[k])]] != 1:
                        all_blue = False
                        break
                if not all_blue:
                    break
            if not all_blue:
                break
        if all_blue:
            return False
    
    return True

def _constraint_propagation_search(n, timeout=5.0):
    """
    Use constraint propagation to find a valid coloring.
    
    Strategy:
    1. Start with a random or algebraic initial coloring
    2. Identify violations
    3. Use constraint propagation to fix violations:
       - If a K4 has 3 red triples, the 4th must be blue
       - If a K5 has 9 blue triples, the 10th must be red
    4. Repeat until no violations or timeout
    """
    start = time.time()
    triple_to_idx = _build_triple_index(n)
    n_triples = len(triple_to_idx)
    
    # Initialize with a random coloring biased towards blue
    # (since blue K5 is harder to satisfy - need all 10 blue)
    colors = [random.choice([0, 1]) for _ in range(n_triples)]
    
    # Build idx_to_triple for reverse lookup
    idx_to_triple = {v: k for k, v in triple_to_idx.items()}
    
    # Build K4 structure: for each triple, which K4s does it participate in?
    # k4_of_triple[t_idx] = list of (other_t1, other_t2, other_t3)
    k4_of_triple = [[] for _ in range(n_triples)]
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        t1 = triple_to_idx[(a,b,c)]
        t2 = triple_to_idx[(a,b,d)]
        t3 = triple_to_idx[(a,c,d)]
        t4 = triple_to_idx[(b,c,d)]
        k4_of_triple[t1].append((t2, t3, t4))
        k4_of_triple[t2].append((t1, t3, t4))
        k4_of_triple[t3].append((t1, t2, t4))
        k4_of_triple[t4].append((t1, t2, t3))
    
    # Build K5 structure: for each triple, which K5s does it participate in?
    # k5_of_triple[t_idx] = list of (other_t1, ..., other_t9)
    # This is expensive: each triple is in C(n-3, 2) K5s
    # For n=34, that's C(31,2) = 465 K5s per triple
    # Total K5s: C(34,5) = 278256
    # Each K5 has 10 triples, so total triple-K5 incidences: 2782560
    # This is manageable but memory-intensive
    
    # Instead of precomputing all K5s, let's use a different approach:
    # For blue K5 constraint, when we have 9 blue triples in a K5,
    # we can find the 10th by checking which triple is missing
    
    # Let's use a hybrid approach:
    # 1. Handle red K4 constraints via propagation (fast)
    # 2. Handle blue K5 constraints via local search (slower but necessary)
    
    best_colors = colors[:]
    best_violations = float('inf')
    
    # Count initial violations
    red_v, blue_v = _count_violations_fast(n, colors, triple_to_idx, None)
    best_violations = red_v + blue_v
    
    if best_violations == 0:
        return {'n': n, 'colors': colors}
    
    # Iterative improvement with constraint propagation
    for iteration in range(10000):
        if time.time() > start + timeout:
            break
        
        # Check for red K4 violations and propagate
        made_progress = False
        
        # For each K4, if 3 are red, force the 4th to be blue
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            
            red_count = sum(1 for t in [t1, t2, t3, t4] if colors[t] == 0)
            
            if red_count == 3:
                # Find the blue one and force it to... wait, if 3 are red,
                # the 4th must be blue to avoid red K4
                # But if the 4th is already blue, no violation
                # If the 4th is red, we have a violation - need to fix
                for t in [t1, t2, t3, t4]:
                    if colors[t] == 0:
                        continue
                    # This triple is blue, but we have 3 red - actually this is fine
                    # The violation is when ALL 4 are red
                    pass
            
            if red_count == 4:
                # Violation! Need to change one to blue
                # Choose the one that causes least additional violations
                best_change = -1
                best_cost = float('inf')
                for t in [t1, t2, t3, t4]:
                    # If we change t to blue, how many new violations?
                    # Check K5s containing t that might become problematic
                    # Simplified: just pick random
                    cost = random.random()
                    if cost < best_cost:
                        best_cost = cost
                        best_change = t
                
                if best_change >= 0:
                    colors[best_change] = 1
                    made_progress = True
        
        # Check for blue K5 violations
        for quint in combinations(range(n), 5):
            all_blue = True
            blue_count = 0
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = triple_to_idx[(quint[i], quint[j], quint[k])]
                        if colors[t] == 1:
                            blue_count += 1
                        else:
                            all_blue = False
                    if not all_blue:
                        break
                if not all_blue:
                    break
            
            if blue_count == 10:
                # Violation! Need to change one to red
                # Find a triple in this K5 to change
                changed = False
                for i in range(5):
                    if changed:
                        break
                    for j in range(i+1, 5):
                        if changed:
                            break
                        for k in range(j+1, 5):
                            t = triple_to_idx[(quint[i], quint[j], quint[k])]
                            if colors[t] == 1:
                                colors[t] = 0
                                changed = True
                                made_progress = True
                                break
        
        # If no progress, do random flips
        if not made_progress:
            # Random local search
            for _ in range(100):
                if time.time() > start + timeout:
                    break
                i = random.randint(0, n_triples - 1)
                old = colors[i]
                colors[i] = 1 - old
                
                # Quick check: does this create a new red K4?
                a, b, c = idx_to_triple[i]
                has_red_k4 = False
                for d in range(n):
                    if d == a or d == b or d == c:
                        continue
                    t1 = triple_to_idx[(a,b,c)]
                    # Check the K4 {a,b,c,d}
                    other_ts = []
                    for (x,y,z) in [(a,b,d), (a,c,d), (b,c,d)]:
                        other_ts.append(triple_to_idx[(x,y,z)])
                    
                    if colors[t1] == 0 and all(colors[t] == 0 for t in other_ts):
                        has_red_k4 = True
                        break
                
                if has_red_k4:
                    colors[i] = old  # Revert
                    continue
                
                # Check blue K5 (expensive, do less often)
                if random.random() < 0.1:
                    # Full check
                    red_v, blue_v = _count_violations_fast(n, colors, triple_to_idx, None)
                    if red_v + blue_v < best_violations:
                        best_violations = red_v + blue_v
                        best_colors = colors[:]
                        if best_violations == 0:
                            return {'n': n, 'colors': colors}
                    else:
                        colors[i] = old  # Revert if worse
    
    # Return best found
    red_v, blue_v = _count_violations_fast(n, best_colors, triple_to_idx, None)
    if red_v + blue_v == 0:
        return {'n': n, 'colors': best_colors}
    
    # Try the original colors too
    red_v, blue_v = _count_violations_fast(n, colors, triple_to_idx, None)
    if red_v + blue_v == 0:
        return {'n': n, 'colors': colors}
    
    return None

def construct():
    """
    Construct a valid two-coloring of triples on n vertices with no red K4^(3)
    and no blue K5^(3), maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_n = 0
    best_colors = None
    
    # Strategy 1: Try constraint propagation search for large n
    for n in range(34, 20, -1):
        if time.time() > deadline:
            break
        
        result = _constraint_propagation_search(n, timeout=min(5.0, deadline - time.time()))
        if result is not None:
            best_n = n
            best_colors = result['colors']
            break
    
    # Strategy 2: Try algebraic constructions
    if best_n < 34:
        for n in range(max(34, best_n + 1), 20, -1):
            if time.time() > deadline:
                break
            
            for m in range(2, n + 10):
                if time.time() > deadline:
                    break
                
                colors = []
                for a in range(n):
                    for b in range(a + 1, n):
                        for c in range(b + 1, n):
                            if (a + b + c) % m == 0:
                                colors.append(0)
                            else:
                                colors.append(1)
                
                # Fast validation
                triple_to_idx = _build_triple_index(n)
                if _is_valid_fast(n, colors, triple_to_idx):
                    if n > best_n:
                        best_n = n
                        best_colors = colors[:]
                    break
            
            if best_n == n:
                break
    
    # Strategy 3: More algebraic patterns
    if best_n < 34:
        for n in range(max(34, best_n + 1), 20, -1):
            if time.time() > deadline:
                break
            
            # Try: red if (a*b + b*c + c*a) % m == 0
            for m in range(2, n + 10):
                if time.time() > deadline:
                    break
                
                colors = []
                for a in range(n):
                    for b in range(a + 1, n):
                        for c in range(b + 1, n):
                            if (a * b + b * c + c * a) % m == 0:
                                colors.append(0)
                            else:
                                colors.append(1)
                
                triple_to_idx = _build_triple_index(n)
                if _is_valid_fast(n, colors, triple_to_idx):
                    if n > best_n:
                        best_n = n
                        best_colors = colors[:]
                    break
            
            if best_n == n:
                break
            
            # Try: red if (a+b+c) % m in {0, 1}
            for m in range(2, n + 10):
                if time.time() > deadline:
                    break
                
                colors = []
                for a in range(n):
                    for b in range(a + 1, n):
                        for c in range(b + 1, n):
                            if (a + b + c) % m in (0, 1):
                                colors.append(0)
                            else:
                                colors.append(1)
                
                triple_to_idx = _build_triple_index(n)
                if _is_valid_fast(n, colors, triple_to_idx):
                    if n > best_n:
                        best_n = n
                        best_colors = colors[:]
                    break
            
            if best_n == n:
                break
    
    if best_n > 0 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 
              for a in range(n) for b in range(a+1, n) for c in range(b+1, n)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
