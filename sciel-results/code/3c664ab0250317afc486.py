import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _independence_number(triples, n=16):
    """Compute the independence number of a 3-uniform hypergraph on n points.
    Returns the size of the largest subset containing no triple."""
    if not triples:
        return n
    # Precompute: for each point, which triples contain it
    triples = [tuple(t) for t in triples]
    # Branch and bound
    best = [0]
    
    def rec(available_mask, current_size):
        # available_mask: bitmask of points still eligible to be added
        if current_size + bin(available_mask).count('1') <= best[0]:
            return
        if available_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Pick a point to branch on (heuristic: pick one in most triples among available)
        # Try to find a good branching point
        # Simple: pick lowest set bit
        # Better: pick point that is in the most triples (to prune more)
        # For speed, just pick the lowest available point
        # Actually, let's pick a point that participates in a triple with other available points
        # to get good pruning
        pt = (available_mask & -available_mask).bit_length() - 1
        # Option 1: exclude pt
        rec(available_mask & ~(1 << pt), current_size)
        # Option 2: include pt -> must break all triples containing pt
        new_mask = available_mask & ~(1 << pt)
        for t in triples:
            if t[0] == pt or t[1] == pt or t[2] == pt:
                # This triple contains pt; to keep it from being fully in S,
                # we must remove at least one of the other two points
                # Heuristic: remove the one that is in more triples (or just remove both? No, remove one)
                # To be safe and simple: remove all other points in triples with pt? That's too aggressive.
                # Correct approach: we need to ensure no triple containing pt is fully present.
                # Since pt is present, for each triple {pt, x, y}, we need x not in S or y not in S.
                # This is a constraint. Simple conservative: remove x and y from available.
                # But that might be too aggressive (removes points unnecessarily).
                # Better: remove one of x,y. But which? For correctness of the bound,
                # we can remove both (safe but may underestimate). 
                # Actually removing both is SAFE (gives a valid independent set subset) but may miss the optimum.
                # For a correct alpha, we need to be careful.
                # Let me use a different approach: when including pt, the new available set must be
                # such that no triple is fully contained. The simplest correct move:
                # for each triple containing pt, remove at least one other member.
                # To keep it simple and correct, I'll remove all other members of all triples containing pt.
                # This gives a valid (possibly not maximal) extension, and the recursion will handle the rest.
                # Wait, that's wrong - removing all others might remove points that don't need removal.
                # Let me reconsider.
                pass
        # This simple approach is getting complicated. Let me use a cleaner method.
        pass
    
    # Let me use a cleaner branch and bound
    # Represent available as a bitmask
    # State: (mask of points that can still be added, current selected size)
    # But we also need to track which triples are "satisfied" (broken)
    # Alternative: use the fact that a set S is independent iff for every triple, S doesn't contain all 3.
    # Branch on points. When we include a point, we don't immediately remove others;
    # instead, we check at the end. But that's exponential.
    
    # Better: use a standard hypergraph independence B&B.
    # State: (candidate_set_mask, forbidden_triples_count)
    # Too complex. Let me just do: try all subsets via B&B on points,
    # with the constraint that we track for each triple how many of its points are selected.
    
    # For 16 points and modest number of triples, let me do:
    # B&B where we process points in order, and maintain a count per triple.
    # If any triple count reaches 3, prune.
    
    triples_list = triples
    # Count per triple: use an array
    # Map each triple to an index
    triple_idx = {}
    for i, t in enumerate(triples_list):
        triple_idx[t] = i
    ntrip = len(triples_list)
    
    # For each point, list of triple indices it belongs to
    pt_triples = [[] for _ in range(n)]
    for t in triples_list:
        for p in t:
            pt_triples[p].append(triple_idx[t])
    
    best_val = [0]
    
    def rec2(pos, selected_mask, triple_counts):
        # pos: next point to consider
        # selected_mask: points selected so far
        # triple_counts: list of counts (how many points of each triple are selected)
        # Prune: if any count >= 3, invalid (shouldn't happen if we check before adding)
        # Upper bound: selected_count + (n - pos)
        sel_count = bin(selected_mask).count('1')
        if sel_count + (n - pos) <= best_val[0]:
            return
        if pos == n:
            if sel_count > best_val[0]:
                best_val[0] = sel_count
            return
        # Option 1: don't select point pos
        rec2(pos + 1, selected_mask, triple_counts)
        # Option 2: select point pos, if it doesn't complete any triple
        ok = True
        for ti in pt_triples[pos]:
            if triple_counts[ti] + 1 >= 3:
                ok = False
                break
        if ok:
            # Select pos
            new_mask = selected_mask | (1 << pos)
            # Update counts
            new_counts = triple_counts[:]
            for ti in pt_triples[pos]:
                new_counts[ti] += 1
            rec2(pos + 1, new_mask, new_counts)
    
    rec2(0, 0, [0] * ntrip)
    return best_val[0]

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s
    
    n = 16
    # All possible triples
    all_triples = list(combinations(range(n), 3))
    
    # We want to build a linear triple system minimizing independence number.
    # Strategy: backtracking with heuristic ordering.
    # Start with a good initial solution, then try to improve.
    
    # Initial: the seed solution (5 disjoint triples), alpha = 11
    best_system = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    best_alpha = _independence_number(best_system, n)
    
    # Now try to build better systems via backtracking.
    # We'll try to add more triples to "break up" large independent sets.
    # Heuristic: at each step, try adding a triple that is linear with existing ones
    # and reduces the independence number the most.
    
    # Precompute compatibility: for speed, we'll check linearity on the fly.
    
    # Backtracking search: build a list of triples.
    # To keep it manageable, limit the number of triples we try to add.
    
    # Let's try a randomized greedy + local search approach:
    # 1. Start with a random linear system (or the seed).
    # 2. Try to add triples greedily, choosing the one that most reduces alpha.
    # 3. If stuck, backtrack.
    
    # Actually, let me do a more structured approach:
    # Generate a random ordering of triples, then greedily add if linear.
    # Try multiple random orderings, keep the best alpha.
    
    random.seed(42)
    
    def greedy_build(order):
        """Given an ordering of candidate triples, greedily add if linear."""
        system = []
        for t in order:
            # Check if t is linear with all in system
            linear = True
            for s in system:
                if len(set(t) & set(s)) > 1:
                    linear = False
                    break
            if linear:
                system.append(list(t))
        return system
    
    # Try many random orderings
    # But also bias toward triples that cover uncovered points and create interactions.
    
    # A better heuristic: prioritize triples that share exactly one point with existing triples
    # (to create a connected, interacting structure) rather than disjoint ones.
    
    def heuristic_greedy():
        system = []
        remaining = set(all_triples)
        # Start with a random triple
        if not remaining:
            return system
        first = random.choice(list(remaining))
        system.append(list(first))
        remaining.discard(first)
        
        while remaining:
            # For each candidate, check linearity and compute a score
            best_candidate = None
            best_score = -1
            # Sample a subset to keep it fast
            candidates = random.sample(list(remaining), min(50, len(remaining)))
            for t in candidates:
                linear = True
                for s in system:
                    if len(set(t) & set(s)) > 1:
                        linear = False
                        break
                if not linear:
                    continue
                # Score: prefer triples that interact (share 1 point) with existing
                # and cover new points
                shared_points = 0
                new_points = 0
                for s in system:
                    common = set(t) & set(s)
                    shared_points += len(common)
                t_set = set(t)
                covered = set()
                for s in system:
                    covered |= set(s)
                new_points = len(t_set - covered)
                # We want some interaction but also coverage
                score = new_points * 2 + shared_points
                if score > best_score:
                    best_score = score
                    best_candidate = t
            if best_candidate is None:
                # Try all remaining
                for t in remaining:
                    linear = True
                    for s in system:
                        if len(set(t) & set(s)) > 1:
                            linear = False
                            break
                    if linear:
                        best_candidate = t
                        break
            if best_candidate is None:
                break
            system.append(list(best_candidate))
            remaining.discard(best_candidate)
        return system
    
    # Run multiple heuristic greedy builds
    for _ in range(200):
        if time.time() > deadline:
            break
        system = heuristic_greedy()
        if not system:
            continue
        alpha = _independence_number(system, n)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = system
            # If we get a good one, try to improve further
            if best_alpha <= 8:
                # Try local search: add more triples
                # (skip for now, keep it simple)
                pass
    
    # Also try the seed and a few structured approaches
    # Structured: use a near-Steiner system
    # For now, return the best found
    return best_system

# EVOLVE-BLOCK-END
