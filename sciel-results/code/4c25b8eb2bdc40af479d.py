"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    """
    Find a minimal maximal no-isosceles set in [0, N)^2.
    
    Strategy: 
    1. Pre-compute conflict structures for fast independence checking.
    2. Use a fast greedy algorithm with randomized ordering to generate candidate sets.
    3. Use an efficient local search (removal) to minimize the set size while maintaining maximality.
    """
    points = [(x, y) for x in range(N) for y in range(N)]
    M = len(points)
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Pre-compute for each pair (i, j), the list of indices k such that {i, j, k} is isosceles
    # bad_points[i, j] = list of k
    # Note: i != j. k can be equal to i or j? 
    # If k=i, dist(i,i)=0, dist(i,j)=d, dist(j,i)=d. 0, d, d. Isosceles? 
    # Usually degenerate triangles are considered isosceles in this context? 
    # The prompt says "no-isosceles set". 
    # If k=i, the points are not distinct. The set S contains distinct points.
    # When checking if p can be added to S, p is distinct from all in S.
    # So we only care about distinct triples.
    
    bad_points = [[None] * M for _ in range(M)]
    
    for i in range(M):
        for j in range(i + 1, M):
            pi = points[i]
            pj = points[j]
            d2 = (pi[0] - pj[0])**2 + (pi[1] - pj[1])**2
            if d2 == 0: continue
            
            # Find all k != i, j such that {pi, pj, pk} is isosceles
            # Conditions:
            # 1. dist(pi, pk) == dist(pi, pj)  => pk on circle center pi radius sqrt(d2)
            # 2. dist(pj, pk) == dist(pi, pj)  => pk on circle center pj radius sqrt(d2)
            # 3. dist(pi, pk) == dist(pj, pk)  => pk on perpendicular bisector of pi-pj
            
            candidates = []
            for k in range(M):
                if k == i or k == j:
                    continue
                pk = points[k]
                d_ik = (pi[0] - pk[0])**2 + (pi[1] - pk[1])**2
                d_jk = (pj[0] - pk[0])**2 + (pj[1] - pk[1])**2
                if d_ik == d2 or d_jk == d2 or d_ik == d_jk:
                    candidates.append(k)
            bad_points[i][j] = candidates
            bad_points[j][i] = candidates

    def is_isosceles_set(S):
        # S is list of indices
        # Check if any triple is isosceles
        # Using bad_points: for each pair i,j in S, if any k in S is in bad_points[i][j], then False
        # But this is O(|S|^3) effectively if we check all k in S.
        # Better: just check pairs.
        # Actually, we maintain conflict counts during construction.
        pass

    best_set = None
    best_size = float('inf')
    
    # Run multiple trials with different random seeds/orderings
    for trial in range(20):
        random.seed(trial)
        indices = list(range(M))
        random.shuffle(indices)
        
        S = []
        # conflict_count[k] = number of pairs (a,b) in S such that k is in bad_points[a][b]
        conflict_count = np.zeros(M, dtype=int)
        
        for idx in indices:
            # If conflict_count[idx] == 0, we can add idx
            if conflict_count[idx] == 0:
                S.append(idx)
                # Update conflict counts for all other points
                # For each existing point j in S, the pair (idx, j) is formed.
                # For each k in bad_points[idx][j], increment conflict_count[k]
                for j in S[:-1]: # S[:-1] because idx is the last one added
                    for k in bad_points[idx][j]:
                        conflict_count[k] += 1
        
        # Now S is a maximal independent set (greedy)
        # Convert to list of points for return later, but work with indices for speed
        S_set = set(S)
        
        # Local Search: Remove non-essential points
        # To check if p is essential:
        # Is there a q not in S such that q is independent from S \ {p}?
        # q is independent from S \ {p} iff conflict_count(q, S \ {p}) == 0.
        # We have conflict_count(q, S).
        # conflict_count(q, S \ {p}) = conflict_count(q, S) - (1 if p is part of a conflicting pair for q else 0)
        # p is part of a conflicting pair for q if there exists a in S \ {p} such that q in bad_points[p][a].
        # Let's precompute for each p, the set of q's for which p "blocks" q?
        # Or just recalculate for candidate removals.
        
        changed = True
        while changed:
            changed = False
            # Try removing each point in S
            # Shuffle to avoid bias
            S_list = list(S_set)
            random.shuffle(S_list)
            
            for p in S_list:
                if p not in S_set:
                    continue
                
                # Check if p is essential
                # Find if there is any q not in S_set such that 
                # (S_set - {p}) U {q} is independent.
                # This is equivalent to: conflict_count(q, S_set - {p}) == 0.
                
                # We can iterate q not in S_set.
                # To speed up, we can precompute for each p, the set of q's that are "freed" by removing p?
                # No, just check.
                
                is_essential = False
                # Optimization: Only check q's that are "close" or have low conflict count?
                # Actually, just iterate all q not in S.
                
                # To make this faster, we can precompute for each p, the list of q's such that 
                # q is in bad_points[p][a] for some a in S.
                # These are the only q's whose conflict count changes when p is removed.
                # If q is not in this list, conflict_count(q, S\{p}) == conflict_count(q, S).
                # If conflict_count(q, S) > 0, it's still > 0.
                # So we only need to check q's in the "affected" set.
                
                affected_q = set()
                for a in S_set:
                    if a == p: continue
                    for q in bad_points[p][a]:
                        affected_q.add(q)
                
                for q in affected_q:
                    if q in S_set:
                        continue
                    # Check if q is independent from S \ {p}
                    # conflict_count(q, S) is the count for S.
                    # We need to know how many times p contributed to this count.
                    # p contributed if there is a in S\{p} such that q in bad_points[p][a].
                    # We can just recount for this specific q and p.
                    # Count pairs (a, b) in S\{p} such that q in bad_points[a][b].
                    
                    # Since affected_q is small, we can just check directly.
                    valid = True
                    for a in S_set:
                        if a == p: continue
                        # Check if q forms isosceles with a and any b in S\{p}
                        # This is still O(|S|^2) per q.
                        # But |affected_q| is small.
                        pass
                    
                    # Let's just do the full check for q against S\{p}
                    # S\{p} size is ~10-20.
                    # Check if q is independent from S\{p}:
                    # For all pairs a,b in S\{p}, is q in bad_points[a][b]?
                    # If no, then q can be added.
                    
                    # We can optimize this by precomputing conflict_count for S\{p} on the fly?
                    # No, just iterate.
                    
                    independent = True
                    # To avoid O(|S|^2) for each q, we can use the fact that 
                    # q is in affected_q, so it conflicts with p and some a.
                    # But it might conflict with other pairs too.
                    
                    # Simple check:
                    for i, a in enumerate(S_set):
                        if a == p: continue
                        for b in S_set:
                            if b == p or b <= a: continue # avoid duplicates, though bad_points is symmetric
                            if q in bad_points[a][b]:
                                independent = False
                                break
                        if not independent:
                            break
                    
                    if independent:
                        is_essential = True
                        break
                
                if not is_essential:
                    S_set.remove(p)
                    changed = True
    
    if best_set is None:
        # Fallback to first trial result if no "best" tracked properly in loop above
        # The loop above doesn't update best_set. Let's fix that.
        pass
        
    # Re-run logic to track best
    best_set = None
    best_size = float('inf')
    
    for trial in range(50):
        random.seed(trial)
        indices = list(range(M))
        random.shuffle(indices)
        
        S = []
        conflict_count = np.zeros(M, dtype=int)
        
        for idx in indices:
            if conflict_count[idx] == 0:
                S.append(idx)
                for j in S[:-1]:
                    for k in bad_points[idx][j]:
                        conflict_count[k] += 1
        
        S_set = set(S)
        
        # Local Search
        changed = True
        while changed:
            changed = False
            S_list = list(S_set)
            random.shuffle(S_list)
            for p in S_list:
                if p not in S_set:
                    continue
                is_essential = False
                # Find affected q's
                affected_q = set()
                for a in S_set:
                    if a == p: continue
                    affected_q.update(bad_points[p][a])
                
                for q in affected_q:
                    if q in S_set:
                        continue
                    # Check if q is independent from S \ {p}
                    independent = True
                    # Optimize: Only check pairs involving q?
                    # No, q must not form isosceles with ANY pair in S\{p}.
                    # We can check: for each a in S\{p}, is q in bad_points[a][b] for any b in S\{p}, b>a?
                    # This is O(|S|^2).
                    # Given |S| is small, it's fine.
                    for a in S_set:
                        if a == p: continue
                        for b in S_set:
                            if b == p or b <= a: continue
                            if q in bad_points[a][b]:
                                independent = False
                                break
                        if not independent:
                            break
                    if independent:
                        is_essential = True
                        break
                
                if not is_essential:
                    S_set.remove(p)
                    changed = True
        
        if len(S_set) < best_size:
            best_size = len(S_set)
            best_set = S_set.copy()
            
    if best_set is None:
        return []
        
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
