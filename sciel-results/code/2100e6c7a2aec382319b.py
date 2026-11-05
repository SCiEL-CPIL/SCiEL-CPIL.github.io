import random
import time
import itertools
import math
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute blocking relationships
# BLOCKERS[p] is a set of pairs (u, v) with u < v such that {p, u, v} is isosceles.
# If u and v are in S, then p is blocked.
BLOCKERS = [set() for _ in range(N_PTS)]

def _precompute_blockers():
    i_idx, j_idx = np.triu_indices(N_PTS, k=1)
    for p in range(N_PTS):
        dp = dist2[p, :]
        dp_i = dp[i_idx]
        dp_j = dp[j_idx]
        dij = dist2[i_idx, j_idx]
        # p is blocked by (u, v) if d(p,u)==d(p,v) or d(p,u)==d(u,v) or d(p,v)==d(u,v)
        cond = (dp_i == dp_j) | (dp_i == dij) | (dp_j == dij)
        bad_indices = np.where(cond)[0]
        for k in bad_indices:
            u, v = i_idx[k], j_idx[k]
            if u > v:
                u, v = v, u
            BLOCKERS[p].add((u, v))

_start_time = time.time()
_precompute_blockers()

def _is_blocked(p, S_set):
    """Check if point p is blocked by set S_set."""
    for u, v in BLOCKERS[p]:
        if u in S_set and v in S_set:
            return True
    return False

def _get_unblocked(S_set):
    """Get the set of points not in S_set and not blocked by S_set."""
    unblocked = set()
    for p in range(N_PTS):
        if p not in S_set and not _is_blocked(p, S_set):
            unblocked.add(p)
    return unblocked

def _greedy_blocking_power(deadline):
    """
    Greedy Set Cover approach:
    Start with S = empty, U = all points.
    While U is not empty:
        Pick p in U that blocks the most points in U.
        Add p to S.
        Update U: remove p and any points blocked by p (and existing S).
    """
    S = set()
    U = set(range(N_PTS))
    
    while U:
        if time.time() > deadline:
            break
        
        best_p = -1
        max_blocked = -1
        
        # To speed up, we can sample candidates if U is large, but N=256 is small enough
        # to check all if we optimize the blocking check.
        # However, checking "how many in U are blocked by p" requires checking pairs in S U {p}.
        
        for p in U:
            # Temporarily add p to S to see what it blocks
            # A point q in U is blocked by S U {p} if:
            # 1. q is blocked by S (already in S? No, q is in U, so not blocked by S alone? 
            #    Wait, U contains points not blocked by S. So no point in U is blocked by S.
            #    Therefore, q is blocked by S U {p} iff there exists v in S such that {q, p, v} is isosceles.
            #    OR if q==p (but q is in U and p is in U, so q=p is possible).
            #    Actually, if we add p to S, p is removed from U.
            #    Other points q in U (q != p) are blocked if there is v in S such that {q, p, v} is iso.
            
            # Let's check how many q in U (q != p) are blocked by adding p.
            count = 0
            for q in U:
                if q == p:
                    continue
                # Check if {q, p, v} is isosceles for some v in S
                # We need to check if there exists v in S such that (p, v) is in BLOCKERS[q]
                # Or (q, v) is in BLOCKERS[p]? No, BLOCKERS[q] contains pairs that block q.
                # So we check if any pair (p, v) with v in S is in BLOCKERS[q].
                
                # Optimization: Precompute for each p, the set of points v such that (p, v) is a "bad pair" for some q?
                # Actually, BLOCKERS[q] is a set of pairs.
                # We can check: is there v in S such that (min(p,v), max(p,v)) in BLOCKERS[q]?
                
                found = False
                for v in S:
                    a, b = (p, v) if p < v else (v, p)
                    if (a, b) in BLOCKERS[q]:
                        found = True
                        break
                if found:
                    count += 1
            
            if count > max_blocked:
                max_blocked = count
                best_p = p
                
        if best_p == -1:
            # No point blocks anything? This shouldn't happen if U is not empty and S is valid.
            # If S is empty, no point blocks anything. We just pick an arbitrary one.
            best_p = next(iter(U))
            
        S.add(best_p)
        U.remove(best_p)
        
        # Update U: remove points that are now blocked by S
        # We need to re-evaluate U because S changed.
        new_U = set()
        for q in U:
            if not _is_blocked(q, S):
                new_U.add(q)
        U = new_U
        
    return S

def _local_search(S_initial, deadline):
    """
    Local search to minimize |S| while maintaining maximality.
    Moves:
    1. Remove a point p from S if S\{p} is still maximal (i.e., p is not essential for blocking any point).
       This is hard to check directly. Instead, we can try removing p and see if U becomes non-empty.
       If U becomes non-empty, we must add points to block them, which increases size.
       So, a valid move that reduces size is:
       - Find p in S such that removing p doesn't "unblock" any point.
       - This means for all q not in S, if q was blocked by S, it is still blocked by S\{p}.
       
    2. Swap: Remove p from S, add q not in S.
       This keeps size same, but might allow further reductions.
       
    Strategy:
    Iterate:
    - Try to remove a random point p from S.
    - Check if S\{p} is maximal. If yes, accept (size decreases).
    - If no, identify which points are unblocked. Let U_new be the set of unblocked points.
    - Try to add a single point q to S\{p} that blocks all of U_new? Unlikely.
    - Instead, try to swap p with q.
    
    Simpler Local Search:
    1. Calculate U (unblocked points). If U is not empty, S is not maximal.
       We must add points to block U.
       This suggests our "maximality" check is crucial.
       
    Let's refine the objective: Minimize |S| subject to U(S) = empty.
    
    Move: 
    - Pick p in S. Remove it. S' = S \ {p}.
    - Calculate U' = Unblocked(S').
    - If U' is empty, accept (size - 1).
    - Else, we need to cover U'. 
      Find q not in S' that blocks the most points in U'.
      Add q to S'. S'' = S' U {q}.
      Calculate U'' = Unblocked(S'').
      If U'' is empty, we have a valid set of size |S|. 
      If U'' is smaller than U' (which is 0? No, U' was non-empty), this is progress.
      
    Actually, a better heuristic:
    1. Try removing a point p.
    2. If valid (U empty), keep it.
    3. If invalid, try to "repair" by adding a point q that blocks the most unblocked points.
       If the repair results in a set of the same size or smaller, and is valid, keep it?
       No, if we remove 1 and add 1, size is same. We want to reduce size.
       
    So, only accept removals that keep maximality.
    If no single removal works, try swaps to find a better configuration?
    Swaps don't reduce size. But they might enable a subsequent removal.
    
    Let's stick to:
    1. Attempt removal of p.
    2. If valid, accept.
    3. If not, revert.
    4. If no removals work, try a "kick": remove p, add q (random or greedy) to restore maximality, even if size increases temporarily?
       No, we want to minimize.
       
    Alternative: 
    Use the "Blocking Power" greedily again but starting from a perturbed state?
    
    Let's implement a fast local search that only accepts size-reducing moves (removals that maintain maximality).
    To escape local minima, we can occasionally perform a "swap" move that maintains size but changes the structure, hoping to unlock a removal.
    """
    S = set(S_initial)
    best_S = set(S)
    best_size = len(S)
    
    # Precompute U for initial S
    U = _get_unblocked(S)
    # If U is not empty, S is not maximal. We must fix it first.
    while U:
        # Greedily add a point that blocks the most in U
        best_q = -1
        max_block = -1
        for q in U:
            # Check how many in U are blocked by S U {q}
            count = 0
            for r in U:
                if r == q: continue
                # Check if {r, q, v} is iso for some v in S
                found = False
                for v in S:
                    a, b = (q, v) if q < v else (v, q)
                    if (a, b) in BLOCKERS[r]:
                        found = True
                        break
                if found:
                    count += 1
            if count > max_block:
                max_block = count
                best_q = q
        if best_q == -1:
            best_q = next(iter(U))
        S.add(best_q)
        U = _get_unblocked(S)
        
    best_S = set(S)
    best_size = len(S)
    
    while time.time() < deadline:
        improved = False
        # Try to remove a point
        S_list = list(S)
        random.shuffle(S_list)
        
        for p in S_list:
            if p not in S: continue # Might have been removed in a previous iteration of this loop? No, we iterate a snapshot.
            
            S.remove(p)
            U = _get_unblocked(S)
            
            if not U:
                # Valid removal
                improved = True
                best_S = set(S)
                best_size = len(S)
                break # Restart the loop with the new smaller S
            else:
                # Invalid removal, try to repair with a single addition?
                # If we can add ONE point to block all of U, size remains same.
                # This doesn't reduce size, but might change structure.
                # Let's try to find a q that blocks all of U.
                repair_q = -1
                for q in U:
                    # Check if S U {q} blocks all points in U
                    # Note: q is in U, so q is not in S.
                    # We need to check if for all r in U (r != q), r is blocked by S U {q}.
                    # And q itself is added to S, so it's not in U anymore.
                    valid_repair = True
                    for r in U:
                        if r == q:
                            continue
                        # Check if r is blocked by S U {q}
                        # r is blocked if there is v in S U {q} such that {r, v, ...}
                        # Actually, r is blocked by S U {q} if:
                        # 1. r is blocked by S (False, r in U)
                        # 2. There is v in S such that {r, q, v} is iso.
                        #    (Because the new point is q, so the pair must involve q)
                        found = False
                        for v in S:
                            a, b = (q, v) if q < v else (v, q)
                            if (a, b) in BLOCKERS[r]:
                                found = True
                                break
                        if not found:
                            valid_repair = False
                            break
                    if valid_repair:
                        repair_q = q
                        break
                        
                if repair_q != -1:
                    S.add(repair_q)
                    # Size is same, but structure changed.
                    # We don't update best_size, but we continue searching.
                    # To avoid infinite loops, we can limit repairs or just accept the change.
                    # Since we shuffled, we will try other removals next.
                    # We don't break, we just continue to next p? 
                    # Actually, if we repair, the set S is valid. 
                    # We should probably restart the removal search with the new S.
                    improved = True 
                    break
                else:
                    # Cannot repair with 1 point. Revert.
                    S.add(p)
        
        if not improved:
            # No single removal works. 
            # Try a random swap to escape local minimum?
            # Remove p, add q.
            if S:
                p = random.choice(list(S))
                # Find a q not in S
                not_in_S = [x for x in range(N_PTS) if x not in S]
                if not_in_S:
                    q = random.choice(not_in_S)
                    S.remove(p)
                    S.add(q)
                    # Check if valid? If not, revert?
                    # For now, just accept the swap to explore.
                    # We must ensure we don't degrade too much.
                    # Let's check if the new set is maximal. If not, we'll fix it in the next iteration's "fix" loop.
                    # But the "fix" loop adds points, increasing size.
                    # So, only accept swap if it doesn't increase size? It doesn't.
                    # But if it makes the set non-maximal, the next step will add points.
                    # This is risky. 
                    # Better: Only accept swap if the resulting set is maximal?
                    U = _get_unblocked(S)
                    if U:
                        # Not maximal. Revert swap.
                        S.remove(q)
                        S.add(p)
                    else:
                        # Valid swap.
                        pass # No size change, but structure changed.

def construct():
    start_time = time.time()
    deadline = start_time + 300.0
    
    best_set = None
    best_size = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline - 10.0:
        # 1. Greedy Initialization
        S_init = _greedy_blocking_power(deadline - 5.0)
        
        # 2. Local Search
        S_opt = _local_search(S_init, deadline - 2.0)
        
        if len(S_opt) < best_size:
            best_size = len(S_opt)
            best_set = S_opt
            
    if best_set is None:
        # Fallback
        best_set = _greedy_blocking_power(deadline)
        
    return [ALL_POINTS[i] for i in best_set]

# Ensure the function is available
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
