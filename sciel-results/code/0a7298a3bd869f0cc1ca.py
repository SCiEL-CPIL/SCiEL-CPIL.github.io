"""Snake-in-the-box Q_9: Backtracking with Warnsdorff heuristic and bitmask state."""
import time
import random

N = 9
SIZE = 1 << N  # 512
TIME_LIMIT = 115.0  # safety margin below 150s hard kill

# Precompute neighbors for every vertex
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(SIZE)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    rng = random.Random(12345)
    
    best_path = None
    best_len = 0
    
    # Precompute bitmask for all vertices? No, we build the mask dynamically.
    # But we can precompute the bitmask for a single vertex: 1 << v
    # Since v < 512, 1 << v fits in a Python int easily.
    
    def backtrack(path, used_mask, forbidden_mask, target_len, time_limit):
        """
        Recursively search for a snake of length target_len.
        path: list of vertices
        used_mask: bitmask of vertices in path
        forbidden_mask: bitmask of vertices at distance 1 from any non-terminal vertex in path
                       (Note: terminals are path[0] and path[-1]. Their neighbors are not forbidden 
                        unless they are also neighbors of other non-terminals? 
                        Actually, the condition is: non-consecutive vertices must differ by >= 2 bits.
                        This is equivalent to: a vertex v can be added to the end of path P if 
                        v is not in P and dist(v, u) >= 2 for all u in P except P[-1].
                        This is equivalent to: v is not in forbidden_mask.
                        forbidden_mask contains neighbors of all u in P except P[-1] and P[0]?
                        Wait. 
                        If we add v to P, v must be dist >= 2 from all u in P except P[-1].
                        So v must NOT be a neighbor of any u in P[:-1].
                        Also v must NOT be in P.
                        So v is valid if (used_mask | forbidden_mask) & (1 << v) == 0.
                        
                        How to update forbidden_mask?
                        When we add v to the end:
                        The new path is P' = P + [v].
                        The new terminal is v.
                        The old terminal P[-1] becomes a non-terminal (unless len(P')==1, but we are extending).
                        So neighbors of P[-1] become forbidden.
                        Neighbors of P[0] are already forbidden? 
                        Initially: P = [start]. 
                        used = {start}.
                        forbidden = {} ? 
                        If we add v1 to [start], v1 must be dist >= 2 from all u in [start] except start. 
                        There are no such u. So v1 only needs to be dist 1 from start and not used.
                        So initially forbidden is empty.
                        After adding v1: P = [start, v1].
                        Next v2 must be dist >= 2 from start.
                        So neighbors of start become forbidden.
                        Generally: forbidden_mask accumulates neighbors of all vertices in path except the last one.
                        When we append v to path:
                        1. v must be valid: (used_mask | forbidden_mask) & (1 << v) == 0.
                        2. New used_mask = used_mask | (1 << v).
                        3. New forbidden_mask = forbidden_mask | (neighbors of path[-1] as bitmask).
                        Note: path[-1] was the terminal, now it is non-terminal.
                        
                        What about path[0]? 
                        If path = [v0, v1, ..., vk].
                        vk is terminal.
                        v0 is non-terminal (if k>0).
                        v1...vk-1 are non-terminal.
                        So forbidden_mask should contain neighbors of v0, v1, ..., vk-1.
                        
                        Initialization:
                        path = [v0].
                        used = 1 << v0.
                        forbidden = 0. (No non-terminals yet? v0 is terminal. If we extend, v0 becomes non-terminal? 
                        No, v0 is the start. The condition "non-consecutive" applies to (vi, vj) with |i-j| > 1.
                        For v1, it is consecutive to v0. So v1 can be neighbor of v0.
                        For v2, it is non-consecutive to v0. So v2 cannot be neighbor of v0.
                        So when we extend from v0 to v1, we don't forbid v0's neighbors yet?
                        Wait. 
                        If P = [v0], and we add v1.
                        P' = [v0, v1].
                        Next v2 must be dist >= 2 from v0.
                        So neighbors of v0 must be forbidden for the *next* step.
                        So when we move from v0 to v1, we should add neighbors of v0 to forbidden.
                        
                        So update rule:
                        When appending v to path ending in u:
                        1. Check v validity against current forbidden.
                        2. new_forbidden = forbidden | neighbors_mask(u)
                        3. new_used = used | (1 << v)
                        
                        This logic holds.
                        """
        
        if time.time() > time_limit:
            return None
            
        if len(path) >= target_len:
            return path[:]
            
        cur = path[-1]
        
        # Get candidates
        # Candidates are neighbors of cur that are not in used_mask and not in forbidden_mask
        # We iterate through NEIGHBORS[cur]
        
        # Heuristic: Warnsdorff (fewest onward options)
        # Onward options for candidate c: number of neighbors of c that are not in (used_mask | new_forbidden_mask)
        # new_forbidden_mask for c would be forbidden_mask | neighbors_mask(cur)
        # Let's compute base_forbidden = forbidden_mask | neighbors_mask(cur)
        # Then for candidate c, valid next neighbors are those not in (used_mask | base_forbidden | (1<<c))
        # Actually, if we pick c, the new state has forbidden = base_forbidden.
        # The next vertex x must be neighbor of c, not in used_mask, not in base_forbidden.
        
        base_forbidden = forbidden_mask | neighbors_mask(cur)
        
        cands = []
        for c in NEIGHBORS[cur]:
            cmask = 1 << c
            if (used_mask | forbidden_mask) & cmask:
                continue
            # Count onward options
            # Options are neighbors of c that are not in used_mask and not in base_forbidden
            # Note: c is in used_mask? No, we check (used_mask|forbidden_mask) & cmask == 0.
            # So c is available.
            # When we move to c, c becomes used.
            # The next vertex x must be neighbor of c.
            # x must not be in used_mask (which includes c? No, c is added to used).
            # x must not be in base_forbidden.
            
            # Let's just count neighbors of c that are not in (used_mask | base_forbidden)
            # Wait, c is not in used_mask yet. But c is in the path.
            # If x is neighbor of c, and x == cur? cur is in used_mask. So x != cur is guaranteed by used_mask check if cur is in used_mask.
            # cur is in used_mask. So x != cur.
            
            cnt = 0
            for x in NEIGHBORS[c]:
                xmask = 1 << x
                if not ((used_mask | base_forbidden) & xmask):
                    cnt += 1
            cands.append((cnt, c))
            
        if not cands:
            return None
            
        # Sort by fewest onward options (Warnsdorff)
        # Tie-breaker: random
        cands.sort(key=lambda t: (t[0], rng.random()))
        
        # Try each candidate
        for _, c in cands:
            new_used = used_mask | (1 << c)
            new_forbidden = base_forbidden
            path.append(c)
            
            res = backtrack(path, new_used, new_forbidden, target_len, time_limit)
            if res:
                return res
                
            path.pop()
            
        return None

    # Precompute neighbors_mask for all vertices
    neighbors_mask = [0] * SIZE
    for v in range(SIZE):
        mask = 0
        for nb in NEIGHBORS[v]:
            mask |= (1 << nb)
        neighbors_mask[v] = mask
    
    # We need to capture neighbors_mask in the closure of backtrack
    # Let's redefine backtrack to use the outer scope's neighbors_mask
    
    def do_backtrack(path, used_mask, forbidden_mask, target_len, time_limit):
        if time.time() > time_limit:
            return None
        
        if len(path) >= target_len:
            return path[:]
            
        cur = path[-1]
        base_forbidden = forbidden_mask | neighbors_mask[cur]
        
        cands = []
        for c in NEIGHBORS[cur]:
            cmask = 1 << c
            if (used_mask | forbidden_mask) & cmask:
                continue
            # Count onward options
            cnt = 0
            for x in NEIGHBORS[c]:
                xmask = 1 << x
                if not ((used_mask | base_forbidden) & xmask):
                    cnt += 1
            cands.append((cnt, c))
            
        if not cands:
            return None
            
        cands.sort(key=lambda t: (t[0], rng.random()))
        
        for _, c in cands:
            new_used = used_mask | (1 << c)
            new_forbidden = base_forbidden
            path.append(c)
            
            res = do_backtrack(path, new_used, new_forbidden, target_len, time_limit)
            if res:
                return res
                
            path.pop()
            
        return None

    # Strategy:
    # 1. Start with a greedy solution to get a baseline.
    # 2. Use backtracking to try to extend it or find a new one with higher target.
    
    # Greedy baseline
    def greedy(start, mode=0):
        path = [start]
        used = {start}
        forbidden = set()
        while True:
            cur = path[-1]
            cands = [nb for nb in NEIGHBORS[cur] if nb not in used and nb not in forbidden]
            if not cands:
                break
            if mode == 0:
                # Warnsdorff
                scored = []
                for c in cands:
                    cnt = 0
                    for nb in NEIGHBORS[c]:
                        if nb == cur: continue
                        if nb not in used and nb not in forbidden:
                            cnt += 1
                    scored.append((cnt, c))
                scored.sort(key=lambda t: (t[0], rng.random()))
                nxt = scored[0][1]
            else:
                nxt = rng.choice(cands)
            path.append(nxt)
            used.add(nxt)
            for nb in NEIGHBORS[cur]:
                forbidden.add(nb)
        return path

    # Initial greedy
    p = greedy(0)
    best_len = len(p)
    best_path = p
    
    # Try to improve with backtracking
    # We will try to find a snake of length L > best_len
    # Start with target = best_len + 1
    # If found, increase target.
    
    current_target = best_len + 1
    
    while time.time() < deadline - 5.0:
        # Try to find a snake of length current_target
        # Use multiple starts? Or just start from 0?
        # Starting from 0 is deterministic-ish but we can randomize the start or the order.
        # Let's try a few random starts.
        
        found = False
        for _ in range(10): # 10 attempts per target
            if time.time() > deadline - 5.0:
                break
                
            start = rng.randrange(SIZE)
            # Initialize state
            path = [start]
            used_mask = 1 << start
            forbidden_mask = 0
            
            res = do_backtrack(path, used_mask, forbidden_mask, current_target, deadline - 5.0)
            if res:
                if len(res) > best_len:
                    best_len = len(res)
                    best_path = res
                found = True
                break # Found one, move to next target? Or try to find longer?
                # If we found one of length current_target, we can try current_target + 1.
                
        if found:
            current_target += 1
        else:
            # If we failed to find current_target, maybe try a slightly lower target?
            # Or just stop? 
            # Let's try to reduce target slightly and try again, or just break if we've been failing.
            # Actually, if we can't find target T, we might not be able to find T+1.
            # But maybe a different start works?
            # Let's try a few more starts with the same target before giving up.
            # If we still fail, we might try target - 1? No, we already have best_len < target.
            # So best_len is the max we found.
            # If we can't find best_len + 1, we are stuck?
            # Maybe we can find a different snake of length best_len + 1 with a different start?
            # The loop above tries 10 starts.
            # If it fails, we might try to optimize the current best_path?
            # Or just stop.
            
            # Let's try one more batch of starts for the same target, then give up on this target.
            # If we give up, we can't increase best_len.
            # So we break.
            
            # Wait, maybe we can try to extend the current best_path specifically?
            # The backtracking search is global.
            
            # Let's just break to avoid wasting time if we can't improve.
            # But maybe a different random seed in the sort order helps?
            # We use rng.random() in sort.
            
            # Let's try 50 more starts if we are close to deadline?
            # No, 10 is reasonable.
            
            break

    return best_path
