import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for all vertices
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(SIZE)]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_path = []
    best_len = 0
    
    # Bitmask-based state
    # used_mask: bit v set if v is in path
    # blocked_mask: bit v set if v is adjacent to any vertex in path except current end
    
    def try_backtrack(start_v, time_limit):
        nonlocal best_path, best_len
        
        used_mask = 1 << start_v
        blocked_mask = 0
        for n in NEIGHBORS[start_v]:
            blocked_mask |= (1 << n)
        path = [start_v]
        
        def dfs(cur, time_limit):
            nonlocal best_path, best_len, used_mask, blocked_mask, path
            
            if time.time() > time_limit:
                return False
            
            # Update best if current path is longer
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            cur = path[-1]
            
            # Find candidates: neighbors of cur that are not used and not blocked
            candidates = []
            for u in NEIGHBORS[cur]:
                if not (used_mask >> u) & 1 and not (blocked_mask >> u) & 1:
                    candidates.append(u)
            
            if not candidates:
                return False
            
            # Heuristic: pick candidate with fewest onward options (MRV)
            # Score: number of valid candidates for next step after choosing u
            def score_candidate(u):
                # If we move to u, cur becomes blocked
                new_blocked = blocked_mask | (1 << cur)
                # Remove cur from blocked (it's now the previous vertex, not blocked for u)
                # Actually, cur is adjacent to u, but cur is in the path.
                # The blocked set for the new end u should include all neighbors of path vertices except u.
                # Current blocked includes neighbors of all path vertices except cur.
                # New path: path + [u].
                # New blocked: neighbors of (path + [u]) except u.
                # = (neighbors of path except cur) union (neighbors of u)
                # = (old_blocked union neighbors_of_cur) union neighbors_of_u
                # But we already have old_blocked which is neighbors of path except cur.
                # So new_blocked = old_blocked | (1<<cur) | neighbors_of_u
                # Wait, cur is adjacent to u, so cur is in neighbors_of_u.
                # Let's just compute: new_blocked = 0
                # For v in path[:-1]: add neighbors of v
                # For v in path[:-1] + [cur]: add neighbors of v (but not u itself)
                # Actually simpler: new_blocked = blocked_mask | (1 << cur)
                # Then add neighbors of u, except u itself (which is not a neighbor of u)
                for w in NEIGHBORS[u]:
                    if w != cur:
                        new_blocked |= (1 << w)
                
                count = 0
                for w in NEIGHBORS[u]:
                    if not (new_blocked >> w) & 1 and not (used_mask >> w) & 1:
                        count += 1
                return count
            
            candidates.sort(key=score_candidate)
            
            # Try candidates in order of increasing score (most constrained first)
            # Add some randomness for variety
            if len(candidates) > 1 and random.random() < 0.3:
                # Shuffle top few
                k = min(3, len(candidates))
                random.shuffle(candidates[:k])
            
            for u in candidates:
                if time.time() > time_limit:
                    return False
                
                # Move to u
                old_used = used_mask
                old_blocked = blocked_mask
                path.append(u)
                used_mask |= (1 << u)
                
                # Compute new blocked
                # blocked was neighbors of path[:-1] (all except cur)
                # New path is path[:-1] + [cur, u]
                # New blocked = neighbors of (path[:-1] + [cur]) except u
                # = old_blocked | neighbors_of_cur | neighbors_of_u
                # But we must exclude u from blocked (it's the new end)
                # And we must exclude vertices in path from blocked? No, blocked is for candidates.
                # A vertex v is blocked if it's adjacent to any path vertex except the current end.
                # Current end is u. So v is blocked if v is adjacent to any vertex in path[:-1] (which includes cur now).
                # So new_blocked = old_blocked | (1<<cur) [cur is now in path[:-1] relative to u]
                # Wait, old_blocked was neighbors of path[:-2] (all except cur).
                # Now path[:-1] relative to u is the old path.
                # So new_blocked should be neighbors of old_path.
                # old_blocked = neighbors of old_path[:-1] (all except cur).
                # So new_blocked = old_blocked | neighbors_of_cur.
                # But we also need to make sure u is not in blocked (it's the end).
                # And we need to remove any bits for vertices in path? No, blocked is a mask of forbidden vertices.
                # Vertices in path are already in used_mask, so they won't be candidates anyway.
                
                new_blocked = old_blocked
                for w in NEIGHBORS[cur]:
                    if w != u:
                        new_blocked |= (1 << w)
                # Also add neighbors of u? No, u is the end, its neighbors are not blocked by u.
                # But u's neighbors might be blocked by other path vertices.
                # The blocked mask should contain all vertices adjacent to any vertex in path except u.
                # path = old_path + [u]
                # path except u = old_path
                # So blocked = neighbors of old_path.
                # old_blocked = neighbors of old_path[:-1]
                # So new_blocked = old_blocked | neighbors_of_cur (since cur is in old_path but was the end before)
                
                blocked_mask = new_blocked
                
                # Recurse
                if dfs(u, time_limit):
                    # Undo move before returning timeout
                    path.pop()
                    used_mask = old_used
                    blocked_mask = old_blocked
                    return True
                
                # Undo move
                path.pop()
                used_mask = old_used
                blocked_mask = old_blocked
            
            return False
        
        dfs(start_v, time_limit)
        return best_path
    
    # Run searches
    search_deadline = start_time + 140.0
    num_searches = 20
    search_time = (search_deadline - time.time()) / num_searches
    
    for i in range(num_searches):
        if time.time() >= search_deadline:
            break
        start_v = random.randint(0, SIZE - 1)
        current_deadline = time.time() + search_time
        try_backtrack(start_v, current_deadline)
    
    # Final verification
    if best_path and len(best_path) > 1:
        valid = True
        for i in range(len(best_path)-1):
            if (best_path[i] ^ best_path[i+1]).bit_count() != 1:
                valid = False
                break
        if valid:
            for i in range(len(best_path)):
                for j in range(i+2, len(best_path)):
                    if (best_path[i] ^ best_path[j]).bit_count() == 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                best_path = [0]
        else:
            best_path = [0]
    else:
        best_path = [0]
        
    return best_path

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)}")
