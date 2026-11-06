import random
import time

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute for each vertex: bitmask of all neighbors
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

def construct():
    deadline = time.time() + 110.0
    rnd = random.Random(20260715)
    best_path = None
    best_len = 0

    def validate_and_score(path):
        """Check validity and return edge count."""
        if not path:
            return 0
        n = len(path)
        for i in range(n - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return -1
        used_mask = 0
        for i, v in enumerate(path):
            if used_mask & (1 << v):
                return -1
            if i >= 2:
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << path[i-1])
                if forbidden:
                    return -1
            used_mask |= (1 << v)
        return n - 1

    def backtracking_search(max_depth=200, max_branching=10, time_limit=None):
        """
        Constraint-based backtracking search.
        At each step, we only choose a next vertex 'v' if it doesn't 'kill' the path.
        Heuristic: 'v' is safe if for all unused vertices 'u', the number of unused neighbors of 'u' 
        (excluding the current vertex) does not decrease by choosing 'v'.
        """
        if time_limit is None:
            time_limit = deadline - 5.0
            
        path = [0]
        used_mask = 1
        best_local_len = 1
        best_local_path = [0]

        # To make backtracking efficient, we don't actually backtrack in a deep tree 
        # but rather use a "commit" strategy with look-ahead, or a limited depth backtrack.
        # Given the 120s limit, deep backtracking on 2^9=512 vertices is risky.
        # We will use a "Greedy with Lookahead" approach which is essentially 
        # backtracking with depth 1 or 2, but systematically applied.
        
        # Actually, let's implement a proper DFS with a strict depth limit and pruning.
        # But DFS can get stuck. Let's use a "Beam of Backtracking" or simply 
        # restart the backtracking with different randomizations of tie-breaking.
        
        # A better strategy for this specific "Island" improvement:
        # Use the "Safe Set" heuristic in a greedy loop, but with random tie-breaking 
        # among safe moves. If stuck, restart.
        
        # Let's implement a single run of this specific heuristic:
        # 1. Start at 0.
        # 2. Find all valid next moves.
        # 3. For each move, calculate a "danger score" = how many future options it kills.
        # 4. Pick the move with the lowest danger score (random tie-break).
        # 5. Repeat.
        
        # We will run this multiple times with different random seeds to explore the space.
        
        def run_once(seed):
            local_rnd = random.Random(seed)
            path = [0]
            used_mask = 1
            cur = 0
            
            for _ in range(300):
                if time.time() > time_limit:
                    break
                
                candidates = []
                for b in range(N):
                    v = cur ^ (1 << b)
                    if used_mask & (1 << v):
                        continue
                    # Check if v is valid (not adjacent to any other used vertex)
                    forbidden = NEIGHBOR_MASK[v] & used_mask
                    forbidden &= ~(1 << cur)
                    if forbidden:
                        continue
                    candidates.append(v)
                
                if not candidates:
                    break
                
                # Heuristic: Score each candidate
                # A candidate v is "good" if it leaves many options open for others.
                # Specifically, for every unused vertex u, let N(u) be its unused neighbors.
                # If we pick v, u loses v as a neighbor (if v was in N(u)).
                # We want to minimize the total "damage".
                # Damage = sum over all unused u of (1 if v is in N(u) else 0)
                # This is equivalent to the number of unused neighbors of v.
                # So we prefer v with the fewest unused neighbors? 
                # Wait, if v has few unused neighbors, it's more "isolated" in the remaining graph.
                # Usually, we want to keep the graph connected.
                # Let's try: Pick v that minimizes the number of unused neighbors of v.
                # This is the "minimum degree" heuristic.
                
                # Alternative: Pick v that maximizes the minimum degree of remaining vertices?
                # That's expensive.
                
                # Let's try "Minimize unused neighbors of v" (greedy for dead-ends)
                # vs "Maximize unused neighbors of v" (keep options open).
                # Standard greedy for snake-in-box often fails.
                # Let's try a randomized version of "Minimize unused neighbors of v".
                
                scored = []
                for v in candidates:
                    # Count unused neighbors of v
                    # NEIGHBOR_MASK[v] contains all neighbors.
                    # unused_neighbors = NEIGHBOR_MASK[v] & ~used_mask
                    # But we must exclude cur, which is used.
                    un = NEIGHBOR_MASK[v] & ~used_mask
                    # cur is in used_mask, so it's already excluded.
                    # However, cur is a neighbor of v. Since cur is used, it's not in ~used_mask.
                    # So un is the count of *future* neighbors.
                    score = un.bit_count()
                    scored.append((score, v))
                
                # Sort by score
                scored.sort(key=lambda x: x[0])
                
                # Random selection among the best (lowest score)
                min_score = scored[0][0]
                best_moves = [v for s, v in scored if s == min_score]
                
                # Sometimes, picking the absolute min is too greedy.
                # Let's pick randomly from the top 3 lowest scores, or just the min.
                # To add diversity, pick randomly from all moves with score <= min_score + 1
                threshold = min_score + 1
                good_moves = [v for s, v in scored if s <= threshold]
                
                nxt = local_rnd.choice(good_moves)
                
                path.append(nxt)
                used_mask |= (1 << nxt)
                cur = nxt
                
            return path

        # Run multiple instances of this heuristic
        best_local = []
        for i in range(50):
            if time.time() > time_limit:
                break
            p = run_once(rnd.randint(0, 10**9))
            L = len(p)
            if L > best_local_len:
                best_local_len = L
                best_local_path = p
        
        return best_local_path, best_local_len

    # Phase 0: Quick greedy baseline (same as before for consistency)
    def greedy(priority):
        path = [0]
        used = {0}
        used_mask = 1
        for _ in range(300):
            if time.time() > deadline:
                break
            cur = path[-1]
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
            used_mask |= (1 << nxt)
        return path

    seed_path = greedy(list(range(N)))
    L = validate_and_score(seed_path)
    if L > 0:
        best_len = L
        best_path = seed_path

    # Phase 1: The New Strategy - Constraint/Heuristic Backtracking
    # We allocate a significant portion of time to this new approach.
    # We will run it in bursts.
    
    # To answer the "parent change" question:
    # The current program (parent) uses Beam Search.
    # This program uses a "Minimum Degree Heuristic" with randomization.
    # If this yields a significantly different score (better or worse), it suggests the search landscape 
    # is sensitive to the heuristic, and the "parent" (beam search) may have been stuck in a local optimum 
    # that the new heuristic escapes, or vice versa.
    
    # Let's run the new strategy
    new_path, new_len = backtracking_search()
    if new_path:
        vL = validate_and_score(new_path)
        if vL > best_len:
            best_len = vL
            best_path = new_path

    # Phase 2: Continue with some original beam search if time remains, 
    # to ensure we don't lose the baseline capability if the new heuristic fails.
    # But the directive says "rethinking the approach". 
    # So we should rely primarily on the new approach.
    # However, mixing is allowed if it improves the score.
    
    # Let's add a "Hybrid" phase: 
    # Use the path found by the new heuristic as a seed for a local search?
    # Or just run more instances of the new heuristic with different parameters.
    
    # Let's try a variation: "Maximum Degree" heuristic (keep options open)
    def run_max_degree(seed):
        local_rnd = random.Random(seed)
        path = [0]
        used_mask = 1
        cur = 0
        
        for _ in range(300):
            if time.time() > deadline:
                break
            
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                candidates.append(v)
            
            if not candidates:
                break
            
            scored = []
            for v in candidates:
                un = NEIGHBOR_MASK[v] & ~used_mask
                score = un.bit_count()
                scored.append((score, v))
            
            # Sort by score DESCENDING (Max degree)
            scored.sort(key=lambda x: -x[0])
            
            max_score = scored[0][0]
            # Pick from top 2
            good_moves = [v for s, v in scored if s >= max_score - 1]
            
            nxt = local_rnd.choice(good_moves)
            
            path.append(nxt)
            used_mask |= (1 << nxt)
            cur = nxt
            
        return path

    # Run Max Degree heuristic
    for i in range(20):
        if time.time() > deadline - 5:
            break
        p = run_max_degree(rnd.randint(0, 10**9))
        L = validate_and_score(p)
        if L > best_len:
            best_len = L
            best_path = p

    if best_path is None:
        return [0]
    return best_path
