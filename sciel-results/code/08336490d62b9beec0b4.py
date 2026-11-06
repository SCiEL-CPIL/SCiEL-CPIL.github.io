"""
Snake-in-the-box in Q_9 using SAT solving.
The strategy models the problem as a constraint satisfaction problem where we search for a sequence of L+1 vertices forming an induced path.
"""
N = 9
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Precompute neighbors for all vertices in Q_9
    # 2^9 = 512 vertices. Each has 9 neighbors.
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    # Precompute a valid adjacency matrix for the snake constraint?
    # Actually, the constraint is: if path[i] and path[j] are in the path, 
    # and |i-j| > 1, then path[i] and path[j] must NOT be neighbors in Q_N.
    # This is a global constraint on the sequence.

    def sat_solve(L, time_limit):
        """
        Tries to find a path of length L (L+1 vertices).
        Variables: x[i, v] = True if vertex v is at position i in the path.
        Constraints:
        1. Exactly one vertex at each position i: sum_v x[i, v] == 1
        2. Consecutive vertices differ by 1 bit: x[i, u] -> (x[i+1, v] for v in neighbors[u])
           Actually: x[i, u] and x[i+1, v] implies dist(u,v)=1.
           Or: x[i, u] -> OR_{v in neighbors[u]} x[i+1, v]
           And: x[i+1, v] -> OR_{u in neighbors[v]} x[i, u]
           Better: x[i, u] ^ x[i+1, v] is only allowed if dist(u,v)=1.
           Clause: NOT(x[i, u] AND x[i+1, v]) for all dist(u,v) != 1.
        3. Non-consecutive vertices differ by at least 2 bits:
           NOT(x[i, u] AND x[j, v]) for all |i-j| > 1 and dist(u,v) == 1.
        4. Vertices are distinct: x[i, u] and x[j, u] is false for i != j.
           Clause: NOT(x[i, u] AND x[j, u]) for i != j.
        
        This formulation has L+1 positions and 512 vertices.
        Number of variables: (L+1) * 512.
        For L=104, that's 105 * 512 = 53760 variables.
        
        The number of clauses might be large, but DPLL with unit propagation and pure literal elimination might work for small L.
        However, L=104 is quite large for a naive DPLL without advanced propagation.
        
        Let's try a smaller L first to see if it works, then increase.
        Or, use a more efficient encoding?
        
        Alternative encoding:
        Instead of x[i, v], use a permutation of a subset?
        
        Let's stick to the x[i, v] encoding but optimize the clause generation.
        We can generate clauses on the fly or precompute them.
        Precomputing 53k variables and potentially millions of clauses might be slow in Python.
        
        Let's estimate clauses for L=20 (small test):
        21 * 512 vars.
        Constraint 1: 21 * (512 choose 2) ~ 21 * 130k = 2.7M clauses. Too many.
        
        We need a smarter SAT encoding or a different approach.
        
        How about encoding the path as a sequence of edges?
        Or, since the graph is a hypercube, maybe we can use the structure.
        
        Let's reconsider the "Local Search" inspiration which scored 1.0096.
        And the "Bitset backtracking" which scored 1.0673.
        
        The SAT approach might be too slow in Python for L > 20.
        
        Let's try a different strategy: **Simulated Annealing** on the snake path.
        State: A valid snake (induced path).
        Moves: 
        1. Extend/Shorten (if possible).
        2. Swap two non-adjacent vertices? No, must maintain path property.
        3. "2-opt" move: Reverse a segment of the path?
           If we reverse a segment [i..j], the new connections are (i-1, j) and (j+1, i).
           We need dist(path[i-1], path[j]) == 1 and dist(path[j+1], path[i]) == 1.
           And we must ensure no new conflicts are introduced.
           Reversing a segment preserves the internal edges.
           The only new edges are at the boundaries.
           The only new non-edges to check are between the reversed segment and the rest.
           But wait, if the path was induced, all non-consecutive pairs were non-neighbors.
           After reversal, the internal pairs are still non-consecutive (unless length 2).
           The pairs between the segment and the outside change their relative indices.
           Actually, if we reverse [i..j], the set of vertices is the same.
           The set of consecutive pairs changes:
           Old: (i-1, i), (i, i+1)...(j, j+1)
           New: (i-1, j), (j, j-1)...(i, j+1)
           So we lose edges (i-1, i) and (j, j+1) and gain (i-1, j) and (j, i).
           We must check if (i-1, j) is an edge and (i, j+1) is an edge? No, (j, i) is the same as (i, j).
           So we need dist(path[i-1], path[j]) == 1 and dist(path[i], path[j+1]) == 1.
           Also, we must ensure that the new consecutive pairs do not create conflicts with other vertices.
           Specifically, path[i-1] and path[j] are now consecutive. Are they neighbors with any other vertex in the path that was previously non-consecutive to one of them?
           This is complex to check incrementally.
           
        Let's try **Bitset-based Backtracking** with a better heuristic, as it scored 1.0673.
        The current program uses a naive O(L) check for validity.
        We can maintain a bitset of "blocked" vertices for the next position.
        
        Let's implement a highly optimized backtracking search.
        State:
        - path: list of vertices
        - used: set of vertices used
        - blocked_next: bitset of vertices that cannot be the next vertex.
          A vertex v is blocked for the next position if:
          1. v is in used.
          2. v is a neighbor of any path[i] for i < len(path)-1.
             (Because if v is next, it would be non-consecutive with path[i] and be a neighbor, which is invalid).
          
        When we add a new vertex u to the path:
        - Mark u as used.
        - Add neighbors of path[-2] to blocked_next? No.
        - The new vertex u becomes the last vertex.
        - The "previous last" vertex (old path[-1]) is no longer the last.
          It becomes path[-2].
          Its neighbors must now be blocked for the NEXT step?
          No, the constraint is: non-consecutive vertices must NOT be neighbors.
          If we are at position k, we choose v.
          v must not be a neighbor of path[i] for i <= k-2.
          
        So, maintain a set `forbidden` which contains all vertices that are neighbors of path[i] for i <= len(path)-2.
        When we extend the path by u:
        - New path is path + [u].
        - The new forbidden set for the next extension (position k+1) will be:
          (Old forbidden set) U (neighbors of path[-2]).
          Wait, path[-2] was already in the path when we chose u.
          When we chose u, we ensured u is not in forbidden (which contained neighbors of path[i] for i <= k-2, where k was the length before adding u).
          So u is not a neighbor of any path[i] for i <= k-2.
          Now, u is at position k-1 (0-indexed).
          For the next vertex w (at position k), w must not be a neighbor of path[i] for i <= k-2.
          So w must not be in (Old forbidden set) U (neighbors of path[-2]).
          
        So the update is:
        forbidden_new = forbidden_old | neighbors(path[-2])
        
        This is O(1) if we use bitsets (integers).
        
        Algorithm:
        1. Start with path = [start_v]. forbidden = 0.
        2. At each step:
           - Available candidates = neighbors(last_v) & ~forbidden & ~used_mask
           - If no candidates, backtrack.
           - Choose a candidate (heuristic: one that leaves the most options?).
           - Update path, used_mask, forbidden.
           
        Heuristic:
        - "Most Constrained Variable" or "Least Constraining Value".
        - For each candidate v, count how many options it leaves for the next step.
          Options for next step from v: neighbors(v) & ~(forbidden | neighbors(last_v)) & ~used_mask.
          Choose v that maximizes this count.
        
        This is a standard DFS with pruning.
        With bitsets, the operations are very fast.
        
        Let's implement this.
        """
        pass

    # Implementing the Bitset Backtracking
    
    def bitset_backtrack(start_v, bit_order, time_limit):
        # path is a list of ints
        path = [start_v]
        used_mask = 1 << start_v
        forbidden = 0  # Bitset of vertices that cannot be next
        
        # Precomputed neighbor masks for each vertex
        # neighbor_mask[v] is an integer with bits set for all neighbors of v
        # This is global, computed once.
        
        def dfs(depth):
            nonlocal path, used_mask, forbidden
            if time.time() > deadline:
                return False
            
            if depth == target_length:
                return True
                
            cur = path[-1]
            # Candidates are neighbors of cur, not used, not forbidden
            cand_mask = neighbor_mask[cur] & ~used_mask & ~forbidden
            
            if cand_mask == 0:
                return False
            
            # Get list of candidate vertices
            candidates = []
            m = cand_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                candidates.append(v)
                m ^= lsb
            
            # Heuristic: sort candidates by number of future options (ascending or descending?)
            # Usually "fail-first" (least options) is good for finding *a* solution quickly.
            # But we want a *long* solution.
            # Actually, we are searching for a fixed length `target_length`.
            # If we just want to find ANY path of that length, fail-first is good.
            # Let's try fail-first.
            
            # For each candidate, calculate how many options it leaves for the NEXT step.
            # Next forbidden will be forbidden | neighbor_mask[path[-2]] (if len(path)>=2)
            # Wait, if we add v, the new last is v.
            # The new forbidden for the step AFTER v will be (current forbidden) | neighbor_mask[path[-2]]?
            # No.
            # Current forbidden contains neighbors of path[i] for i <= len(path)-2.
            # When we add v, v is at index len(path)-1.
            # For the next vertex w, w must not be neighbor of path[i] for i <= len(path)-1.
            # So w must not be in (current forbidden) | neighbor_mask[v].
            # Wait, v is the new last. Its neighbors are forbidden for the NEXT next vertex?
            # Yes, because if w is neighbor of v, and w is at position k+1, and v is at k, they are consecutive. That's allowed.
            # BUT, w must not be neighbor of path[i] for i <= k-1.
            # So w must not be in (current forbidden) | neighbor_mask[path[-2]]?
            # Let's re-verify.
            # Path: p0, p1, ..., pk.
            # Forbidden for pk+1: neighbors of p0, ..., pk-1.
            # When we add pk+1, the new forbidden for pk+2 is: neighbors of p0, ..., pk.
            # So new_forbidden = old_forbidden | neighbor_mask[pk].
            
            # So if we choose v as the next vertex (pk+1):
            # new_forbidden = forbidden | neighbor_mask[path[-1]]  (where path[-1] is pk)
            # Wait, path[-1] is the current last vertex before adding v.
            # So new_forbidden = forbidden | neighbor_mask[cur].
            
            # Options for the vertex after v:
            # neighbors(v) & ~(new_forbidden) & ~(used_mask | (1<<v))
            
            def score_candidate(v):
                new_forb = forbidden | neighbor_mask[cur]
                new_used = used_mask | (1 << v)
                next_cand = neighbor_mask[v] & ~new_forb & ~new_used
                return next_cand.bit_count()
            
            # Sort candidates by score (ascending for fail-first)
            candidates.sort(key=score_candidate)
            
            for v in candidates:
                # Save state
                old_path_len = len(path)
                old_used = used_mask
                old_forb = forbidden
                
                path.append(v)
                used_mask = old_used | (1 << v)
                forbidden = old_forb | neighbor_mask[cur]
                
                if dfs(depth + 1):
                    return True
                
                # Restore state
                path.pop()
                used_mask = old_used
                forbidden = old_forb
                
            return False
            
        return dfs(1)

    # Precompute neighbor masks
    neighbor_mask = [0] * (1 << N)
    for v in range(1 << N):
        mask = 0
        for b in range(N):
            mask |= 1 << (v ^ (1 << b))
        neighbor_mask[v] = mask

    best_path = [0]
    
    # Try to find a path of increasing length
    # Start from the seed length (104) and go up?
    # Or start from a small length and increase?
    # SAT/backtracking is exponential. We should try to find the maximum L.
    # Let's try a range of L.
    
    # To be safe, let's first ensure we can find a path of length 104 quickly.
    # Then try 105, 106, etc.
    
    # We need to be careful with the time limit.
    # Let's allocate time to each attempt.
    
    current_time = time.time()
    
    # Try lengths from 104 upwards
    # We might not be able to go much higher than 104 quickly.
    # Let's try 104, 105, 106, 107, 108, 109, 110.
    
    for L in range(104, 120):
        if time.time() > deadline - 10:
            break
            
        # Try a few random starts and bit orders
        found = False
        attempts = 0
        while not found and attempts < 5 and time.time() < deadline - 10:
            start_v = random.randint(0, (1 << N) - 1)
            bit_order = list(range(N))
            random.shuffle(bit_order)
            
            # The bitset backtracker doesn't use bit_order directly in the candidate generation
            # because it uses masks. But we can use it to break ties or order candidates if needed.
            # Currently, we sort by score.
            
            if bitset_backtrack(start_v, bit_order, deadline):
                if len(path) > len(best_path):
                    best_path = path[:]
                found = True
            attempts += 1
            
    return best_path

# EVOLVE-BLOCK-END
