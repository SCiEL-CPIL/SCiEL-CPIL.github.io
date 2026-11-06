"""Snake-in-the-box Q_9: beam search with bitmask validity, multi-restart, deadline guard."""
import time
import random
from collections import deque

N = 9
TOTAL_VERTS = 1 << N
# Precompute bit masks for each dimension
BIT_MASKS = [1 << b for b in range(N)]

def _bit_count(x):
    return bin(x).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    best_path = None
    best_len = 0
    
    # Precompute adjacency: for each vertex, list of neighbors
    # adj[v] = list of (neighbor, bit_index)
    # Actually, let's just compute on the fly but use a valid_mask approach.
    
    # valid_mask: for a given path, which vertices are NOT allowed to be the next one.
    # A vertex v is invalid if:
    #   1. v is already in the path
    #   2. v differs from any non-consecutive vertex in the path by exactly 1 bit
    #      (i.e., v is adjacent to path[i] for i < len(path)-2)
    # Note: v is allowed to be adjacent to path[-1] (that's the extension)
    #       and v is allowed to be at distance >= 2 from all other vertices.
    
    # Let's use a different approach: maintain a set of "forbidden" vertices.
    # forbidden = set of vertices that cannot be the next vertex.
    # Initially forbidden = {path[0]} (can't revisit)
    # When we add vertex v to path:
    #   - v becomes forbidden (can't revisit)
    #   - All neighbors of v (except the previous vertex in path) become forbidden
    #     because if we later try to go to a neighbor of v, it would be at distance 1
    #     from v which is non-consecutive.
    # Wait, that's not quite right. Let me think again.
    # 
    # The constraint is: for any i < j-1, path[i] and path[j] must differ in >= 2 bits.
    # So when we extend path by adding new vertex w (so w = path[len(path)]),
    # we need: for all i < len(path)-1, (w ^ path[i]).bit_count() >= 2.
    # Equivalently, w must not be adjacent to any path[i] for i < len(path)-1.
    # And w must not be in the path already.
    #
    # So forbidden set F: vertices that cannot be the next vertex.
    # F initially = {path[0]}
    # When we add w to path (making it the new last element):
    #   - w is added to F (can't revisit)
    #   - All neighbors of w EXCEPT the previous last vertex (path[-2]) are added to F
    #     because those neighbors would be at distance 1 from w, which is now non-consecutive
    #     to any future vertex... wait, no.
    #
    # Let me re-think. The forbidden set for the NEXT vertex after path is:
    #   {v : v in path} ∪ {v : exists i in [0, len(path)-2] such that (v ^ path[i]).bit_count() == 1}
    #
    # So when we add w as the new last vertex:
    #   - w is now in path, so w is forbidden.
    #   - w was previously the "current last", so its neighbors were NOT forbidden 
    #     (they were allowed as potential next vertices). Now w is non-consecutive to 
    #     future vertices, so all neighbors of w become forbidden.
    #   - BUT: the previous last vertex (call it p) is still the second-to-last. 
    #     p's neighbors that are NOT w are already forbidden (they were forbidden when 
    #     p was added... wait, no).
    #
    # Let me trace through:
    # path = [a, b]
    # Forbidden for next vertex: {a, b} ∪ {neighbors of a except b}
    #   = {a, b} ∪ {x : x adjacent to a, x != b}
    #
    # Now we add c: path = [a, b, c]
    # Forbidden for next vertex: {a, b, c} ∪ {neighbors of a except b} ∪ {neighbors of b except c}
    #   = {a, b, c} ∪ {x : x adjacent to a, x != b} ∪ {x : x adjacent to b, x != c}
    #
    # So when we add c:
    #   - c is added to forbidden
    #   - All neighbors of c except b are added to forbidden
    #   - b was previously "allowed" as a neighbor (since b was the last vertex, its neighbors 
    #     were not forbidden). Now b is non-consecutive to future vertices, so all neighbors 
    #     of b except c become forbidden.
    #   - But wait, b was already in the path, so b was already forbidden. The neighbors of b 
    #     (except c) were NOT previously forbidden (they were potential next vertices). Now they are.
    #
    # So the update when adding w (new last) to path [..., p, w] where p was the old last:
    #   1. Add w to forbidden
    #   2. For each neighbor x of w: if x != p, add x to forbidden
    #   3. For each neighbor x of p: if x != w, add x to forbidden  (these were previously allowed)
    #
    # Wait, step 3 is wrong. Let me re-examine.
    #
    # Before adding w: path = [..., p]. Forbidden = {vertices in path} ∪ {neighbors of each path[i] for i < len-1, except path[i+1]}
    # After adding w: path = [..., p, w]. Forbidden = {vertices in path} ∪ {neighbors of each path[i] for i < len-1, except path[i+1]}
    #
    # The difference:
    # - w is now in path: add w
    # - p is now at index len-2 (was len-1): p's neighbors except w are now forbidden. 
    #   Previously, p's neighbors (except the one before p) were forbidden, but p's neighbors 
    #   that are NOT the one before p AND NOT w were... wait.
    #
    # Let me just use a simpler approach: maintain the forbidden set explicitly.
    # When adding w to path (where path was [..., p]):
    #   - Add w to forbidden
    #   - Add all neighbors of w except p to forbidden
    #   - Add all neighbors of p except w to forbidden  (because p is now non-consecutive)
    #   - BUT: we should NOT remove anything. The forbidden set only grows.
    #
    # Wait, that's not right either. Let me think about what's forbidden.
    #
    # Actually, I think the cleanest way is:
    # forbidden = set()
    # path = [v0]
    # forbidden = {v0}
    # For each new vertex w added to path (path = [..., p, w]):
    #   forbidden.add(w)
    #   for x in neighbors(w):
    #       if x != p:
    #           forbidden.add(x)
    #   for x in neighbors(p):
    #       if x != w and x not in path_set:  # x is a neighbor of p, not w
    #           # x is now forbidden because p is non-consecutive to future vertices
    #           forbidden.add(x)
    #
    # Hmm, but this adds neighbors of p that might already be forbidden. That's fine.
    # But we need to be careful: when we add w, we're saying that p is now "locked in" as 
    # non-consecutive to future vertices. So all neighbors of p (except w, which is the 
    # consecutive neighbor) become forbidden.
    #
    # Let me verify with a small example.
    # path = [0, 1] (0 and 1 differ in bit 0)
    # forbidden = {0, 1}
    # When we added 1 to [0]:
    #   forbidden.add(1)
    #   neighbors(1) = {0, 3, 5, ...} (in Q_3: 1^1=0, 1^2=3, 1^4=5)
    #   for x in neighbors(1): if x != 0: forbidden.add(x) -> add 3, 5
    #   neighbors(0) = {1, 2, 4}
    #   for x in neighbors(0): if x != 1 and x not in {0,1}: forbidden.add(x) -> add 2, 4
    # So forbidden = {0, 1, 2, 3, 4, 5}
    # In Q_3, that's all vertices except 6, 7.
    # Can we extend [0, 1] to 6? 6 = 110, 1 = 001, diff = 111 (3 bits). 0 = 000, 6 = 110, diff = 110 (2 bits). OK.
    # Can we extend to 7? 7 = 111, 1 = 001, diff = 110 (2 bits). 0 = 000, 7 = 111, diff = 111 (3 bits). OK.
    # So 6 and 7 are valid next vertices. Forbidden has 6 and 7 not in it. Correct!
    #
    # Great, so the algorithm is:
    # forbidden = {v0}
    # path = [v0]
    # While we can extend:
    #   next_candidates = [v for v in neighbors(path[-1]) if v not in forbidden]
    #   if not next_candidates: break
    #   pick w from next_candidates
    #   p = path[-1]
    #   path.append(w)
    #   forbidden.add(w)
    #   for x in neighbors(w):
    #       if x != p:
    #           forbidden.add(x)
    #   for x in neighbors(p):
    #       if x != w:
    #           forbidden.add(x)
    
    # Now, beam search: maintain a set of (path, forbidden) states.
    # To save memory, we can store path as a tuple and forbidden as a frozenset or a bitmask.
    # Actually, for Q_9, 512 vertices, we can use a 512-bit integer as a bitmask for forbidden.
    
    # Let's use a bitmask for forbidden: an integer where bit i is 1 if vertex i is forbidden.
    
    def _neighbors(v):
        return [v ^ (1 << b) for b in range(N)]
    
    # Beam search with multiple restarts
    beam_width = 50
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random starting vertex
        start_v = random.randint(0, TOTAL_VERTS - 1)
        
        # Initialize state
        path = [start_v]
        forbidden_mask = 1 << start_v
        
        # Beam: list of (path, forbidden_mask)
        beam = [(path, forbidden_mask)]
        
        while True:
            if time.time() > deadline:
                break
            
            new_beam = []
            for (p, fm) in beam:
                cur = p[-1]
                prev = p[-2] if len(p) >= 2 else None
                
                # Find valid next vertices
                candidates = []
                for b in range(N):
                    v = cur ^ (1 << b)
                    if not (fm >> v) & 1:
                        candidates.append(v)
                
                if not candidates:
                    continue
                
                # Score candidates: prefer those that leave more options
                # Simple heuristic: just take all, or take top-k by some score
                # For now, just add all to new_beam (beam search)
                for v in candidates:
                    new_fm = fm | (1 << v)
                    # Add neighbors of v except cur
                    for b2 in range(N):
                        x = v ^ (1 << b2)
                        if x != cur:
                            new_fm |= (1 << x)
                    # Add neighbors of cur except v
                    for b2 in range(N):
                        x = cur ^ (1 << b2)
                        if x != v:
                            new_fm |= (1 << x)
                    
                    new_path = p + [v]
                    new_beam.append((new_path, new_fm))
            
            if not new_beam:
                break
            
            # Update best
            for (p, fm) in new_beam:
                if len(p) > best_len:
                    best_len = len(p)
                    best_path = p[:]
            
            # Keep top beam_width states
            # Score: longer is better, then maybe diversity
            new_beam.sort(key=lambda x: -len(x[0]))
            beam = new_beam[:beam_width]
            
            # If all beams have same length, we might be stuck
            # Check if we can continue
            if all(len(p) == best_len for (p, fm) in beam):
                # All beams have reached best_len, no progress possible
                break
        
        # Try a few more restarts with different beam widths
        if time.time() > deadline:
            break
    
    if best_path is None:
        # Fallback to greedy
        path, used = [0], {0}
        forbidden_mask = 1
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if not (forbidden_mask >> v) & 1:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            prev = cur
            forbidden_mask |= (1 << nxt)
            for b2 in range(N):
                x = nxt ^ (1 << b2)
                if x != prev:
                    forbidden_mask |= (1 << x)
            for b2 in range(N):
                x = cur ^ (1 << b2)
                if x != nxt:
                    forbidden_mask |= (1 << x)
        best_path = path
    
    return best_path
